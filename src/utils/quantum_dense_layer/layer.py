from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

import pennylane as qml
from pennylane.templates.layers import SimplifiedTwoDesign, StronglyEntanglingLayers

DenseTemplate = Literal["strong", "two_design"]
Embedding = Literal["amplitude", "rotation"]


@dataclass(frozen=True)
class _DeviceSpec:
    name: str
    kwargs: dict


def _pick_device(n_qubits: int) -> _DeviceSpec:
    """Pick the best available PennyLane device.

    Preference order:
    1) default.qubit.torch (older plugin name; may be unavailable)
    2) default.qubit

    Note: The user requested a PyTorch backend; we always build the QNode with
    interface='torch'. If CUDA is available and inputs/params live on CUDA,
    default.qubit will compute with torch tensors and can run on GPU.
    """

    # In PL 0.43, requesting a non-existing device raises.
    for name in ["default.qubit"]:
        try:
            qml.device(name, wires=n_qubits)
            return _DeviceSpec(name=name, kwargs={"wires": n_qubits})
        except Exception:
            continue

    # As a last resort, let PennyLane raise with a helpful message.
    return _DeviceSpec(name="default.qubit", kwargs={"wires": n_qubits})


def _normalize_amplitudes(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Safe L2-normalization row-wise.

    Ensures we never feed an all-zero vector to amplitude embedding.
    """

    # (batch, dim)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    # For zero rows: set them to |0...0>.
    zero_mask = norm <= eps
    if torch.any(zero_mask):
        x = x.clone()
        x[zero_mask.squeeze(-1), 0] = 1.0
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    return x / norm


def _rotation_pool_to_12(x: torch.Tensor) -> torch.Tensor:
    """Pool (batch, m) -> (batch, 12) using patch averaging.

    Spec: patch_size = m//12 + 1.

    For qubit i, we average x[:, i*patch_size : (i+1)*patch_size] (clipped to m).
    If the slice is empty, we return 0.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x to have shape (batch, m). Got {tuple(x.shape)}")

    b, m = x.shape
    patch_size = m // 12 + 1

    pooled = []
    for i in range(12):
        start = i * patch_size
        end = min((i + 1) * patch_size, m)
        if start >= m:
            pooled.append(torch.zeros((b,), device=x.device, dtype=x.dtype))
        else:
            pooled.append(x[:, start:end].mean(dim=-1))

    return torch.stack(pooled, dim=-1)


class QuantumDenseLayer(nn.Module):
    """A trainable quantum dense layer.

    Input:  x of shape (batch_size, m)
    Output: tensor of shape (batch_size, output_dim)

    - Uses 12 qubits.
    - Embedding: "amplitude" (m<=4096) or "rotation" (pooled to 12 angles).
    - Template: "strong" or "two_design".
    - depth: number of repeated blocks.
    - output_dim: number of Z expectation values returned (first output_dim wires).
    - batch_chunk_size: process batches in chunks of this size (None = auto-detect, 0 = all at once)

    This module uses PennyLane with a QNode configured for the PyTorch interface.
    """

    n_qubits: int = 12

    def __init__(
        self,
        *,
        output_dim: int,
        embedding: Embedding = "amplitude",
        template: DenseTemplate = "strong",
        depth: int = 1,
        batch_chunk_size: int | None = None,
    ) -> None:
        super().__init__()

        if not (1 <= output_dim <= self.n_qubits):
            raise ValueError(f"output_dim must be in [1, {self.n_qubits}] but got {output_dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1 but got {depth}")

        self.output_dim = int(output_dim)
        self.embedding = str(embedding).lower()
        self.template = str(template).lower().replace("-", "_")
        self.depth = int(depth)

        # Auto-detect optimal chunk size based on embedding type
        if batch_chunk_size is None:
            # Amplitude encoding: smaller chunks for better GPU utilization
            # Rotation encoding: larger chunks since it's simpler
            self.batch_chunk_size = 8 if self.embedding == "amplitude" else 32
        else:
            self.batch_chunk_size = batch_chunk_size

        if self.embedding not in {"amplitude", "rotation"}:
            raise ValueError(f"Unknown embedding: {embedding}")
        if self.template not in {"strong", "two_design"}:
            raise ValueError(f"Unknown template: {template}")

        # Trainable parameters depend on the chosen template.
        if self.template == "strong":
            # StronglyEntanglingLayers: (depth, n_wires, 3)
            self.theta = nn.Parameter(0.01 * torch.randn(self.depth, self.n_qubits, 3))
            self.init_theta = None
        else:
            # SimplifiedTwoDesign:
            # - initial_layer_weights: (n_wires,)
            # - weights: (depth, n_wires-1, 2)
            self.init_theta = nn.Parameter(0.01 * torch.randn(self.n_qubits))
            self.theta = nn.Parameter(0.01 * torch.randn(self.depth, self.n_qubits - 1, 2))

        # Device + QNode
        spec = _pick_device(self.n_qubits)
        self._quantum_device_type = spec.name
        self._dev = qml.device(spec.name, **spec.kwargs)

        # Build a separate QNode per template so we don't pass None into the QNode
        # (this tends to be friendlier for tracing/vmap and CUDA execution).
        if self.template == "strong":

            @qml.qnode(self._dev, interface="torch", diff_method="backprop")
            def _circuit_single(encoded_inputs: torch.Tensor, theta: torch.Tensor):
                if self.embedding == "amplitude":
                    qml.AmplitudeEmbedding(features=encoded_inputs, wires=range(self.n_qubits), normalize=False)
                else:
                    for w in range(self.n_qubits):
                        qml.RY(encoded_inputs[w], wires=w)

                StronglyEntanglingLayers(weights=theta, wires=range(self.n_qubits))
                return [qml.expval(qml.PauliZ(w)) for w in range(self.output_dim)]

            self._qnode_single = _circuit_single
            self._qnode_expects_init = False

        else:

            @qml.qnode(self._dev, interface="torch", diff_method="backprop")
            def _circuit_single(encoded_inputs: torch.Tensor, init_theta: torch.Tensor, theta: torch.Tensor):
                if self.embedding == "amplitude":
                    qml.AmplitudeEmbedding(features=encoded_inputs, wires=range(self.n_qubits), normalize=False)
                else:
                    for w in range(self.n_qubits):
                        qml.RY(encoded_inputs[w], wires=w)

                SimplifiedTwoDesign(initial_layer_weights=init_theta, weights=theta, wires=range(self.n_qubits))
                return [qml.expval(qml.PauliZ(w)) for w in range(self.output_dim)]

            self._qnode_single = _circuit_single
            self._qnode_expects_init = True

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x to have shape (batch, m). Got {tuple(x.shape)}")

        if self.embedding == "amplitude":
            b, m = x.shape
            dim = 2 ** self.n_qubits
            if m > dim:
                raise ValueError(f"Amplitude embedding requires m <= {dim}, got m={m}")

            padded = torch.zeros((b, dim), device=x.device, dtype=x.dtype)
            padded[:, :m] = x
            padded = _normalize_amplitudes(padded)
            return padded

        # rotation
        return _rotation_pool_to_12(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(x)
        batch_size = encoded.shape[0]

        def _eval_one(v: torch.Tensor) -> torch.Tensor:
            if self._qnode_expects_init:
                res = self._qnode_single(v, self.init_theta, self.theta)
            else:
                res = self._qnode_single(v, self.theta)
            if isinstance(res, (tuple, list)):
                res = torch.stack(list(res), dim=-1)
            return res

        # Process in chunks for better GPU utilization
        if self.batch_chunk_size > 0 and batch_size > self.batch_chunk_size:
            # Chunked processing
            chunks = []
            for i in range(0, batch_size, self.batch_chunk_size):
                chunk_end = min(i + self.batch_chunk_size, batch_size)
                chunk = encoded[i:chunk_end]

                # Process this chunk with vmap
                try:
                    from torch.func import vmap
                    chunk_out = vmap(_eval_one)(chunk)
                except Exception:
                    chunk_out = torch.stack([_eval_one(chunk[j]) for j in range(chunk.shape[0])], dim=0)

                chunks.append(chunk_out)

            out = torch.cat(chunks, dim=0)
        else:
            # Process all at once (original behavior)
            try:
                from torch.func import vmap
                out = vmap(_eval_one)(encoded)
            except Exception:
                out = torch.stack([_eval_one(encoded[i]) for i in range(encoded.shape[0])], dim=0)

        # PennyLane default.qubit often returns float64 expvals; cast back to input dtype
        # for smoother integration with torch modules/losses.
        return out.to(dtype=x.dtype)
