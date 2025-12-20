from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

import pennylane as qml


Architecture = Literal["all_to_all", "ring"]
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
    for name in ["default.qubit.torch", "default.qubit"]:
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
    - Architecture: "all_to_all" or "ring" entanglement.
    - depth: number of repeated blocks.
    - output_dim: number of Z expectation values returned (first output_dim wires).

    This module uses PennyLane with a QNode configured for the PyTorch interface.
    """

    n_qubits: int = 12

    def __init__(
        self,
        *,
        output_dim: int,
        embedding: Embedding = "amplitude",
        architecture: Architecture = "all_to_all",
        depth: int = 1,
    ) -> None:
        super().__init__()

        if not (1 <= output_dim <= self.n_qubits):
            raise ValueError(f"output_dim must be in [1, {self.n_qubits}] but got {output_dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1 but got {depth}")

        self.output_dim = int(output_dim)
        self.embedding = str(embedding).lower()
        self.architecture = str(architecture).lower().replace("-", "_")
        self.depth = int(depth)

        if self.embedding not in {"amplitude", "rotation"}:
            raise ValueError(f"Unknown embedding: {embedding}")
        if self.architecture not in {"all_to_all", "ring"}:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Trainable parameters: one RY per wire per layer.
        self.theta = nn.Parameter(0.01 * torch.randn(self.depth, self.n_qubits))

        # Device + QNode
        spec = _pick_device(self.n_qubits)
        self._quantum_device_type = spec.name
        self._dev = qml.device(spec.name, **spec.kwargs)

        @qml.qnode(self._dev, interface="torch", diff_method="backprop")
        def _circuit_single(encoded_inputs: torch.Tensor, theta: torch.Tensor):
            # encoded_inputs has shape (4096,) for amplitude, or (12,) for rotation.

            if self.embedding == "amplitude":
                qml.AmplitudeEmbedding(features=encoded_inputs, wires=range(self.n_qubits), normalize=False)
            else:
                for w in range(self.n_qubits):
                    qml.RY(encoded_inputs[w], wires=w)

            for d in range(self.depth):
                # Local trainable rotations
                for w in range(self.n_qubits):
                    qml.RY(theta[d, w], wires=w)

                # Entangling pattern
                if self.architecture == "ring":
                    for w in range(self.n_qubits):
                        qml.CNOT(wires=[w, (w + 1) % self.n_qubits])
                else:  # all_to_all
                    for i in range(self.n_qubits):
                        for j in range(i + 1, self.n_qubits):
                            qml.CNOT(wires=[i, j])

            return [qml.expval(qml.PauliZ(w)) for w in range(self.output_dim)]

        self._qnode_single = _circuit_single

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

        def _eval_one(v: torch.Tensor) -> torch.Tensor:
            res = self._qnode_single(v, self.theta)
            if isinstance(res, (tuple, list)):
                res = torch.stack(list(res), dim=-1)
            return res

        # Vectorize across the batch with torch.func.vmap when available.
        try:
            from torch.func import vmap

            out = vmap(_eval_one)(encoded)
        except Exception:
            out = torch.stack([_eval_one(encoded[i]) for i in range(encoded.shape[0])], dim=0)

        # PennyLane default.qubit often returns float64 expvals; cast back to input dtype
        # for smoother integration with torch modules/losses.
        return out.to(dtype=x.dtype)
