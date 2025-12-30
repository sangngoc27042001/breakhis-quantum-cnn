from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

import pennylane as qml
from pennylane.templates.layers import SimplifiedTwoDesign, StronglyEntanglingLayers, BasicEntanglerLayers

DenseTemplate = Literal["strong", "two_design", "basic"]
Embedding = Literal["amplitude", "rotation"]


@dataclass(frozen=True)
class _DeviceSpec:
    name: str
    kwargs: dict


def _pick_device(n_qubits: int) -> _DeviceSpec:
    """Pick the best available PennyLane device.

    Uses default.qubit which supports backprop differentiation with PyTorch.
    This is the most compatible option for training with gradient descent.
    """

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


def _rotation_pool(x: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Pool (batch, m) -> (batch, n_qubits) using patch averaging.

    Spec: patch_size = m//n_qubits + 1.

    For qubit i, we average x[:, i*patch_size : (i+1)*patch_size] (clipped to m).
    If the slice is empty, we return 0.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x to have shape (batch, m). Got {tuple(x.shape)}")

    b, m = x.shape
    patch_size = m // n_qubits + 1

    pooled = []
    for i in range(n_qubits):
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

    - Uses configurable number of qubits (default 12).
    - Embedding: "amplitude" (m<=2^n_qubits) or "rotation" (pooled to n_qubits angles).
    - Template: "strong" or "two_design" or "basic".
    - depth: number of repeated blocks.
    - output_dim: number of Z expectation values returned (first output_dim wires).

    This module uses PennyLane with a QNode configured for the PyTorch interface.
    """

    def __init__(
        self,
        *,
        output_dim: int,
        n_qubits: int = 12,
        embedding: Embedding = "amplitude",
        template: DenseTemplate = "strong",
        depth: int = 1,
    ) -> None:
        super().__init__()

        self.n_qubits = int(n_qubits)
        if self.n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1 but got {n_qubits}")

        if not (1 <= output_dim <= self.n_qubits):
            raise ValueError(f"output_dim must be in [1, {self.n_qubits}] but got {output_dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1 but got {depth}")

        self.output_dim = int(output_dim)
        self.embedding = str(embedding).lower()
        self.template = str(template).lower().replace("-", "_")
        self.depth = int(depth)

        if self.embedding not in {"amplitude", "rotation"}:
            raise ValueError(f"Unknown embedding: {embedding}")
        if self.template not in {"strong", "two_design", "basic"}:
            raise ValueError(f"Unknown template: {template}")

        # Trainable parameters depend on the chosen template.
        if self.template == "strong":
            # StronglyEntanglingLayers: (depth, n_wires, 3)
            self.theta = nn.Parameter(0.01 * torch.randn(self.depth, self.n_qubits, 3))
            self.init_theta = None
        elif self.template == "two_design":
            # SimplifiedTwoDesign:
            # - initial_layer_weights: (n_wires,)
            # - weights: (depth, n_wires-1, 2)
            self.init_theta = nn.Parameter(0.01 * torch.randn(self.n_qubits))
            self.theta = nn.Parameter(0.01 * torch.randn(self.depth, self.n_qubits - 1, 2))
        else:  # basic
            # BasicEntanglerLayers: (depth, n_wires)
            self.theta = nn.Parameter(0.01 * torch.randn(self.depth, self.n_qubits))
            self.init_theta = None

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
                    qml.AmplitudeEmbedding(features=encoded_inputs, wires=range(self.n_qubits), normalize=True)
                else:
                    for w in range(self.n_qubits):
                        qml.Hadamard(wires=w)
                    for w in range(self.n_qubits):
                        qml.RY(encoded_inputs[w], wires=w)

                StronglyEntanglingLayers(weights=theta, wires=range(self.n_qubits))
                return qml.probs(wires=range(self.n_qubits))

            self._qnode_single = _circuit_single
            self._qnode_expects_init = False

        elif self.template == "two_design":

            @qml.qnode(self._dev, interface="torch", diff_method="backprop")
            def _circuit_single(encoded_inputs: torch.Tensor, init_theta: torch.Tensor, theta: torch.Tensor):
                if self.embedding == "amplitude":
                    qml.AmplitudeEmbedding(features=encoded_inputs, wires=range(self.n_qubits), normalize=True)
                else:
                    for w in range(self.n_qubits):
                        qml.Hadamard(wires=w)
                    for w in range(self.n_qubits):
                        qml.RY(encoded_inputs[w], wires=w)

                SimplifiedTwoDesign(initial_layer_weights=init_theta, weights=theta, wires=range(self.n_qubits))
                return qml.probs(wires=range(self.n_qubits))

            self._qnode_single = _circuit_single
            self._qnode_expects_init = True

        else:  # basic

            @qml.qnode(self._dev, interface="torch", diff_method="backprop")
            def _circuit_single(encoded_inputs: torch.Tensor, theta: torch.Tensor):
                if self.embedding == "amplitude":
                    qml.AmplitudeEmbedding(features=encoded_inputs, wires=range(self.n_qubits), normalize=True)
                else:
                    for w in range(self.n_qubits):
                        qml.Hadamard(wires=w)
                    for w in range(self.n_qubits):
                        qml.RY(encoded_inputs[w], wires=w)

                BasicEntanglerLayers(weights=theta, wires=range(self.n_qubits))
                return qml.probs(wires=range(self.n_qubits))

            self._qnode_single = _circuit_single
            self._qnode_expects_init = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x to have shape (batch, m). Got {tuple(x.shape)}")

        if self.embedding == "amplitude":
            b, m = x.shape
            dim = 2 ** self.n_qubits
            if m > dim:
                raise ValueError(f"Amplitude embedding requires m <= {dim}, got m={m}")

            # Just pad - let PennyLane's AmplitudeEmbedding handle normalization
            padded = torch.zeros((b, dim), device=x.device, dtype=x.dtype)
            padded[:, :m] = x
            return padded

        # rotation
        return _rotation_pool(x, self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(x)

        def _eval_one(v: torch.Tensor) -> torch.Tensor:
            if self._qnode_expects_init:
                res = self._qnode_single(v, self.init_theta, self.theta)
            else:
                res = self._qnode_single(v, self.theta)
            if isinstance(res, (tuple, list)):
                res = torch.stack(list(res), dim=-1)
            # Slice to get only the first output_dim probabilities
            return res[:self.output_dim]

        # Vectorize across the batch with torch.func.vmap when available.
        try:
            from torch.func import vmap
            out = vmap(_eval_one)(encoded)
        except Exception:
            out = torch.stack([_eval_one(encoded[i]) for i in range(encoded.shape[0])], dim=0)

        # PennyLane default.qubit often returns float64 expvals; cast back to input dtype
        # for smoother integration with torch modules/losses.
        # Scale probabilities from [0, 1] to [-1, 1]
        out = 2 * out - 1
        return out.to(dtype=x.dtype)
