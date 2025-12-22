from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

import pennylane as qml

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


class QuantumRingRotationLayer(nn.Module):
    """A trainable quantum ring rotation layer with custom architecture.

    Input:  x of shape (batch_size, m)
    Output: tensor of shape (batch_size, output_dim)

    - Uses configurable number of qubits (default 12).
    - Embedding: "amplitude" (m<=2^n_qubits) or "rotation" (pooled to n_qubits angles).
    - Custom ring rotation architecture:
        * Each layer applies RX, RY, RZ rotations to each qubit
        * Ring entanglement: CNOT gates connecting qubits in a ring (i -> i+1, last -> first)
    - depth: number of repeated rotation + entanglement blocks.
    - output_dim: number of Z expectation values returned (first output_dim wires).

    This module uses PennyLane with a QNode configured for the PyTorch interface.
    """

    def __init__(
        self,
        *,
        output_dim: int,
        n_qubits: int = 12,
        embedding: Embedding = "amplitude",
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
        self.depth = int(depth)

        if self.embedding not in {"amplitude", "rotation"}:
            raise ValueError(f"Unknown embedding: {embedding}")

        # Trainable parameters for custom ring rotation architecture
        # For each depth layer, we have 3 rotation angles (RX, RY, RZ) per qubit
        # Shape: (depth, n_qubits, 3) for [RX, RY, RZ] angles
        self.theta = nn.Parameter(0.01 * torch.randn(self.depth, self.n_qubits, 3))

        # Device + QNode
        spec = _pick_device(self.n_qubits)
        self._quantum_device_type = spec.name
        self._dev = qml.device(spec.name, **spec.kwargs)

        # Build custom ring rotation circuit
        @qml.qnode(self._dev, interface="torch", diff_method="backprop")
        def _circuit_single(encoded_inputs: torch.Tensor, theta: torch.Tensor):
            # Data embedding
            if self.embedding == "amplitude":
                qml.AmplitudeEmbedding(features=encoded_inputs, wires=range(self.n_qubits), normalize=True)
            else:
                for w in range(self.n_qubits):
                    qml.RY(encoded_inputs[w], wires=w)

            # Custom ring rotation layers
            for d in range(self.depth):
                # Rotation layer: apply RX, RY, RZ to each qubit
                for w in range(self.n_qubits):
                    qml.RX(theta[d, w, 0], wires=w)
                    qml.RY(theta[d, w, 1], wires=w)
                    qml.RZ(theta[d, w, 2], wires=w)

                # Ring entanglement: CNOT gates in a ring topology
                for w in range(self.n_qubits):
                    control = w
                    target = (w + 1) % self.n_qubits  # Ring: last qubit connects back to first
                    qml.CNOT(wires=[control, target])

            # Measurements
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

            # Just pad - let PennyLane's AmplitudeEmbedding handle normalization
            padded = torch.zeros((b, dim), device=x.device, dtype=x.dtype)
            padded[:, :m] = x
            return padded

        # rotation
        return _rotation_pool(x, self.n_qubits)

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
