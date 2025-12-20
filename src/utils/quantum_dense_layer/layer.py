"""
Quantum Neural Network (QNN) Dense Layer - PyTorch Implementation

Implements a quantum dense layer with configurable embedding methods and depth.
Supports amplitude embedding and rotation embedding for input encoding.
Migrated from TensorFlow to PyTorch with PennyLane.
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


# We avoid creating a global device because the optimal wire count depends on the
# actual input dimension and embedding. A global 12-wire device forces simulation
# of a 2^12 statevector even when fewer qubits are needed.
N_QUBITS_MAX = 12


def get_quantum_device(n_qubits: int):
    """Get the best available PennyLane device for the current hardware."""
    # Prefer lightning.gpu when available; otherwise use lightning.qubit CPU.
    if torch.cuda.is_available():
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits)
            return dev, "cuda"
        except Exception:
            pass

    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        device_type = (
            "mps"
            if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            else "cpu"
        )
        return dev, device_type
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        return dev, "cpu"


class QuantumDenseLayer(nn.Module):
    """
    Quantum Neural Network (QNN) dense layer with configurable embedding.

    Transforms input from (batch_size, m) to (batch_size, n) where both m and n <= 2^12.
    Supports two embedding methods:
    - 'amplitude': Amplitude embedding (requires m to be power of 2 up to 2^12)
    - 'rotation': Rotation embedding (input divided into 12 patches and averaged)

    Args:
        output_dim: Output dimension n (must be <= 2^12 = 4096)
        embedding: Embedding method ('amplitude' or 'rotation', default: 'amplitude')
        depth: Number of quantum layers (default: 1)
    """

    def __init__(self, output_dim, embedding='amplitude', depth=1):
        super().__init__()

        if output_dim <= 0 or output_dim > 4096:  # 2^12
            raise ValueError(f"output_dim must be > 0 and <= 4096 (2^12), got {output_dim}")

        if embedding not in ['amplitude', 'rotation']:
            raise ValueError(f"embedding must be 'amplitude' or 'rotation', got {embedding}")

        self.output_dim = output_dim
        self.embedding = embedding
        self.depth = depth
        self.n_qubits = N_QUBITS_MAX
        self.m = None
        self.n_qubits_input = None

    def build(self, input_dim):
        """Build the layer and initialize quantum weights."""
        m = input_dim

        if m > 4096:  # 2^12
            raise ValueError(f"input dimension must be <= 4096 (2^12), got {m}")

        self.m = m

        # For amplitude embedding, use only the required number of qubits.
        if self.embedding == "amplitude":
            self.n_qubits_input = max(1, int(np.ceil(np.log2(max(m, 2)))))
            if self.n_qubits_input > N_QUBITS_MAX:
                raise ValueError(
                    f"Input dimension {m} requires {self.n_qubits_input} qubits, "
                    f"but only {N_QUBITS_MAX} available"
                )
        else:
            # Rotation embedding always uses the maximum number of qubits
            self.n_qubits_input = N_QUBITS_MAX

        # Quantum weights: (depth, n_qubits, 3) for Rot gates
        # Force float32 for MPS compatibility
        self.quantum_weights = nn.Parameter(
            torch.rand(self.depth, self.n_qubits_input, 3, dtype=torch.float32) * 2 * np.pi
        )

        # Create a device with the exact number of wires and cache the QNode.
        # This avoids re-creating a QNode inside every forward() call.
        self._pl_device, self._quantum_device_type = get_quantum_device(self.n_qubits_input)
        self._qnode = None

    def _prepare_rotation_input(self, x):
        """
        Convert input tensor to 12-dimensional vector by dividing into patches.

        Args:
            x: Input tensor of shape (batch_size, m)

        Returns:
            Tensor of shape (batch_size, 12)
        """
        batch_size = x.shape[0]

        # Calculate patch size
        patch_size = self.m // 12
        remainder = self.m % 12

        patches = []
        start_idx = 0

        for i in range(12):
            # Distribute remainder across first patches
            current_patch_size = patch_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_patch_size

            # Extract patch and compute mean
            patch = x[:, start_idx:end_idx]
            patch_mean = torch.mean(patch, dim=1, keepdim=True)
            patches.append(patch_mean)

            start_idx = end_idx

        # Concatenate all patch means
        result = torch.cat(patches, dim=1)  # (batch_size, 12)
        return result

    def _amplitude_embedding(self, x):
        """Amplitude embedding: encode classical data into quantum state amplitudes.

        Notes:
            We keep the features as a Torch tensor so that, when using
            `lightning.gpu`, we do not force a CPU/Numpy conversion.

        Args:
            x: 1D tensor-like of shape (m,)
        """
        # Ensure torch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Pad/trim to nearest power-of-2 state size
        n_states = 2 ** self.n_qubits_input
        if x.numel() < n_states:
            pad = torch.zeros(n_states - x.numel(), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        elif x.numel() > n_states:
            x = x[:n_states]

        # Normalize to unit vector (avoid division by 0)
        norm = torch.linalg.norm(x)
        x = torch.where(norm > 1e-10, x / norm, torch.ones_like(x) / (n_states ** 0.5))

        qml.AmplitudeEmbedding(features=x, wires=range(self.n_qubits_input), normalize=False, pad_with=0.0)

    def _rotation_embedding(self, x):
        """
        Rotation embedding: encode classical data using rotation gates.

        Args:
            x: Input vector of length 12
        """
        for i in range(self.n_qubits_input):
            qml.RY(x[i] * np.pi, wires=i)

    def _get_qnode(self):
        """Create (and cache) a QNode.

        Important performance note:
            Returning `qml.probs` creates a very large output and often prevents
            using fast differentiation methods on GPU. For classification we
            instead return a small vector of expectation values.
        """
        if self._qnode is not None:
            return self._qnode

        # Decide measurement strategy.
        # - Prefer expvals (fast + GPU friendly) when output_dim is small.
        # - Fallback to probs only when output_dim is too large.
        max_expval_outputs = 3 * self.n_qubits_input
        use_probs = self.output_dim > max_expval_outputs

        # Choose diff method candidates.
        # For lightning.gpu, adjoint is usually fastest for expval circuits.
        # For probs circuits, adjoint is often unsupported -> fallback.
        if self._quantum_device_type == "cuda":
            diff_candidates = ["adjoint", "backprop", "parameter-shift"]
        else:
            diff_candidates = ["parameter-shift"]

        def build_qnode(diff_method: str):
            @qml.qnode(self._pl_device, interface="torch", diff_method=diff_method)
            def circuit(x, w):
                # Encoding layer
                if self.embedding == "amplitude":
                    self._amplitude_embedding(x)
                else:
                    self._rotation_embedding(x)

                # Variational quantum layers
                for layer in range(self.depth):
                    for i in range(self.n_qubits_input):
                        qml.Rot(w[layer, i, 0], w[layer, i, 1], w[layer, i, 2], wires=i)

                    for i in range(self.n_qubits_input):
                        for j in range(i + 1, self.n_qubits_input):
                            qml.CNOT(wires=[i, j])

                if use_probs:
                    return qml.probs(wires=range(self.n_qubits_input))

                # Build <= output_dim expectation values.
                outs = []
                for out_idx in range(self.output_dim):
                    wire = out_idx // 3
                    axis = out_idx % 3
                    if axis == 0:
                        obs = qml.PauliX(wire)
                    elif axis == 1:
                        obs = qml.PauliY(wire)
                    else:
                        obs = qml.PauliZ(wire)
                    outs.append(qml.expval(obs))
                return tuple(outs)

            return circuit

        # Build with a candidate diff method.
        self._qnode = build_qnode(diff_candidates[0])
        self._qnode_diff_candidates = diff_candidates
        self._qnode_use_probs = use_probs
        return self._qnode

    def forward(self, inputs):
        """Forward pass through quantum dense layer."""
        batch_size = inputs.shape[0]
        original_device = inputs.device

        # Initialize weights if not done yet
        if not hasattr(self, "quantum_weights"):
            self.build(inputs.shape[1])

        # Prepare input based on embedding type
        if self.embedding == "rotation":
            processed_input = torch.tanh(self._prepare_rotation_input(inputs))
        else:
            processed_input = inputs

        # Move tensors to appropriate device for quantum processing
        quantum_device = torch.device("cuda") if self._quantum_device_type == "cuda" else torch.device("cpu")
        processed_input_quantum = processed_input.to(quantum_device)
        weights_quantum = self.quantum_weights.to(quantum_device)

        circuit = self._get_qnode()

        # Run circuit for each sample in batch.
        # (Still Python-looped; next optimization would be batching/vmap.)
        results = []

        for b_idx in range(batch_size):
            try:
                out = circuit(processed_input_quantum[b_idx], weights_quantum)
            except Exception as e:
                # If the selected diff method is unsupported (common with probs+adjoint),
                # rebuild the QNode with a fallback diff method and retry.
                msg = str(e)
                if (
                    self._quantum_device_type == "cuda"
                    and hasattr(self, "_qnode_diff_candidates")
                    and "does not support adjoint" in msg
                    and len(self._qnode_diff_candidates) > 1
                ):
                    # rebuild with next candidate
                    next_diff = self._qnode_diff_candidates[1]

                    @qml.qnode(self._pl_device, interface="torch", diff_method=next_diff)
                    def fallback_circuit(x, w):
                        if self.embedding == "amplitude":
                            self._amplitude_embedding(x)
                        else:
                            self._rotation_embedding(x)

                        for layer in range(self.depth):
                            for i in range(self.n_qubits_input):
                                qml.Rot(w[layer, i, 0], w[layer, i, 1], w[layer, i, 2], wires=i)
                            for i in range(self.n_qubits_input):
                                for j in range(i + 1, self.n_qubits_input):
                                    qml.CNOT(wires=[i, j])

                        if getattr(self, "_qnode_use_probs", False):
                            return qml.probs(wires=range(self.n_qubits_input))

                        outs = []
                        for out_idx in range(self.output_dim):
                            wire = out_idx // 3
                            axis = out_idx % 3
                            if axis == 0:
                                obs = qml.PauliX(wire)
                            elif axis == 1:
                                obs = qml.PauliY(wire)
                            else:
                                obs = qml.PauliZ(wire)
                            outs.append(qml.expval(obs))
                        return tuple(outs)

                    self._qnode = fallback_circuit
                    circuit = self._qnode
                    out = circuit(processed_input_quantum[b_idx], weights_quantum)
                else:
                    raise

            # `out` can be a tensor (probs) or a tuple of scalars (expvals)
            if isinstance(out, tuple):
                out = torch.stack(list(out))
            else:
                out = out[: self.output_dim]

            results.append(out)

        output = torch.stack(results, dim=0)
        output = output.to(device=original_device, dtype=torch.float32)
        return output


def create_quantum_mlp(input_dim, output_dim, hidden_dims=None,
                       quantum_depth=1, embedding='amplitude'):
    """
    Create a Multi-Layer Perceptron with quantum dense layers.

    Args:
        input_dim: Input dimension (must be <= 4096)
        output_dim: Output dimension for final layer
        hidden_dims: List of hidden layer dimensions (quantum layers)
        quantum_depth: Depth of quantum circuits
        embedding: Embedding method ('amplitude' or 'rotation')

    Returns:
        nn.Module model
    """
    if hidden_dims is None:
        hidden_dims = [128, 64]

    class QuantumMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()

            # Hidden quantum layers
            for hidden_dim in hidden_dims:
                self.layers.append(
                    QuantumDenseLayer(
                        output_dim=hidden_dim,
                        embedding=embedding,
                        depth=quantum_depth
                    )
                )

            # Output layer (classical)
            last_hidden = hidden_dims[-1] if hidden_dims else input_dim
            self.output_layer = nn.Linear(last_hidden, output_dim)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
                x = torch.relu(x)

            # Output layer
            x = self.output_layer(x)
            return x

    return QuantumMLP()
