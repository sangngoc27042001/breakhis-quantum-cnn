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


# Define quantum device globally with 12 qubits (max for 2^12)
N_QUBITS = 12

# Determine best quantum device based on available hardware
def get_quantum_device(n_qubits):
    """Get the best available quantum device for the current hardware."""
    try:
        # Check if CUDA is available (e.g., V100 on cloud)
        if torch.cuda.is_available():
            try:
                dev = qml.device('lightning.gpu', wires=n_qubits)
                # print(f"QuantumDenseLayer: Using lightning.gpu device (CUDA GPU detected)")
                return dev, 'cuda'
            except:
                # print("QuantumDenseLayer: lightning.gpu not available, falling back to CPU")
                pass

        # For MPS or CPU, use lightning.qubit (CPU-optimized)
        dev = qml.device('lightning.qubit', wires=n_qubits)
        device_type = 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
        # print(f"QuantumDenseLayer: Using lightning.qubit device (detected {device_type})")
        return dev, device_type
    except:
        # Ultimate fallback
        dev = qml.device('default.qubit', wires=n_qubits)
        # print("QuantumDenseLayer: Using default.qubit device")
        return dev, 'cpu'

dev, QUANTUM_DEVICE_TYPE = get_quantum_device(N_QUBITS)


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
        self.n_qubits = N_QUBITS
        self.m = None
        self.n_qubits_input = None

    def build(self, input_dim):
        """Build the layer and initialize quantum weights."""
        m = input_dim

        if m > 4096:  # 2^12
            raise ValueError(f"input dimension must be <= 4096 (2^12), got {m}")

        self.m = m

        # For amplitude embedding, check if we have enough qubits
        if self.embedding == 'amplitude':
            # Calculate required qubits for amplitude embedding
            self.n_qubits_input = max(1, int(np.ceil(np.log2(max(m, 2)))))
            if self.n_qubits_input > N_QUBITS:
                raise ValueError(
                    f"Input dimension {m} requires {self.n_qubits_input} qubits, "
                    f"but only {N_QUBITS} available"
                )
        else:
            # Rotation embedding always uses 12 qubits
            self.n_qubits_input = N_QUBITS

        # Quantum weights: (depth, n_qubits, 3) for Rot gates
        # Force float32 for MPS compatibility
        self.quantum_weights = nn.Parameter(
            torch.rand(self.depth, self.n_qubits_input, 3, dtype=torch.float32) * 2 * np.pi
        )

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
        """
        Amplitude embedding: encode classical data into quantum state amplitudes.

        Args:
            x: Input vector (should be normalized)
        """
        # Convert to numpy for PennyLane
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # Pad to nearest power of 2 if needed
        n_states = 2 ** self.n_qubits_input
        if len(x) < n_states:
            x = np.pad(x, (0, n_states - len(x)), mode='constant')
        elif len(x) > n_states:
            x = x[:n_states]

        # Normalize to unit vector
        norm = np.linalg.norm(x)
        if norm > 1e-10:
            x = x / norm
        else:
            # If norm is too small, use uniform distribution
            x = np.ones(n_states) / np.sqrt(n_states)

        # Use pad_with parameter to ensure correct length
        qml.AmplitudeEmbedding(features=x, wires=range(self.n_qubits_input), normalize=True, pad_with=0.0)

    def _rotation_embedding(self, x):
        """
        Rotation embedding: encode classical data using rotation gates.

        Args:
            x: Input vector of length 12
        """
        for i in range(self.n_qubits_input):
            qml.RY(x[i] * np.pi, wires=i)

    def forward(self, inputs):
        """Forward pass through quantum dense layer."""
        batch_size = inputs.shape[0]
        original_device = inputs.device  # Remember the original device

        # Initialize weights if not done yet
        if not hasattr(self, 'quantum_weights'):
            self.build(inputs.shape[1])

        # Prepare input based on embedding type
        if self.embedding == 'rotation':
            processed_input = self._prepare_rotation_input(inputs)
            # Normalize to [-1, 1] range for rotation encoding
            processed_input = torch.tanh(processed_input)
        else:
            # For amplitude embedding, use input as-is
            processed_input = inputs

        # Move tensors to appropriate device for quantum processing
        # lightning.gpu works with CUDA tensors, lightning.qubit needs CPU
        if QUANTUM_DEVICE_TYPE == 'cuda' and torch.cuda.is_available():
            # Keep on CUDA for lightning.gpu
            quantum_device = torch.device('cuda')
        else:
            # Move to CPU for lightning.qubit
            quantum_device = torch.device('cpu')

        processed_input_quantum = processed_input.to(quantum_device)
        weights_quantum = self.quantum_weights.to(quantum_device)

        # Choose differentiation method based on device
        # parameter-shift is universally compatible but slower
        # adjoint is faster but only works on GPU with certain circuits
        if QUANTUM_DEVICE_TYPE == 'cuda':
            diff_method = 'adjoint'
        else:
            diff_method = 'parameter-shift'  # Universal compatibility

        # Define quantum circuit
        @qml.qnode(dev, interface='torch', diff_method=diff_method)
        def circuit(x, w):
            # Encoding layer
            if self.embedding == 'amplitude':
                self._amplitude_embedding(x)
            else:
                self._rotation_embedding(x)

            # Variational quantum layers
            for layer in range(self.depth):
                # Rotation layer - apply to each qubit
                for i in range(self.n_qubits_input):
                    qml.Rot(w[layer, i, 0], w[layer, i, 1], w[layer, i, 2], wires=i)

                # All-to-all entangling layer
                for i in range(self.n_qubits_input):
                    for j in range(i + 1, self.n_qubits_input):
                        qml.CNOT(wires=[i, j])

            # Measure probabilities of all computational basis states
            return qml.probs(wires=range(self.n_qubits_input))

        # Run circuit for each sample in batch
        results = []

        for b_idx in range(batch_size):
            probs = circuit(processed_input_quantum[b_idx], weights_quantum)

            # Extract first n probabilities
            probs_n = probs[:self.output_dim]

            # Renormalize to ensure probabilities sum to 1
            prob_sum = torch.sum(probs_n)
            if prob_sum > 1e-10:
                probs_n = probs_n / prob_sum
            else:
                # If sum is too small, use uniform distribution
                probs_n = torch.ones(self.output_dim) / self.output_dim

            results.append(probs_n)

        # Stack to tensor: (batch, output_dim)
        output = torch.stack(results, dim=0)

        # Move output back to original device and ensure correct dtype (float32 for MPS)
        # Force float32 to avoid MPS float64 issues
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
