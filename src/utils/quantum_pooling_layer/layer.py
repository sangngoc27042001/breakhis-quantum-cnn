"""
Quantum Neural Network (QNN) Pooling Layer - PyTorch Implementation

Uses all-to-all connectivity in the quantum circuit for maximum entanglement.
Migrated from TensorFlow to PyTorch with PennyLane.
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


# Define quantum device globally
N_QUBITS = 4
dev = qml.device('default.qubit', wires=N_QUBITS)


class QuantumPoolingLayer(nn.Module):
    """
    Quantum Neural Network (QNN) pooling layer with all-to-all connectivity.

    Uses all-to-all CNOT gates for maximum entanglement between qubits.
    Compatible with PyTorch's autograd.

    Args:
        depth: Number of quantum layers
        n_qubits: Number of qubits (default: 4)
    """

    def __init__(self, depth=1, n_qubits=4):
        super().__init__()
        self.depth = depth
        self.n_qubits = n_qubits
        self.n = None
        self.m = None

    def build(self, input_shape):
        """Initialize quantum weights based on input shape."""
        _, n, n_check, m = input_shape

        if n != n_check:
            raise ValueError(f"Expected square dims, got {n}x{n_check}")

        self.n = n
        self.m = m

        # Quantum weights: (m, depth, n_qubits, 3)
        # Using nn.Parameter to make it trainable
        self.quantum_weights = nn.Parameter(
            torch.rand(m, self.depth, self.n_qubits, 3) * 2 * np.pi
        )

    def _adaptive_pool_to_2x2(self, x):
        """Pool to 2x2 using adaptive pooling."""
        # x shape: (batch, m, n, n)
        # AdaptiveAvgPool2d expects (batch, channels, height, width)

        # MPS has limitations with adaptive pooling when sizes aren't divisible
        # Move to CPU for this operation if on MPS
        device = x.device
        if device.type == 'mps':
            x_cpu = x.cpu()
            pooled = torch.nn.functional.adaptive_avg_pool2d(x_cpu, (2, 2))
            pooled = pooled.to(device)
        else:
            pooled = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))

        return pooled

    def forward(self, inputs):
        """Forward pass through quantum pooling layer."""
        # inputs shape: (batch, n, n, m) - channel last format
        # Convert to channel first for PyTorch: (batch, m, n, n)
        x = inputs.permute(0, 3, 1, 2)

        batch_size = x.shape[0]

        # Initialize weights if not done yet
        if not hasattr(self, 'quantum_weights'):
            self.build(inputs.shape)

        # Step 1: Pool to (batch, m, 2, 2)
        pooled = self._adaptive_pool_to_2x2(x)

        # Step 2: Process each channel with quantum circuit
        results = []

        for ch_idx in range(self.m):
            # Get 2x2 patch for this channel
            patch = pooled[:, ch_idx, :, :]  # (batch, 2, 2)
            flat = patch.reshape(batch_size, 4)  # (batch, 4)
            normalized = torch.tanh(flat)

            # Get weights for this channel
            weights = self.quantum_weights[ch_idx]

            # Define quantum circuit
            @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
            def circuit(x, w):
                # Angle encoding
                for i in range(self.n_qubits):
                    qml.RY(x[i] * np.pi, wires=i)

                # Quantum layers with all-to-all connectivity
                for layer in range(self.depth):
                    # Rotation layer - apply to each qubit
                    for i in range(self.n_qubits):
                        qml.Rot(w[layer, i, 0], w[layer, i, 1], w[layer, i, 2], wires=i)

                    # All-to-all entangling layer
                    for i in range(self.n_qubits):
                        for j in range(i + 1, self.n_qubits):
                            qml.CNOT(wires=[i, j])

                return qml.expval(qml.PauliZ(0))

            # Run circuit for each sample in batch
            ch_results = []
            for b_idx in range(batch_size):
                result = circuit(normalized[b_idx], weights)
                ch_results.append(result)

            results.append(torch.stack(ch_results))

        # Stack to tensor: (batch, m)
        output = torch.stack(results, dim=1)

        # Ensure output is float32 to match input dtype
        if output.dtype != inputs.dtype:
            output = output.to(inputs.dtype)

        return output


def create_simple_cnn(input_shape, num_classes, quantum_depth=1):
    """Create a simple CNN with quantum pooling."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(input_shape[2], 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool1 = nn.MaxPool2d(2)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
            self.quantum_pool = QuantumPoolingLayer(depth=quantum_depth)
            self.fc1 = nn.Linear(128, 64)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            # Classical layers
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool1(x)
            x = torch.relu(self.conv3(x))
            x = torch.relu(self.conv4(x))

            # Convert to channel-last for quantum layer
            x = x.permute(0, 2, 3, 1)

            # Quantum pooling
            x = self.quantum_pool(x)

            # Classification
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

            return x

    return SimpleCNN()
