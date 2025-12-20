"""
Quantum Neural Network (QNN) Pooling Layer - PyTorch Implementation

Uses all-to-all connectivity in the quantum circuit for maximum entanglement.
Migrated from TensorFlow to PyTorch with PennyLane.
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from tqdm import tqdm
import time


# Define quantum device globally with optimized backend
N_QUBITS = 4

# Determine best quantum device based on available hardware
def get_quantum_device(n_qubits):
    """Get the best available quantum device for the current hardware."""
    try:
        # Check if CUDA is available (e.g., V100 on cloud)
        if torch.cuda.is_available():
            try:
                dev = qml.device('lightning.gpu', wires=n_qubits)
                print(f"QuantumPoolingLayer: Using lightning.gpu device (CUDA GPU detected)")
                return dev, 'cuda'
            except:
                print("QuantumPoolingLayer: lightning.gpu not available, falling back to CPU")

        # For MPS or CPU, use lightning.qubit (CPU-optimized)
        dev = qml.device('lightning.qubit', wires=n_qubits)
        device_type = 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
        print(f"QuantumPoolingLayer: Using lightning.qubit device (detected {device_type})")
        return dev, device_type
    except:
        # Ultimate fallback
        dev = qml.device('default.qubit', wires=n_qubits)
        print("QuantumPoolingLayer: Using default.qubit device")
        return dev, 'cpu'

dev, QUANTUM_DEVICE_TYPE = get_quantum_device(N_QUBITS)


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
        # Force float32 for MPS compatibility
        self.quantum_weights = nn.Parameter(
            torch.rand(m, self.depth, self.n_qubits, 3, dtype=torch.float32) * 2 * np.pi
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
        start_time = time.time()

        # inputs shape: (batch, n, n, m) - channel last format
        # Convert to channel first for PyTorch: (batch, m, n, n)
        x = inputs.permute(0, 3, 1, 2)

        batch_size = x.shape[0]
        original_device = inputs.device  # Remember the original device

        # Initialize weights if not done yet
        if not hasattr(self, 'quantum_weights'):
            self.build(inputs.shape)

        # Step 1: Pool to (batch, m, 2, 2)
        pool_start = time.time()
        pooled = self._adaptive_pool_to_2x2(x)
        pool_time = time.time() - pool_start

        # Step 2: Process each channel with quantum circuit (BATCHED)
        # Reshape pooled to (batch * m, 4) for batch processing
        pooled_flat = pooled.reshape(batch_size, self.m, 4)  # (batch, m, 4)
        normalized = torch.tanh(pooled_flat)  # (batch, m, 4)

        # Move tensors to appropriate device for quantum processing
        # lightning.gpu works with CUDA tensors, lightning.qubit needs CPU
        if QUANTUM_DEVICE_TYPE == 'cuda' and torch.cuda.is_available():
            # Keep on CUDA for lightning.gpu
            quantum_device = torch.device('cuda')
        else:
            # Move to CPU for lightning.qubit
            quantum_device = torch.device('cpu')

        normalized_quantum = normalized.to(quantum_device)
        weights_quantum = self.quantum_weights.to(quantum_device)

        # Choose differentiation method based on device
        # parameter-shift is universally compatible but slower
        # adjoint is faster but only works on GPU with certain circuits
        if QUANTUM_DEVICE_TYPE == 'cuda':
            diff_method = 'adjoint'
        else:
            diff_method = 'parameter-shift'  # Universal compatibility

        # Define quantum circuit that processes a single input
        @qml.qnode(dev, interface='torch', diff_method=diff_method)
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

        # Batch process all channels and samples together
        # Flatten to (batch * m, 4) and process in larger batches
        all_inputs = normalized_quantum.reshape(batch_size * self.m, 4)  # (batch*m, 4)
        all_weights = weights_quantum.repeat_interleave(batch_size, dim=0)  # (batch*m, depth, n_qubits, 3)

        # Process quantum circuits
        circuit_start = time.time()
        results = []
        num_circuits = batch_size * self.m

        # Show progress with timing info
        pbar = tqdm(range(num_circuits), desc=f'QuantumPool (batch={batch_size}, channels={self.m})', leave=False)
        for i in pbar:
            iter_start = time.time()
            result = circuit(all_inputs[i], all_weights[i])
            results.append(result)
            iter_time = time.time() - iter_start
            pbar.set_postfix({'circuit_ms': f'{iter_time*1000:.1f}'})

        circuit_time = time.time() - circuit_start

        # Reshape back to (batch, m)
        output = torch.stack(results).reshape(batch_size, self.m)

        # Move output back to original device and ensure correct dtype (float32 for MPS)
        # Force float32 to avoid MPS float64 issues
        output = output.to(device=original_device, dtype=torch.float32)

        total_time = time.time() - start_time

        # Log timing info (only show for first few batches to avoid spam)
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1
        if self._call_count <= 3:
            print(f"\n[QuantumPooling] Total: {total_time:.2f}s | Pool: {pool_time:.3f}s | Circuits: {circuit_time:.2f}s ({num_circuits} circuits @ {circuit_time/num_circuits*1000:.1f}ms each)")

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
