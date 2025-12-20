# Quantum Neural Network (QNN) Pooling Layer for TensorFlow CNN

A quantum pooling layer using **all-to-all connectivity** for maximum entanglement. Uses PennyLane to process CNN feature maps through parameterized quantum circuits.

## Overview

The QNN pooling layer:
1. **Adaptive pooling**: Reduces `(batch, n, n, m)` → `(batch, 2, 2, m)`
   - For even n: uses (n/2) × (n/2) average pooling
   - For odd n: uses ((n+1)/2) × ((n+1)/2) average pooling

2. **Quantum processing**: Applies a 4-qubit QNN circuit to each 2×2 patch
   - Angle encoding with RY gates
   - Parameterized rotation layers (Rot gates)
   - **All-to-all entanglement** with CNOT gates (6 CNOTs per layer)
   - Measures ⟨Z₀⟩ expectation value

3. **Output**: Returns `(batch, m)` tensor

## Files

- **[layer.py](layer.py)** - Main implementation (QNN with all-to-all connectivity)
- **[tests.py](tests.py)** - Comprehensive test suite (11 tests)
- **[examples.py](examples.py)** - Usage examples
- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference

## Installation

```bash
# PennyLane is already in requirements.txt
uv sync
```

## Quick Start

```python
from src.utils import QuantumPoolingLayer, create_simple_cnn
import tensorflow as tf

# Option 1: Use the layer directly
layer = QuantumPoolingLayer(depth=2)
x = tf.random.normal((4, 8, 8, 64))  # batch, height, width, channels
y = layer(x)  # (4, 64)

# Option 2: Use the pre-built model
model = create_simple_cnn(
    input_shape=(32, 32, 3),
    num_classes=10,
    quantum_depth=2
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train (requires eager execution)
model.fit(x_train, y_train, epochs=10)
```

## Running Tests

```bash
# From project root
uv run python src/utils/test_quantum_simple.py
```

Expected output:
```
============================================================
Testing Simplified Quantum Pooling Layer
============================================================

Test 1: Basic layer
----------------------------------------
Input shape: (2, 4, 4, 3)
Output shape: (2, 3)
✓ Basic test passed!

Test 2: Odd dimension (n=5)
----------------------------------------
Input: (2, 5, 5, 2) -> Output: (2, 2)
✓ Odd dimension test passed!

Test 3: Full CNN model
----------------------------------------
Model test: (2, 16, 16, 3) -> (2, 5)
Output sums (should be ~1.0): [1. 1.]
✓ Model test passed!

ALL TESTS PASSED!
```

## API Reference

### QuantumPoolingLayer

```python
QuantumPoolingLayer(depth=1, n_qubits=4, **kwargs)
```

**Parameters:**
- `depth` (int): Number of quantum layers (default: 1)
- `n_qubits` (int): Number of qubits, should be 4 for 2×2 patches (default: 4)

**Input:** `(batch_size, n, n, m)` where n can be any positive integer

**Output:** `(batch_size, m)`

**Trainable parameters:** `m × depth × n_qubits × 3`
- Example: 128 channels, depth=2 → 3,072 parameters

### create_simple_cnn

```python
create_simple_cnn(input_shape, num_classes, quantum_depth=1)
```

**Parameters:**
- `input_shape` (tuple): Input image shape (height, width, channels)
- `num_classes` (int): Number of output classes
- `quantum_depth` (int): Depth of quantum circuit (default: 1)

**Returns:** Compiled Keras model with quantum pooling

## Quantum Circuit Architecture

For each 2×2 patch (4 values → 4 qubits):

```
1. Angle Encoding:
   RY(input[0] * π) |0⟩ → |ψ₀⟩
   RY(input[1] * π) |0⟩ → |ψ₁⟩
   RY(input[2] * π) |0⟩ → |ψ₂⟩
   RY(input[3] * π) |0⟩ → |ψ₃⟩

2. Repeat `depth` times:
   a. Rotation Layer:
      Rot(φ, θ, ω) on each qubit

   b. All-to-All Entangling Layer:
      CNOT(q₀, q₁)
      CNOT(q₀, q₂)
      CNOT(q₀, q₃)
      CNOT(q₁, q₂)
      CNOT(q₁, q₃)
      CNOT(q₂, q₃)  # 6 CNOTs for full connectivity

3. Measurement:
   Return ⟨Z₀⟩ (expectation value of Pauli-Z on qubit 0)
```

## Example: Custom CNN Architecture

```python
import tensorflow as tf
from src.utils import QuantumPoolingLayer

# Build custom model
inputs = tf.keras.Input(shape=(64, 64, 3))

# Classical feature extraction
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)  # → (32, 32, 64)

x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)

# Quantum pooling layer (replaces GlobalAveragePooling + Dense)
x = QuantumPoolingLayer(depth=3)(x)  # → (batch, 256)

# Classification head
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

## Important Notes

### Execution Mode
**This implementation requires eager execution mode** (TensorFlow default).

The layer works by:
- Converting tensors to NumPy arrays
- Running PennyLane quantum circuits
- Converting results back to TensorFlow tensors

This approach is compatible with:
- ✅ Model building and prediction
- ✅ Training with `model.fit()` in eager mode
- ✅ Custom training loops
- ❌ Graph mode / `@tf.function` decorated training (AutoGraph issues)

### Performance Considerations

1. **Computation time**: Quantum simulation is slow
   - Each forward pass: batch_size × m quantum circuits
   - Larger `depth` = slower execution
   - Recommendation: Start with depth=1, batch_size=4-8

2. **Memory**: O(2^n_qubits) per circuit
   - 4 qubits = 16 complex amplitudes (manageable)

3. **Optimization tips**:
   - Reduce channels before quantum layer (e.g., 256 → 64)
   - Use smaller depth initially
   - Consider gradient checkpointing for large models

## Known Limitations

1. **PennyLane TensorFlow deprecation**: PennyLane is deprecating TensorFlow support in favor of JAX/PyTorch. This may cause warnings or future compatibility issues.

2. **AutoGraph incompatibility**: The TensorFlow AutoGraph system conflicts with PennyLane's quantum operations, causing errors in graph mode.

3. **For production use**, consider:
   - Migrating to PyTorch with PennyLane (recommended)
   - Using JAX with PennyLane
   - Implementing with Qiskit or other quantum frameworks

## Troubleshooting

### "SymbolicTensor has no attribute 'numpy'"
- Ensure you're in eager execution mode (TensorFlow default)
- Don't wrap training code with `@tf.function`

### "ValueError: the else branch must also have a return statement"
- This is an AutoGraph conflict with PennyLane
- Use `quantum_pooling_simple.py` (current version) instead of `quantum_pooling.py`

### Slow training
- Reduce `depth` parameter (try depth=1)
- Use smaller batch sizes
- Reduce number of channels before quantum layer
- Use fewer training samples initially to test

### Import errors
```bash
uv add pennylane
```

## Research Context

This quantum pooling layer is designed for:
- Investigating quantum advantage in computer vision
- Hybrid quantum-classical neural networks
- Parameter-efficient CNN architectures
- Quantum feature learning research

## References

- PennyLane: https://pennylane.ai
- Quantum Convolutional Neural Networks: https://arxiv.org/abs/1810.03787
- Your thesis work on quantum-classical hybrid models

## License

Part of the BreakHis classification pipeline research project.
