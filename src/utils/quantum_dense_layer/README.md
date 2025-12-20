# Quantum Dense Layer

A TensorFlow/Keras implementation of a Quantum Neural Network (QNN) dense layer using PennyLane. This layer transforms inputs from (batch_size, m) to (batch_size, n) using quantum circuits, where both m and n must be ≤ 2^12 (4096).

## Features

- **Two Embedding Methods**:
  - **Amplitude Embedding**: Encodes classical data into quantum state amplitudes
  - **Rotation Embedding**: Divides input into 12 patches, averages each patch, and uses rotation gates

- **Configurable Depth**: Control the number of quantum circuit layers

- **Probability-based Output**: Returns probabilities of first n quantum states

- **All-to-all Entanglement**: Uses CNOT gates between all qubit pairs for maximum expressiveness

## Installation

Ensure you have the required dependencies:

```bash
pip install tensorflow pennylane numpy
```

## Quick Start

### Basic Usage

```python
import tensorflow as tf
from quantum_dense_layer import QuantumDenseLayer

# Create a quantum dense layer
layer = QuantumDenseLayer(
    output_dim=64,          # Output dimension (≤ 4096)
    embedding='amplitude',   # or 'rotation'
    depth=2                 # Number of quantum layers
)

# Use in a model
inputs = tf.keras.Input(shape=(256,))
x = layer(inputs)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

### Amplitude Embedding Example

```python
# Amplitude embedding for power-of-2 dimensions
layer = QuantumDenseLayer(
    output_dim=128,
    embedding='amplitude',
    depth=3
)

# Input shape: (batch_size, 512) -> Output shape: (batch_size, 128)
x = tf.random.normal([4, 512])
output = layer(x)
```

### Rotation Embedding Example

```python
# Rotation embedding with automatic patch averaging
layer = QuantumDenseLayer(
    output_dim=64,
    embedding='rotation',
    depth=2
)

# Input of any size (≤ 4096) -> divided into 12 patches -> output
x = tf.random.normal([8, 1000])
output = layer(x)  # Shape: (8, 64)
```

## Embedding Methods

### Amplitude Embedding

- Encodes classical data directly into quantum state amplitudes
- More efficient for power-of-2 input dimensions
- Requires normalization to unit vector
- Uses fewer quantum gates for encoding

**How it works:**
1. Pads input to nearest power of 2 if needed
2. Normalizes to unit vector
3. Uses `AmplitudeEmbedding` to encode data
4. Applies variational quantum layers
5. Measures state probabilities

### Rotation Embedding

- Divides input vector into 12 patches
- Computes average of each patch
- Encodes using RY rotation gates on 12 qubits
- Works with any input dimension

**How it works:**
1. Divides m-dimensional input into 12 patches
2. Computes mean of each patch → 12 values
3. Applies `tanh` normalization to [-1, 1]
4. Encodes using RY(x[i] * π) on each qubit
5. Applies variational quantum layers
6. Measures state probabilities

**Patch Division Example:**
- Input dimension m = 1000
- Patch size = 1000 // 12 = 83
- Remainder = 1000 % 12 = 4
- First 4 patches get 84 elements each
- Last 8 patches get 83 elements each
- Result: 12-dimensional vector

## Quantum Circuit Architecture

The quantum circuit consists of:

1. **Encoding Layer**: Amplitude or Rotation embedding
2. **Variational Layers** (repeated `depth` times):
   - **Rotation gates**: `Rot(φ, θ, ω)` on each qubit
   - **Entangling gates**: All-to-all CNOT connections
3. **Measurement**: Probability measurement of all states

```
|0⟩ ─ Encoding ─ Rot ─ ╭● ─ ╭● ─ ... ─ Rot ─ ╭● ─ ╭● ─ Measure
|0⟩ ─ Encoding ─ Rot ─ ╰X ─ ├● ─ ... ─ Rot ─ ╰X ─ ├● ─ Measure
|0⟩ ─ Encoding ─ Rot ─     ─ ╰X ─ ... ─ Rot ─     ─ ╰X ─ Measure
...                        (depth layers)
```

## Parameters

### QuantumDenseLayer

- **output_dim** (int): Output dimension n, must be ≤ 4096 (2^12)
- **embedding** (str): 'amplitude' or 'rotation'
- **depth** (int): Number of quantum circuit layers (default: 1)

### Constraints

- Input dimension m ≤ 4096 (2^12)
- Output dimension n ≤ 4096 (2^12)
- 12 qubits available (supports up to 2^12 states)

## Advanced Usage

### Building a Quantum MLP

```python
from quantum_dense_layer import create_quantum_mlp

model = create_quantum_mlp(
    input_dim=512,
    output_dim=10,           # Number of classes
    hidden_dims=[256, 128],  # Quantum hidden layers
    quantum_depth=2,
    embedding='rotation'
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=10)
```

### Hybrid Quantum-Classical Network

```python
inputs = tf.keras.Input(shape=(1024,))

# Classical preprocessing
x = tf.keras.layers.Dense(512, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.3)(x)

# Quantum layer
x = QuantumDenseLayer(
    output_dim=128,
    embedding='rotation',
    depth=3
)(x)

# Classical post-processing
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

## Output Interpretation

The layer outputs **probability distributions**:
- Each output value represents the probability of measuring the corresponding quantum state
- Sum of all outputs ≈ 1.0
- Can be interpreted as learned features for downstream tasks
- Probabilities are computed from quantum state measurements

## Performance Considerations

- **Eager Execution Only**: This implementation requires TensorFlow eager mode
- **Batch Processing**: Processes each sample in the batch sequentially (quantum circuits run one at a time)
- **Computational Cost**: Increases with depth and number of qubits
- **Recommended Depth**: Start with depth=1-3 for faster training

### Choosing Embedding Method

| Criterion | Amplitude | Rotation |
|-----------|-----------|----------|
| Input dimension is power of 2 | ✓ Preferred | ○ Works |
| Arbitrary input dimension | ○ Works (pads) | ✓ Preferred |
| Computational efficiency | ✓ More efficient | ○ Less efficient |
| Input size flexibility | ○ Best for 2^n | ✓ Any size |

## Examples

Run the examples file to see various use cases:

```bash
cd /path/to/quantum_dense_layer
python examples.py
```

Examples include:
1. Amplitude embedding
2. Rotation embedding with patch averaging
3. Maximum dimension (4096)
4. Quantum MLP
5. Binary classification with training
6. Comparing embedding methods

## Limitations

- **Eager Mode Only**: Does not support TensorFlow graph mode
- **Batch Size**: Each sample processed sequentially (no parallel quantum execution)
- **Max Dimension**: Limited to 2^12 = 4096 for both input and output
- **Device**: Uses simulated quantum device (not real quantum hardware)

## Technical Details

### Quantum Gates Used

- **RY**: Single-qubit Y-rotation for encoding
- **Rot**: General single-qubit rotation (3 parameters)
- **CNOT**: Two-qubit entangling gate
- **AmplitudeEmbedding**: State preparation from classical data

### Trainable Parameters

For a layer with depth `d` and input requiring `q` qubits:
- Number of parameters: `d × q × 3`
- Each parameter initialized uniformly in [0, 2π]

Example:
- depth=2, rotation embedding (12 qubits): 2 × 12 × 3 = 72 parameters

## References

- PennyLane Documentation: https://pennylane.ai
- Quantum Machine Learning: https://pennylane.ai/qml/
- TensorFlow Quantum: https://www.tensorflow.org/quantum

## License

See repository license.
