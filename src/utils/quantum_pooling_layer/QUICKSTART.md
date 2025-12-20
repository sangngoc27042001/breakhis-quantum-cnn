# Quantum Neural Network (QNN) Pooling Layer - Quick Start

**All-to-All Connectivity** for maximum entanglement

## Installation

PennyLane is already in your requirements.txt. Just sync:
```bash
uv sync
```

## Basic Usage

```python
from src.utils import QuantumPoolingLayer

# Create layer
layer = QuantumPoolingLayer(depth=2)

# Use with CNN features (batch, n, n, m) → (batch, m)
output = layer(cnn_features)
```

## Full Model Example

```python
from src.utils import create_simple_cnn

model = create_simple_cnn(
    input_shape=(32, 32, 3),
    num_classes=10,
    quantum_depth=2
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)
```

## Run Tests

```bash
# All 11 tests
uv run python -m src.utils.quantum_pooling_layer.tests
```

## Run Examples

```bash
uv run python -m src.utils.quantum_pooling_layer.examples
```

## Architecture

- **Input**: `(batch, n, n, m)` where n can be any size
- **Adaptive pooling**: Reduces to `(batch, 2, 2, m)`
- **Quantum circuits**: 4 qubits with all-to-all connectivity
  - 6 CNOT gates per layer (full connectivity)
  - All qubit pairs are entangled: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
- **Output**: `(batch, m)`

## Parameters

- `depth`: Number of quantum circuit layers (default: 1)
- Total parameters: `m × depth × 12`

Example: 64 channels, depth=2 → **1,536 quantum parameters**

## Documentation

See [README.md](README.md) for complete documentation.
