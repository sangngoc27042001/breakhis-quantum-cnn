# Changelog

## Version 2.0.0 - QNN with All-to-All Connectivity (2025-12-20)

### Changed
- **Architecture**: Changed from QCNN (circular connectivity) to QNN (all-to-all connectivity)
- **Entanglement**: Now uses 6 CNOT gates per layer instead of 4
- **Connectivity**: Full all-to-all connectivity between all qubit pairs
- **Documentation**: Updated to reflect QNN architecture

### Architecture Change Details

**Before (QCNN - Circular):**
```python
# Circular entanglement pattern
for i in range(n_qubits):
    qml.CNOT(wires=[i, (i + 1) % n_qubits])
# 4 CNOTs: (0→1), (1→2), (2→3), (3→0)
```

**After (QNN - All-to-All):**
```python
# All-to-all entanglement pattern
for i in range(n_qubits):
    for j in range(i + 1, n_qubits):
        qml.CNOT(wires=[i, j])
# 6 CNOTs: (0→1), (0→2), (0→3), (1→2), (1→3), (2→3)
```

### Benefits
- ✅ Maximum entanglement between all qubits
- ✅ More expressive quantum circuits
- ✅ Better quantum feature extraction capability
- ✅ Full connectivity for all qubit pairs

### Backward Compatibility
- ⚠️ **Breaking change**: Circuit behavior differs from v1.0
- ✅ **API compatible**: Same function signatures and parameters
- ✅ **Tests**: All 11 tests pass

### Files Modified
- `layer.py` - Updated quantum circuit implementation
- `README.md` - Updated documentation
- `QUICKSTART.md` - Updated quick reference
- `__init__.py` - Updated docstrings

### Files Added
- `ARCHITECTURE.md` - Detailed architecture documentation
- `CHANGELOG.md` - This file

### Migration Guide

No code changes needed! The API remains the same:

```python
from src.utils import QuantumPoolingLayer

# Same usage as before
layer = QuantumPoolingLayer(depth=2)
output = layer(cnn_features)
```

The only difference is the internal quantum circuit now uses all-to-all connectivity for better entanglement.

---

## Version 1.0.0 - Initial Release (2025-12-20)

### Features
- ✅ Quantum pooling layer for TensorFlow CNNs
- ✅ Adaptive pooling (handles any input size)
- ✅ 4-qubit quantum circuits
- ✅ QCNN with circular connectivity
- ✅ Parameterized quantum circuits
- ✅ TensorFlow Keras integration
- ✅ Comprehensive test suite (11 tests)
- ✅ Complete documentation
- ✅ Usage examples

### Architecture (v1.0)
- Angle encoding with RY gates
- Parameterized rotations (Rot gates)
- Circular CNOT pattern (4 gates)
- Single measurement (Z expectation on qubit 0)
