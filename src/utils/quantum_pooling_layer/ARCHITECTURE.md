# QNN Architecture - All-to-All Connectivity

## Circuit Diagram

For each 2×2 patch (4 values → 4 qubits):

```
|0⟩ ──RY(x₀π)──Rot(φ₀,θ₀,ω₀)──●─────●─────●────────────────────── ⟨Z⟩
                               │     │     │
|0⟩ ──RY(x₁π)──Rot(φ₁,θ₁,ω₁)──X──●──┼─────┼──●─────●─────────────
                                  │  │     │  │     │
|0⟩ ──RY(x₂π)──Rot(φ₂,θ₂,ω₂)──────X──┼──●──X──┼──●──●────────────
                                     │  │     │  │  │
|0⟩ ──RY(x₃π)──Rot(φ₃,θ₃,ω₃)─────────X──┼─────X──┼──X────────────
                                        │        │
                                       (Repeat depth times)
```

## All-to-All Connectivity

**CNOT gates per layer: 6**

Entanglement pattern (all qubit pairs):
1. CNOT(q₀, q₁) - Qubit 0 controls Qubit 1
2. CNOT(q₀, q₂) - Qubit 0 controls Qubit 2
3. CNOT(q₀, q₃) - Qubit 0 controls Qubit 3
4. CNOT(q₁, q₂) - Qubit 1 controls Qubit 2
5. CNOT(q₁, q₃) - Qubit 1 controls Qubit 3
6. CNOT(q₂, q₃) - Qubit 2 controls Qubit 3

This creates **maximum entanglement** between all qubits.

## Layer Structure

Each quantum layer consists of:

### 1. Rotation Layer
Applies `Rot(φ, θ, ω)` to each qubit independently:
- 4 qubits × 3 parameters = **12 parameters per layer**

### 2. Entangling Layer
Applies all-to-all CNOT gates:
- **6 CNOT gates** creating full connectivity
- No trainable parameters (CNOT is fixed)

### 3. Measurement
After all layers, measure expectation value:
- `⟨Z₀⟩` on qubit 0

## Parameters

**Total trainable parameters per channel:**
- `depth × 4 qubits × 3 angles = depth × 12`

**For full model:**
- `m channels × depth × 12`

### Examples:
- 64 channels, depth=1: **768 parameters**
- 64 channels, depth=2: **1,536 parameters**
- 128 channels, depth=2: **3,072 parameters**

## Comparison: QNN vs QCNN

### Previous (QCNN - Circular):
```
Entanglement: Circular pattern
CNOT(0→1), CNOT(1→2), CNOT(2→3), CNOT(3→0)
Total: 4 CNOT gates per layer
```

### Current (QNN - All-to-All):
```
Entanglement: Full connectivity
CNOT(0→1), CNOT(0→2), CNOT(0→3),
CNOT(1→2), CNOT(1→3), CNOT(2→3)
Total: 6 CNOT gates per layer
```

**Advantage of All-to-All:**
- Maximum entanglement between all qubit pairs
- More expressive quantum circuits
- Better quantum feature extraction
- Potentially better performance

## Implementation

```python
# All-to-all entangling layer
for i in range(n_qubits):
    for j in range(i + 1, n_qubits):
        qml.CNOT(wires=[i, j])
```

This creates all unique pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).

## Full Forward Pass

For an input CNN feature map `(batch, n, n, m)`:

1. **Adaptive Pool**: `(batch, n, n, m)` → `(batch, 2, 2, m)`

2. **For each of m channels**:
   - Extract 2×2 patch (4 values)
   - Normalize with tanh: values ∈ [-1, 1]
   - Encode into 4 qubits via RY gates
   - Apply `depth` quantum layers (Rot + All-to-All CNOTs)
   - Measure ⟨Z₀⟩ → single scalar

3. **Output**: `(batch, m)` tensor

## Complexity

- **Circuit depth**: O(depth)
- **Gate count per layer**: 4 Rot gates + 6 CNOT gates = 10 gates
- **Total gates**: `depth × 10` gates per patch
- **Simulation complexity**: O(2^4) = 16 quantum amplitudes (manageable)

---

**Type**: Quantum Neural Network (QNN)
**Connectivity**: All-to-All
**Entanglement**: Maximum (all pairs)
**Status**: Production-ready for research
