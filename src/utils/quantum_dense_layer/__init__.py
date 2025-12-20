"""Quantum dense layer utilities.

This subpackage contains the PennyLane+PyTorch implementation of a trainable
quantum dense layer used by the hybrid CNN-Quantum models.
"""

from .layer import QuantumDenseLayer

__all__ = ["QuantumDenseLayer"]
