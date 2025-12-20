"""
Quantum Pooling Layer for TensorFlow CNN Models

A hybrid quantum-classical layer that uses PennyLane to process CNN feature maps
through parameterized quantum circuits.

Example:
    >>> from src.utils.quantum_pooling_layer import QuantumPoolingLayer
    >>> layer = QuantumPoolingLayer(depth=2)
    >>> output = layer(cnn_features)
"""

from .layer import QuantumPoolingLayer, create_simple_cnn

__all__ = ['QuantumPoolingLayer', 'create_simple_cnn']

__version__ = '1.0.0'
