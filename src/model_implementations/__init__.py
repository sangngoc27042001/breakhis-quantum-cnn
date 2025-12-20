"""
Model implementations for BreakHis classification using PyTorch.
Each module contains a build_model() function that returns a PyTorch nn.Module.
"""
from .small_models import build_model as build_small_model
from .cnn_quantum import build_model as build_cnn_quantum

__all__ = [
    'build_small_model',
    'build_cnn_quantum',
]
