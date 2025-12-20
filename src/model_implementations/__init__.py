"""
Model implementations for BreakHis classification using PyTorch.
Each module contains a build_model() function that returns a PyTorch nn.Module.
"""
from .vgg16 import build_model as build_vgg16
from .efficientnetv2b3 import build_model as build_efficientnetv2b3
from .densenet169 import build_model as build_densenet169
from .mobilenetv3large import build_model as build_mobilenetv3large
from .nasnetmobile import build_model as build_nasnetmobile
from .cnn_quantum import build_model as build_cnn_quantum

__all__ = [
    'build_vgg16',
    'build_efficientnetv2b3',
    'build_densenet169',
    'build_mobilenetv3large',
    'build_nasnetmobile',
    'build_cnn_quantum',
]
