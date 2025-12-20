"""
DenseNet169 model implementation for BreakHis classification using PyTorch.
"""
import torch
import torch.nn as nn
import timm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def build_model(num_classes: int = None,
                input_shape: tuple = None,
                dropout_rate: float = None,
                l2_reg: float = None) -> nn.Module:
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE

    model = timm.create_model('densenet169', pretrained=True, num_classes=num_classes, drop_rate=dropout_rate)
    return model


if __name__ == "__main__":
    print("Building DenseNet169 model...")
    model = build_model()
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
