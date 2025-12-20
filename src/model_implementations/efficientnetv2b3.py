"""
EfficientNetV2B3 model implementation for BreakHis classification using PyTorch.
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
    """
    Build EfficientNetV2B3 model for BreakHis classification.

    Args:
        num_classes: Number of output classes (uses config.NUM_CLASSES if None)
        input_shape: Input image shape (not used in PyTorch)
        dropout_rate: Dropout rate (uses config.DROPOUT_RATE if None)
        l2_reg: L2 regularization factor (applied via optimizer)

    Returns:
        PyTorch nn.Module model
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE

    # Create EfficientNetV2 model
    model = timm.create_model('tf_efficientnetv2_b3', pretrained=True, num_classes=num_classes, drop_rate=dropout_rate)

    return model


if __name__ == "__main__":
    print("Building EfficientNetV2B3 model...")
    model = build_model()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Non-trainable params: {total_params - trainable_params:,}")
