"""
VGG16 model implementation for BreakHis classification using PyTorch.
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
    Build VGG16 model for BreakHis classification.

    Args:
        num_classes: Number of output classes (uses config.NUM_CLASSES if None)
        input_shape: Input image shape (uses config.INPUT_SHAPE if None) - not used in PyTorch
        dropout_rate: Dropout rate (uses config.DROPOUT_RATE if None)
        l2_reg: L2 regularization factor (uses config.L2_REG if None) - applied via optimizer

    Returns:
        PyTorch nn.Module model
    """
    # Use config defaults if not specified
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE

    # Create VGG16 model using timm (with pretrained weights)
    model = timm.create_model('vgg16', pretrained=True, num_classes=0)  # num_classes=0 removes classifier

    # Get the number of features from the model
    num_features = model.head.in_features if hasattr(model, 'head') else model.classifier.in_features

    # Replace classifier head
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.BatchNorm1d(num_features),
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, num_classes)
    )

    return model


if __name__ == "__main__":
    # Test model creation
    print("Building VGG16 model...")
    model = build_model()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Non-trainable params: {total_params - trainable_params:,}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
