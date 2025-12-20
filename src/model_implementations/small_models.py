"""
Small models (<7M parameters) implementation for BreakHis classification using PyTorch.
This module provides a flexible build_model function that supports multiple small timm models.
Only the biggest model from each family is included.
"""
import torch
import torch.nn as nn
import timm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


# Available small models - biggest from each family (<7M parameters)
AVAILABLE_SMALL_MODELS = {
    # MobileNetV3-Small family (biggest variant)
    'mobilenetv3_small_100': 1.53,

    # ShuffleNetV2 family
    'shufflenetv2_x1_0': 2.3,

    # MNASNet family (biggest variant)
    'mnasnet_100': 3.11,

    # RegNetX family
    'regnetx_002': 2.32,

    # RegNetY family (fastest inference)
    'regnety_002': 2.80,

    # GhostNet family (biggest variant)
    'ghostnet_100': 3.91,

    # EfficientNet-Lite family
    'efficientnet_lite0': 3.38,

    # MobileViT family (biggest variant - hybrid CNN-Transformer)
    'mobilevit_xs': 1.94,
}

# Current backbones from config (for compatibility)
CURRENT_BACKBONES = {
    'vgg16': 'VGG16',
    'efficientnetv2_rw_s': 'EfficientNetV2-S',  # efficientnetv2b3 uses this
    'densenet169': 'DenseNet169',
    'mobilenetv3_large_100': 'MobileNetV3-Large',  # mobilenetv3large
    'nasnetamobile': 'NASNet-Mobile',  # nasnetmobile
}

# Combined available models
AVAILABLE_MODELS = {**AVAILABLE_SMALL_MODELS, **CURRENT_BACKBONES}


def build_model(model_name: str = 'regnety_002',
                num_classes: int = None,
                input_shape: tuple = None,
                dropout_rate: float = None,
                l2_reg: float = None,
                pretrained: bool = True) -> nn.Module:
    """
    Build a small model for BreakHis classification.

    Args:
        model_name: Name of the timm model to use. Must be one of AVAILABLE_MODELS.
                   Default is 'regnety_002' (fastest with good accuracy).
        num_classes: Number of output classes. Defaults to config.NUM_CLASSES.
        input_shape: Input shape (not used, kept for compatibility).
        dropout_rate: Dropout rate for the model. Defaults to config.DROPOUT_RATE.
        l2_reg: L2 regularization (not used directly, kept for compatibility).
        pretrained: Whether to use pretrained weights. Default is True.

    Returns:
        PyTorch model (nn.Module)

    Example:
        # Use default (RegNetY-002 - fastest)
        model = build_model()

        # Use MobileNetV2-100
        model = build_model(model_name='mobilenetv2_100')

        # Use GhostNet-100 (largest small model)
        model = build_model(model_name='ghostnet_100', pretrained=False)

        # Use current backbone (DenseNet169)
        model = build_model(model_name='densenet169')
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE

    # Validate model name
    if model_name not in AVAILABLE_MODELS and model_name not in AVAILABLE_SMALL_MODELS:
        available = ', '.join(sorted(AVAILABLE_SMALL_MODELS.keys()))
        raise ValueError(
            f"Model '{model_name}' not available. "
            f"Choose from small models: {available}"
        )

    # Create model
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_rate
    )

    return model


def list_available_models():
    """Print all available small models with their parameter counts."""
    print("=" * 80)
    print("SMALL MODELS (<7M parameters) - Biggest from each family")
    print("=" * 80)
    for model_name, params in sorted(AVAILABLE_SMALL_MODELS.items(), key=lambda x: x[1]):
        print(f"  {model_name:<30} {params:>6.2f}M parameters")
    print("-" * 80)
    print(f"Total: {len(AVAILABLE_SMALL_MODELS)} small models")

    print("\n" + "=" * 80)
    print("CURRENT BACKBONES (from config)")
    print("=" * 80)
    for model_name, description in CURRENT_BACKBONES.items():
        print(f"  {model_name:<30} {description}")
    print("-" * 80)
    print(f"Total: {len(CURRENT_BACKBONES)} current backbones")

    print("\n" + "=" * 80)
    print("RECOMMENDED SMALL MODELS")
    print("=" * 80)
    print("  - regnety_002: Fastest inference (23.33ms, 2.80M params)")
    print("  - regnetx_002: Second fastest (24.35ms, 2.32M params)")
    print("  - ghostnet_100: Largest small model (3.91M params, 62.38ms)")
    print("  - mobilenetv2_100: Classic efficient (2.23M params, 54.61ms)")
    print("  - mobilevit_xs: Hybrid CNN-Transformer (1.94M params, 70.36ms)")
    print("  - efficientnet_lite0: EfficientNet variant (3.38M params)")
    print("=" * 80)


if __name__ == "__main__":
    print("Building small model for BreakHis classification...")

    # List all available models
    list_available_models()

    print("\n" + "=" * 80)
    print("TESTING DEFAULT MODEL (regnety_002)")
    print("=" * 80)
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Parameters (M): {total_params/1e6:.2f}M")

    # Test with different model
    print("\n" + "=" * 80)
    print("TESTING GHOSTNET_100 (Largest small model)")
    print("=" * 80)
    model2 = build_model(model_name='ghostnet_100')
    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"Total params: {total_params2:,}")
    print(f"Parameters (M): {total_params2/1e6:.2f}M")

    # Test with current backbone
    print("\n" + "=" * 80)
    print("TESTING CURRENT BACKBONE (densenet169)")
    print("=" * 80)
    model3 = build_model(model_name='densenet169')
    total_params3 = sum(p.numel() for p in model3.parameters())
    print(f"Total params: {total_params3:,}")
    print(f"Parameters (M): {total_params3/1e6:.2f}M")
