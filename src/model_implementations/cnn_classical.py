"""
CNN-Classical Hybrid model implementation for BreakHis classification using PyTorch.

Combines classical CNN backbones with a single dense layer:
1. CNN Backbone (any model from AVAILABLE_SMALL_MODELS or legacy backbones)
2. Global Average Pooling
3. Layer Normalization
4. Dropout
5. Single Dense Layer
"""
import torch
import torch.nn as nn
import timm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


class CNNClassicalHybrid(nn.Module):
    """CNN-Classical Hybrid Model for classification."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        dropout_rate: float,
        pretrained: bool = True,
    ):
        import time
        init_start = time.time()
        super().__init__()

        self.backbone_name = backbone

        # Create backbone model without classifier
        # Support any model from AVAILABLE_SMALL_MODELS or legacy backbone names
        backbone_models = {
            'vgg16': 'vgg16',
            'efficientnetv2b3': 'tf_efficientnetv2_b3',
            'densenet169': 'densenet169',
            'mobilenetv3large': 'mobilenetv3_large_100',
            'nasnetmobile': 'nasnetalarge'  # Using NASNetALarge as mobile variant
        }

        # If backbone is in the legacy mapping, use it; otherwise use backbone name directly
        # This allows any model from AVAILABLE_SMALL_MODELS to be used directly
        if backbone.lower() in backbone_models:
            model_name = backbone_models[backbone.lower()]
        else:
            # Assume it's a timm model name (like those in AVAILABLE_SMALL_MODELS)
            model_name = backbone

        # Create backbone
        print(f"  [{time.time() - init_start:.2f}s] Loading {'pretrained' if pretrained else 'random'} backbone '{model_name}'...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        print(f"  [{time.time() - init_start:.2f}s] Backbone loaded")

        # Get number of features from backbone
        print(f"  [{time.time() - init_start:.2f}s] Running dummy forward pass to detect feature dimensions...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            _, num_features, h, w = dummy_output.shape
        print(f"  [{time.time() - init_start:.2f}s] Detected {num_features} features")

        # Average pooling layer (global average pooling)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Layer normalization (works better than BatchNorm for small batches)
        self.layer_norm = nn.LayerNorm(num_features)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Single dense layer (similar to quantum layer structure)
        print(f"  [{time.time() - init_start:.2f}s] Initializing dense layer ({num_features} -> {num_classes})...")
        self.dense = nn.Linear(num_features, num_classes)
        print(f"  [{time.time() - init_start:.2f}s] Dense layer initialized")

    def forward(self, x):
        # Backbone forward pass
        x = self.backbone(x)  # (batch, channels, h, w)

        # Global average pooling
        x = self.avg_pool(x)  # (batch, channels, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch, channels)

        # Layer normalization
        x = self.layer_norm(x)

        # Dropout
        x = self.dropout(x)

        # Dense layer
        x = self.dense(x)

        # softmax
        # x = torch.softmax(x, dim=-1)

        return x


def build_model(
    model_name: str = 'regnety_002',
    num_classes: int = None,
    input_shape: tuple = None,
    dropout_rate: float = None,
    l2_reg: float = None,
    pretrained: bool = True
) -> nn.Module:
    """
    Build CNN-Classical hybrid model for BreakHis classification.

    Args:
        model_name: Name of the timm model to use as backbone. Default is 'regnety_002'.
        num_classes: Number of output classes (uses config.NUM_CLASSES if None)
        input_shape: Input image shape (not used in PyTorch)
        dropout_rate: Dropout rate (uses config.DROPOUT_RATE if None)
        l2_reg: L2 regularization factor (applied via optimizer)
        pretrained: Whether to use pretrained weights. Default is True.

    Returns:
        PyTorch nn.Module model

    Example:
        # Default configuration
        model = build_model()

        # With specific backbone
        model = build_model(model_name='regnety_002')

        # Without pretrained weights
        model = build_model(model_name='ghostnet_100', pretrained=False)
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE

    model = CNNClassicalHybrid(
        backbone=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
    )

    return model


if __name__ == "__main__":
    print("Testing CNN-Classical Hybrid Model...")

    # Default model (regnety_002)
    model = build_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
