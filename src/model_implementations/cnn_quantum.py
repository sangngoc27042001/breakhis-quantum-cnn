"""
CNN-Quantum Hybrid model implementation for BreakHis classification using PyTorch.

Combines classical CNN backbones with quantum layers:
1. CNN Backbone (VGG16, EfficientNetV2B3, DenseNet169, MobileNetV3Large, or NASNetMobile)
2. Quantum Pooling Layer
3. Batch Normalization
4. Dropout
5. Quantum Dense Layer
"""
import torch
import torch.nn as nn
import timm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.utils import QuantumPoolingLayer, QuantumDenseLayer


class CNNQuantumHybrid(nn.Module):
    """CNN-Quantum Hybrid Model for classification."""

    def __init__(self, backbone: str, num_classes: int, dropout_rate: float,
                 pooling_depth: int, dense_encoding_method: str, dense_depth: int):
        super().__init__()

        self.backbone_name = backbone

        # Create backbone model without classifier
        backbone_models = {
            'vgg16': 'vgg16',
            'efficientnetv2b3': 'tf_efficientnetv2_b3',
            'densenet169': 'densenet169',
            'mobilenetv3large': 'mobilenetv3_large_100',
            'nasnetmobile': 'nasnetalarge'  # Using NASNetALarge as mobile variant
        }

        if backbone.lower() not in backbone_models:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Create backbone
        model_name = backbone_models[backbone.lower()]
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')

        # Get number of features from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            _, num_features, h, w = dummy_output.shape

        # Quantum pooling layer (expects channel-last format)
        self.quantum_pool = QuantumPoolingLayer(depth=pooling_depth)

        # Layer normalization (works better than BatchNorm for small batches and quantum circuits)
        self.layer_norm = nn.LayerNorm(num_features)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Quantum dense layer
        self.quantum_dense = QuantumDenseLayer(
            output_dim=num_classes,
            embedding=dense_encoding_method,
            depth=dense_depth
        )

    def forward(self, x):
        # Backbone forward pass
        x = self.backbone(x)  # (batch, channels, h, w)

        # Convert to channel-last for quantum pooling
        x = x.permute(0, 2, 3, 1)  # (batch, h, w, channels)

        # Quantum pooling
        x = self.quantum_pool(x)  # (batch, channels)

        # Layer normalization
        x = self.layer_norm(x)

        # Dropout
        x = self.dropout(x)

        # Quantum dense layer (outputs logits)
        x = self.quantum_dense(x)

        return x


def build_model(num_classes: int = None,
                input_shape: tuple = None,
                dropout_rate: float = None,
                l2_reg: float = None,
                backbone: str = "mobilenetv3large",
                pooling_depth: int = 1,
                dense_encoding_method: str = "amplitude",
                dense_depth: int = 1) -> nn.Module:
    """
    Build CNN-Quantum hybrid model for BreakHis classification.

    Args:
        num_classes: Number of output classes (uses config.NUM_CLASSES if None)
        input_shape: Input image shape (not used in PyTorch)
        dropout_rate: Dropout rate (uses config.DROPOUT_RATE if None)
        l2_reg: L2 regularization factor (applied via optimizer)
        backbone: CNN backbone architecture
        pooling_depth: Depth for quantum pooling layer
        dense_encoding_method: Encoding method for quantum dense layer
        dense_depth: Depth for quantum dense layer

    Returns:
        PyTorch nn.Module model
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE

    model = CNNQuantumHybrid(
        backbone=backbone,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pooling_depth=pooling_depth,
        dense_encoding_method=dense_encoding_method,
        dense_depth=dense_depth
    )

    return model


if __name__ == "__main__":
    print("Testing CNN-Quantum Hybrid Model...")

    model = build_model(
        backbone='mobilenetv3large',
        pooling_depth=1,
        dense_encoding_method='amplitude',
        dense_depth=1
    )

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
