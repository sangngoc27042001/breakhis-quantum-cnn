"""
CNN-Quantum Hybrid model implementation for BreakHis classification using PyTorch.

Combines classical CNN backbones with quantum layers:
1. CNN Backbone (any model from AVAILABLE_SMALL_MODELS or legacy backbones)
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
from src.utils import QuantumDenseLayer, QuantumRingRotationLayer


class CNNQuantumHybrid(nn.Module):
    """CNN-Quantum Hybrid Model for classification."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        dropout_rate: float,
        # pooling_depth is currently unused (quantum pooling not implemented)
        pooling_depth: int,
        dense_encoding_method: str,
        dense_template: str,
        dense_depth: int,
        n_qubits: int = 12,
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
        print(f"  [{time.time() - init_start:.2f}s] Loading pretrained backbone '{model_name}'...")
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
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

        # Layer normalization (works better than BatchNorm for small batches and quantum circuits)
        self.layer_norm = nn.LayerNorm(num_features)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Quantum dense layer
        print(f"  [{time.time() - init_start:.2f}s] Initializing quantum layer (n_qubits={n_qubits}, template={dense_template}, depth={dense_depth})...")
        self.quantum_dense = QuantumDenseLayer(
            output_dim=num_classes,
            n_qubits=n_qubits,
            embedding=dense_encoding_method,
            template=dense_template,
            depth=dense_depth,
        )
        # self.quantum_dense = QuantumRingRotationLayer(
        #     output_dim=num_classes,
        #     n_qubits=n_qubits,
        #     embedding=dense_encoding_method,
        #     # template=dense_template,
        #     depth=dense_depth,
        # )
        print(f"  [{time.time() - init_start:.2f}s] Quantum layer initialized")

        # Learnable temperature parameter for scaling quantum logits
        # Initialized to 5.0 to provide stronger initial signals
        self.temperature = nn.Parameter(torch.tensor(5.0))

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

        # Quantum dense layer (outputs logits in range [-1, 1])
        x = self.quantum_dense(x)

        # Scale logits with learnable temperature parameter
        x = x * self.temperature

        return x


def build_model(
    num_classes: int = None,
    input_shape: tuple = None,
    dropout_rate: float = None,
    l2_reg: float = None,
    backbone: str = None,  # Uses config.QUANTUM_CNN_CONFIG_BACKBONE if None
    # pooling_depth is currently unused (quantum pooling not implemented)
    pooling_depth: int = 1,
    dense_encoding_method: str = "amplitude",
    dense_template: str = "strong",
    dense_depth: int = 1,
    n_qubits: int = None,
) -> nn.Module:
    """
    Build CNN-Quantum hybrid model for BreakHis classification.

    Args:
        num_classes: Number of output classes (uses config.NUM_CLASSES if None)
        input_shape: Input image shape (not used in PyTorch)
        dropout_rate: Dropout rate (uses config.DROPOUT_RATE if None)
        l2_reg: L2 regularization factor (applied via optimizer)
        backbone: CNN backbone architecture - can be any model from AVAILABLE_SMALL_MODELS
                 (uses config.QUANTUM_CNN_CONFIG_BACKBONE if None)
        pooling_depth: (Unused) Depth for quantum pooling layer
        dense_encoding_method: Encoding method for quantum dense layer
        dense_template: PennyLane template for quantum dense layer ("strong"|"two_design"|"basic")
        dense_depth: Depth for quantum dense layer
        n_qubits: Number of qubits (uses config.QUANTUM_CNN_CONFIG_NO_QUBITS if None)

    Returns:
        PyTorch nn.Module model
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE
    if n_qubits is None:
        n_qubits = config.QUANTUM_CNN_CONFIG_NO_QUBITS
    if backbone is None:
        backbone = config.QUANTUM_CNN_CONFIG_BACKBONE

    model = CNNQuantumHybrid(
        backbone=backbone,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pooling_depth=pooling_depth,
        dense_encoding_method=dense_encoding_method,
        dense_template=dense_template,
        dense_depth=dense_depth,
        n_qubits=n_qubits,
    )

    return model


if __name__ == "__main__":
    print("Testing CNN-Quantum Hybrid Model...")

    # Uses config.QUANTUM_CNN_CONFIG_BACKBONE (regnety_002) by default
    model = build_model(
        pooling_depth=1,
        dense_encoding_method='amplitude',
        dense_template='strong',
        dense_depth=1
    )

    print(f"Using backbone: {config.QUANTUM_CNN_CONFIG_BACKBONE}")

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
