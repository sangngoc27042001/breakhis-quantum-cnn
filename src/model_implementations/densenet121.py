"""
DenseNet121 model implementation for BreakHis classification.
Includes multi-level feature fusion from intermediate layers.
"""
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Lambda, Concatenate
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def build_model(num_classes: int = None,
                input_shape: tuple = None,
                dropout_rate: float = None,
                l2_reg: float = None,
                use_fusion: bool = True) -> Model:
    """
    Build DenseNet121 model for BreakHis classification.
    Can optionally use multi-level feature fusion.

    Args:
        num_classes: Number of output classes (uses config.NUM_CLASSES if None)
        input_shape: Input image shape (uses config.INPUT_SHAPE if None)
        dropout_rate: Dropout rate (uses config.DROPOUT_RATE if None)
        l2_reg: L2 regularization factor (uses config.L2_REG if None)
        use_fusion: Whether to use multi-level feature fusion

    Returns:
        Compiled Keras Model
    """
    # Use config defaults if not specified
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if input_shape is None:
        input_shape = config.INPUT_SHAPE
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE
    if l2_reg is None:
        l2_reg = config.L2_REG

    # Input layer
    inputs = Input(shape=input_shape)

    # Base model: DenseNet121 (pretrained on ImageNet)
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    if use_fusion:
        # Multi-level feature fusion
        # Extract features from multiple dense blocks
        layer_names = ['conv3_block12_concat', 'conv4_block24_concat', 'conv5_block16_concat']
        intermediate_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Create intermediate model
        intermediate_model = Model(inputs=base_model.input, outputs=intermediate_outputs)

        # Process each intermediate output
        branch_outputs = []
        for output in intermediate_outputs:
            x = GlobalAveragePooling2D()(output)
            x = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)
            x = Dense(64, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
            x = BatchNormalization()(x)
            branch_outputs.append(x)

        # Concatenate all branches (score-level fusion)
        x = Concatenate()(branch_outputs)

        # Additional dense layers
        x = Dense(16, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    else:
        # Simple architecture without fusion
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    # Output layer with explicit float32 dtype for mixed precision compatibility
    # When using mixed precision, the output layer must be float32 to work with class_weight
    outputs = Dense(num_classes, activation='softmax', dtype='float32', name='predictions')(x)

    # Create model
    model_name = 'DenseNet121_Fusion_BreakHis' if use_fusion else 'DenseNet121_BreakHis'
    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


if __name__ == "__main__":
    # Test both variants
    print("Building DenseNet121 with fusion...")
    model_fusion = build_model(use_fusion=True)
    model_fusion.summary()

    print(f"\nModel parameters (with fusion):")
    print(f"  Total params: {model_fusion.count_params():,}")

    print("\n" + "="*80 + "\n")

    print("Building DenseNet121 without fusion...")
    model_simple = build_model(use_fusion=False)
    model_simple.summary()

    print(f"\nModel parameters (without fusion):")
    print(f"  Total params: {model_simple.count_params():,}")
