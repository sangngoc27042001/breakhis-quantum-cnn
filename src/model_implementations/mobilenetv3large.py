"""
MobileNetV3Large model implementation for BreakHis classification.
"""
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def build_model(num_classes: int = None,
                input_shape: tuple = None,
                dropout_rate: float = None,
                l2_reg: float = None) -> Model:
    """
    Build MobileNetV3Large model for BreakHis classification.

    Args:
        num_classes: Number of output classes (uses config.NUM_CLASSES if None)
        input_shape: Input image shape (uses config.INPUT_SHAPE if None)
        dropout_rate: Dropout rate (uses config.DROPOUT_RATE if None)
        l2_reg: L2 regularization factor (uses config.L2_REG if None)

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

    # Base model: MobileNetV3Large (pretrained on ImageNet)
    base_model = tf.keras.applications.MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    # Get base model output
    x = base_model(inputs)

    # Classification head
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Output layer with explicit float32 dtype for mixed precision compatibility
    # When using mixed precision, the output layer must be float32 to work with class_weight
    outputs = Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        dtype='float32',
        name='predictions'
    )(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='MobileNetV3Large_BreakHis')

    return model


if __name__ == "__main__":
    # Test model creation
    print("Building MobileNetV3Large model...")
    model = build_model()
    model.summary()

    print(f"\nModel parameters:")
    print(f"  Total params: {model.count_params():,}")
    print(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print(f"  Non-trainable params: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}")
