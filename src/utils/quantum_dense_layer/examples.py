"""
Examples of using the Quantum Dense Layer

This module demonstrates various use cases for the QuantumDenseLayer.
"""

import tensorflow as tf
import numpy as np
from layer import QuantumDenseLayer, create_quantum_mlp


def example_amplitude_embedding():
    """Example using amplitude embedding."""
    print("\n" + "="*60)
    print("Example 1: Amplitude Embedding")
    print("="*60)

    # Create layer with amplitude embedding
    # Input: 256 features -> Output: 64 features
    layer = QuantumDenseLayer(
        output_dim=64,
        embedding='amplitude',
        depth=2
    )

    # Create sample input
    batch_size = 4
    input_dim = 256
    x = tf.random.normal([batch_size, input_dim])

    # Forward pass
    output = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample (first 5 values): {output[0, :5].numpy()}")
    print(f"Output sum (should be ~1 as probabilities): {tf.reduce_sum(output[0]).numpy():.4f}")


def example_rotation_embedding():
    """Example using rotation embedding with patch averaging."""
    print("\n" + "="*60)
    print("Example 2: Rotation Embedding with Patch Averaging")
    print("="*60)

    # Create layer with rotation embedding
    # Input: 1000 features -> divided into 12 patches -> Output: 128 features
    layer = QuantumDenseLayer(
        output_dim=128,
        embedding='rotation',
        depth=3
    )

    # Create sample input
    batch_size = 8
    input_dim = 1000
    x = tf.random.normal([batch_size, input_dim])

    # Forward pass
    output = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample (first 5 values): {output[0, :5].numpy()}")
    print(f"Output sum (should be ~1 as probabilities): {tf.reduce_sum(output[0]).numpy():.4f}")


def example_max_dimension():
    """Example using maximum allowed dimensions (2^12 = 4096)."""
    print("\n" + "="*60)
    print("Example 3: Maximum Dimension (2^12 = 4096)")
    print("="*60)

    # Create layer with max dimensions
    layer = QuantumDenseLayer(
        output_dim=4096,  # Maximum output dimension
        embedding='rotation',
        depth=1
    )

    # Create sample input with max dimension
    batch_size = 2
    input_dim = 4096  # Maximum input dimension
    x = tf.random.normal([batch_size, input_dim])

    # Forward pass
    output = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be ~1 as probabilities): {tf.reduce_sum(output[0]).numpy():.4f}")


def example_quantum_mlp():
    """Example using the quantum MLP builder."""
    print("\n" + "="*60)
    print("Example 4: Quantum Multi-Layer Perceptron")
    print("="*60)

    # Create a quantum MLP for classification
    input_dim = 512
    num_classes = 10
    model = create_quantum_mlp(
        input_dim=input_dim,
        output_dim=num_classes,
        hidden_dims=[256, 128],  # Two quantum hidden layers
        quantum_depth=2,
        embedding='rotation'
    )

    # Create sample input
    batch_size = 4
    x = tf.random.normal([batch_size, input_dim])

    # Forward pass
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0].numpy()}")
    print(f"Predicted class: {tf.argmax(output[0]).numpy()}")

    # Print model summary
    print("\nModel architecture:")
    model.summary()


def example_classification_task():
    """Example: Binary classification task with quantum dense layer."""
    print("\n" + "="*60)
    print("Example 5: Binary Classification with Training")
    print("="*60)

    # Generate synthetic binary classification data
    num_samples = 100
    input_dim = 64

    # Create two clusters
    np.random.seed(42)
    X_class0 = np.random.randn(num_samples // 2, input_dim) + np.array([2.0] * input_dim)
    X_class1 = np.random.randn(num_samples // 2, input_dim) + np.array([-2.0] * input_dim)
    X = np.vstack([X_class0, X_class1]).astype(np.float32)
    y = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))
    y = tf.keras.utils.to_categorical(y, 2)

    # Shuffle data
    indices = np.random.permutation(num_samples)
    X = X[indices]
    y = y[indices]

    # Build model
    inputs = tf.keras.Input(shape=(input_dim,))
    x = QuantumDenseLayer(output_dim=32, embedding='rotation', depth=2)(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nTraining quantum model...")
    # Train for just a few epochs as demonstration
    history = model.fit(
        X, y,
        batch_size=16,
        epochs=3,
        verbose=1,
        validation_split=0.2
    )

    print(f"\nFinal training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


def example_compare_embeddings():
    """Compare amplitude vs rotation embedding."""
    print("\n" + "="*60)
    print("Example 6: Comparing Embedding Methods")
    print("="*60)

    # Create same input for both
    batch_size = 4
    input_dim = 512
    output_dim = 64
    x = tf.random.normal([batch_size, input_dim])

    # Amplitude embedding
    layer_amp = QuantumDenseLayer(
        output_dim=output_dim,
        embedding='amplitude',
        depth=2
    )
    output_amp = layer_amp(x)

    # Rotation embedding
    layer_rot = QuantumDenseLayer(
        output_dim=output_dim,
        embedding='rotation',
        depth=2
    )
    output_rot = layer_rot(x)

    print(f"Input shape: {x.shape}")
    print(f"\nAmplitude embedding output shape: {output_amp.shape}")
    print(f"Amplitude output sample: {output_amp[0, :5].numpy()}")
    print(f"Amplitude output sum: {tf.reduce_sum(output_amp[0]).numpy():.4f}")

    print(f"\nRotation embedding output shape: {output_rot.shape}")
    print(f"Rotation output sample: {output_rot[0, :5].numpy()}")
    print(f"Rotation output sum: {tf.reduce_sum(output_rot[0]).numpy():.4f}")


if __name__ == '__main__':
    # Enable eager execution
    tf.config.run_functions_eagerly(True)

    print("\n" + "="*60)
    print("QUANTUM DENSE LAYER EXAMPLES")
    print("="*60)

    # Run all examples
    example_amplitude_embedding()
    example_rotation_embedding()
    example_max_dimension()
    example_quantum_mlp()
    example_compare_embeddings()
    example_classification_task()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
