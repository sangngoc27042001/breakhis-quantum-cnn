"""
Quick usage example of Quantum Pooling Layer

Run from project root:
    uv run python -m src.utils.quantum_pooling_layer.examples
"""

import tensorflow as tf
from .layer import QuantumPoolingLayer, create_simple_cnn

print("Quantum Pooling Layer - Usage Example")
print("=" * 60)

# Example 1: Using the layer directly
print("\n1. Direct layer usage:")
print("-" * 40)

# Create layer
quantum_layer = QuantumPoolingLayer(depth=2)

# Simulate CNN output (batch=4, spatial=8x8, channels=64)
cnn_output = tf.random.normal((4, 8, 8, 64))
print(f"CNN output shape: {cnn_output.shape}")

# Apply quantum pooling
result = quantum_layer(cnn_output)
print(f"After quantum pooling: {result.shape}")
print(f"Result sample: {result[0, :5].numpy()}")  # First 5 channels

# Example 2: Complete model for image classification
print("\n2. Full model for classification:")
print("-" * 40)

model = create_simple_cnn(
    input_shape=(32, 32, 3),
    num_classes=10,
    quantum_depth=1  # Start with depth=1 for faster training
)

print("Model architecture:")
model.summary()

# Test prediction
sample_images = tf.random.normal((2, 32, 32, 3))
predictions = model(sample_images, training=False)
print(f"\nPredictions shape: {predictions.shape}")
print(f"Predictions sum to 1.0: {tf.reduce_sum(predictions, axis=1).numpy()}")

# Example 3: Training snippet
print("\n3. Training setup:")
print("-" * 40)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled and ready for training!")
print("\nTo train:")
print("  model.fit(x_train, y_train, epochs=10, batch_size=32)")

# Show parameter counts
print("\n4. Parameter breakdown:")
print("-" * 40)

total_params = model.count_params()
quantum_params = 0

for layer in model.layers:
    if 'quantum' in layer.name.lower():
        quantum_params = sum([tf.size(w).numpy() for w in layer.trainable_weights])
        print(f"Quantum layer: {quantum_params:,} parameters")

classical_params = total_params - quantum_params
print(f"Classical layers: {classical_params:,} parameters")
print(f"Total: {total_params:,} parameters")
print(f"Quantum fraction: {100 * quantum_params / total_params:.1f}%")

print("\n" + "=" * 60)
print("✓ All examples completed successfully!")
print("\nNext steps:")
print("  1. Prepare your dataset")
print("  2. Adjust quantum_depth and model architecture")
print("  3. Run training with model.fit()")
print("  4. Monitor performance vs classical baseline")
