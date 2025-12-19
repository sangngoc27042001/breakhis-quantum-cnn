"""
Example usage of the BreakHis classification pipeline.
This script demonstrates how to use the various components.
"""

# Example 1: Prepare the dataset
def example_prepare_dataset():
    """Example: Download and prepare the BreakHis dataset."""
    from data_preparation import prepare_breakhis_dataset

    print("="*80)
    print("Example 1: Preparing BreakHis Dataset")
    print("="*80 + "\n")

    prepare_breakhis_dataset(
        kaggle_dataset="ambarish/breakhis",
        output_dir="./processed_breakhis_balanced",
        target_size=(224, 224),
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        balance_train=True,
        random_state=42
    )


# Example 2: Load and inspect datasets
def example_load_datasets():
    """Example: Load datasets and inspect them."""
    from src import breakhis_data_loader
    from src import config

    print("\n" + "="*80)
    print("Example 2: Loading and Inspecting Datasets")
    print("="*80 + "\n")

    # Create datasets
    train_ds, val_ds, test_ds = breakhis_data_loader.create_all_datasets(
        batch_size=32
    )

    # Show class distribution
    print("\nClass distribution:")
    for split_name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        dist = breakhis_data_loader.get_class_distribution(ds)
        print(f"\n{split_name}:")
        for class_idx, count in sorted(dist.items()):
            class_name = [k for k, v in config.CLASS_MAP.items() if v == class_idx][0]
            print(f"  {class_name:20} (class {class_idx}): {count:5d} images")

    # Test batch shape
    print("\n\nSample batch:")
    for images, labels in train_ds.take(1):
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image dtype: {images.dtype}")
        print(f"Label dtype: {labels.dtype}")
        print(f"Image value range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")


# Example 3: Build and inspect a model
def example_build_model():
    """Example: Build a model and show its architecture."""
    from src.model_implementations import build_convnext_tiny

    print("\n" + "="*80)
    print("Example 3: Building ConvNeXt Tiny Model")
    print("="*80 + "\n")

    model = build_convnext_tiny()

    print("Model Summary:")
    model.summary()

    print(f"\n\nModel Statistics:")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")


# Example 4: Train a model programmatically
def example_train_model():
    """Example: Train a model using the training function."""
    from train import train_model, evaluate_model

    print("\n" + "="*80)
    print("Example 4: Training a Model")
    print("="*80 + "\n")

    # Train ConvNeXt Tiny for 5 epochs (quick test)
    model, history = train_model(
        model_name='convnext_tiny',
        epochs=5,  # Small number for quick testing
        batch_size=32,
        learning_rate=1e-4,
        use_class_weights=True
    )

    # Show training history
    print("\n\nTraining History:")
    for epoch in range(len(history.history['loss'])):
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {history.history['loss'][epoch]:.4f}")
        print(f"  Train Accuracy: {history.history['accuracy'][epoch]:.4f}")
        print(f"  Val Loss: {history.history['val_loss'][epoch]:.4f}")
        print(f"  Val Accuracy: {history.history['val_accuracy'][epoch]:.4f}")

    # Evaluate on test set
    test_results = evaluate_model(model, batch_size=32)


# Example 5: Compare multiple models
def example_compare_models():
    """Example: Train and compare multiple models."""
    from train import MODEL_REGISTRY, train_model

    print("\n" + "="*80)
    print("Example 5: Comparing Multiple Models")
    print("="*80 + "\n")

    models_to_compare = ['convnext_tiny', 'efficientnetb0', 'mobilenetv2']
    results = {}

    for model_name in models_to_compare:
        print(f"\n\nTraining {model_name}...")
        print("-" * 80)

        model, history = train_model(
            model_name=model_name,
            epochs=3,  # Quick test
            batch_size=32,
            use_class_weights=True
        )

        # Store results
        results[model_name] = {
            'final_train_acc': history.history['accuracy'][-1],
            'final_val_acc': history.history['val_accuracy'][-1],
            'best_val_acc': max(history.history['val_accuracy']),
            'params': model.count_params()
        }

    # Compare results
    print("\n\n" + "="*80)
    print("Model Comparison")
    print("="*80)
    print(f"{'Model':<20} {'Params':<15} {'Train Acc':<12} {'Val Acc':<12} {'Best Val Acc':<12}")
    print("-" * 80)

    for model_name, result in results.items():
        print(f"{model_name:<20} {result['params']:>14,} {result['final_train_acc']:>11.4f} "
              f"{result['final_val_acc']:>11.4f} {result['best_val_acc']:>11.4f}")


# Example 6: Custom configuration
def example_custom_config():
    """Example: Using custom configuration."""
    from src import config

    print("\n" + "="*80)
    print("Example 6: Custom Configuration")
    print("="*80 + "\n")

    # Show current configuration
    print("Current Configuration:")
    print(f"  Image size: {config.IMG_SIZE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Number of classes: {config.NUM_CLASSES}")
    print(f"  Learning rate: {config.INITIAL_LEARNING_RATE}")
    print(f"  Dropout rate: {config.DROPOUT_RATE}")
    print(f"  L2 regularization: {config.L2_REG}")
    print(f"  Mixed precision: {config.USE_MIXED_PRECISION}")

    print("\n\nClass mappings:")
    for class_name, class_idx in sorted(config.CLASS_MAP.items(), key=lambda x: x[1]):
        weight = config.CLASS_WEIGHTS.get(class_idx, 1.0)
        print(f"  {class_idx}: {class_name:20} (weight: {weight:.4f})")


# Example 7: Load and use a trained model
def example_load_trained_model():
    """Example: Load a trained model and make predictions."""
    import tensorflow as tf
    import numpy as np
    from src import breakhis_data_loader
    from src import config

    print("\n" + "="*80)
    print("Example 7: Loading and Using a Trained Model")
    print("="*80 + "\n")

    # Note: This assumes you have a trained model saved
    # Replace with your actual model path
    model_path = "checkpoints/convnext_tiny_YYYYMMDD_HHMMSS_best.keras"

    try:
        # Load model
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Load test dataset
        test_ds = breakhis_data_loader.create_dataset('test', is_training=False, batch_size=32)

        # Make predictions on first batch
        for images, labels in test_ds.take(1):
            predictions = model.predict(images)

            print(f"\nPredictions for first 5 samples:")
            for i in range(min(5, len(labels))):
                true_label = labels[i].numpy()
                pred_label = np.argmax(predictions[i])
                confidence = predictions[i][pred_label]

                true_class = [k for k, v in config.CLASS_MAP.items() if v == true_label][0]
                pred_class = [k for k, v in config.CLASS_MAP.items() if v == pred_label][0]

                print(f"\nSample {i+1}:")
                print(f"  True class: {true_class} (index {true_label})")
                print(f"  Predicted class: {pred_class} (index {pred_label})")
                print(f"  Confidence: {confidence:.4f}")
                print(f"  Correct: {'Yes' if true_label == pred_label else 'No'}")

    except Exception as e:
        print(f"Could not load model: {e}")
        print("Please train a model first using train.py")


if __name__ == "__main__":
    import sys

    examples = {
        '1': ('Prepare dataset', example_prepare_dataset),
        '2': ('Load and inspect datasets', example_load_datasets),
        '3': ('Build model', example_build_model),
        '4': ('Train model', example_train_model),
        '5': ('Compare models', example_compare_models),
        '6': ('Show configuration', example_custom_config),
        '7': ('Use trained model', example_load_trained_model),
    }

    if len(sys.argv) > 1:
        # Run specific example
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"\nRunning Example {example_num}: {name}\n")
            func()
        else:
            print(f"Unknown example: {example_num}")
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        # Show menu
        print("="*80)
        print("BreakHis Classification - Example Usage")
        print("="*80)
        print("\nAvailable examples:")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")

        print("\nUsage:")
        print("  python example_usage.py <example_number>")
        print("\nExample:")
        print("  python example_usage.py 3")
