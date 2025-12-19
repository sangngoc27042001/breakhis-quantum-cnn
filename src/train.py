"""
Training script for BreakHis classification models.
All configuration is controlled via config.py.
"""
import os
import tensorflow as tf
from tensorflow.keras import callbacks, mixed_precision
import numpy as np
from datetime import datetime

from src import config
from src import breakhis_data_loader
from src.model_implementations import (
    build_vgg16,
    build_efficientnetv2b3,
    build_densenet169,
    build_mobilenetv3large,
    build_nasnetmobile,
)
from src.checkpoint_callbacks import EpochCheckpointCallback, BestModelTracker


# Model registry
MODEL_REGISTRY = {
    "vgg16": build_vgg16,
    "efficientnetv2b3": build_efficientnetv2b3,
    "densenet169": build_densenet169,
    "mobilenetv3large": build_mobilenetv3large,
    "nasnetmobile": build_nasnetmobile,
}


def setup_mixed_precision():
    """Enable mixed precision training for better performance."""
    if config.USE_MIXED_PRECISION:
        policy = mixed_precision.Policy(config.MIXED_PRECISION_DTYPE)
        mixed_precision.set_global_policy(policy)
        print(f"Mixed precision enabled: {policy.name}")
    else:
        print("Mixed precision disabled")


def get_callbacks(model_name: str, timestamp: str, run_dir: str, train_ds=None, val_ds=None, test_ds=None,
                  checkpoint_epochs=[20, 30, 50], config_dict=None):
    """
    Create training callbacks.

    Args:
        model_name: Name of the model being trained
        timestamp: Timestamp string for logging
        run_dir: Directory for this specific training run
        train_ds: Training dataset (for evaluation at checkpoints)
        val_ds: Validation dataset (for evaluation at checkpoints)
        test_ds: Test dataset (for evaluation at checkpoints)
        checkpoint_epochs: List of epochs at which to save checkpoints
        config_dict: Configuration dictionary to save

    Returns:
        List of Keras callbacks
    """
    callback_list = []

    # ModelCheckpoint - save best model based on val_accuracy
    checkpoint_path = os.path.join(run_dir, 'model_best.keras')
    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=config.CHECKPOINT_MONITOR,
        save_best_only=True,
        mode=config.CHECKPOINT_MODE,
        verbose=1
    )
    callback_list.append(checkpoint)

    # Best Model Tracker - track and save info about best model
    best_tracker = BestModelTracker(
        model_name=model_name,
        timestamp=timestamp,
        run_dir=run_dir,
        monitor=config.CHECKPOINT_MONITOR,
        mode=config.CHECKPOINT_MODE
    )
    callback_list.append(best_tracker)

    # Epoch-specific checkpoints - save models at specific epochs with full evaluation
    if train_ds is not None and val_ds is not None and test_ds is not None and config_dict is not None:
        epoch_checkpoint = EpochCheckpointCallback(
            checkpoint_epochs=checkpoint_epochs,
            model_name=model_name,
            timestamp=timestamp,
            run_dir=run_dir,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            config_dict=config_dict
        )
        callback_list.append(epoch_checkpoint)

    # ReduceLROnPlateau - reduce learning rate when validation metric plateaus
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor=config.CHECKPOINT_MONITOR,
        factor=config.LR_REDUCTION_FACTOR,
        patience=config.LR_REDUCTION_PATIENCE,
        min_lr=config.MIN_LEARNING_RATE,
        verbose=1
    )
    callback_list.append(reduce_lr)

    # TensorBoard - logging (optional, can be disabled if not needed)
    log_dir = os.path.join(run_dir, 'tensorboard_logs')
    tensorboard = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    callback_list.append(tensorboard)

    # CSVLogger - save training history to CSV
    csv_path = os.path.join(run_dir, 'training_history.csv')
    csv_logger = callbacks.CSVLogger(csv_path)
    callback_list.append(csv_logger)

    return callback_list


def compile_model(model: tf.keras.Model, learning_rate: float = None):
    """
    Compile model with optimizer, loss, and metrics.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate (uses config.INITIAL_LEARNING_RATE if None)
    """
    if learning_rate is None:
        learning_rate = config.INITIAL_LEARNING_RATE

    # When using mixed precision with class weights, we need to ensure
    # the loss is computed in float32 to avoid dtype mismatch

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )


def train_model(model_name: str,
                epochs: int = None,
                batch_size: int = None,
                learning_rate: float = None,
                use_class_weights: bool = True):
    """
    Train a model on the BreakHis dataset.

    Args:
        model_name: Name of the model to train (from MODEL_REGISTRY)
        epochs: Number of training epochs (uses config.EPOCHS if None)
        batch_size: Batch size (uses config.BATCH_SIZE if None)
        learning_rate: Initial learning rate (uses config.INITIAL_LEARNING_RATE if None)
        use_class_weights: Whether to use class weights for imbalanced data

    Returns:
        Tuple of (trained_model, history)
    """
    # Set defaults
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if learning_rate is None:
        learning_rate = config.INITIAL_LEARNING_RATE

    # Setup mixed precision
    setup_mixed_precision()

    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create run directory: ./results/{model_name}_{timestamp}/
    run_dir = os.path.join(config.RESULTS_DIR, f'{model_name}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    print("="*80)
    print(f"Training Configuration")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Use class weights: {use_class_weights}")
    print(f"Mixed precision: {config.USE_MIXED_PRECISION}")
    print(f"Results directory: {run_dir}")
    print("="*80 + "\n")

    # Load datasets
    print("Loading datasets...")
    train_ds = breakhis_data_loader.create_dataset('train', is_training=True, batch_size=batch_size)
    val_ds = breakhis_data_loader.create_dataset('val', is_training=False, batch_size=batch_size)
    test_ds = breakhis_data_loader.create_dataset('test', is_training=False, batch_size=batch_size)

    # Get class weights if requested
    class_weights = None
    if use_class_weights:
        class_weights = config.CLASS_WEIGHTS
        print("\nUsing class weights:")
        for class_idx, weight in sorted(class_weights.items()):
            class_name = [k for k, v in config.CLASS_MAP.items() if v == class_idx][0]
            print(f"  {class_name:20} (class {class_idx}): {weight:.4f}")
        print()

    # Build model
    print(f"\nBuilding {model_name} model...")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[model_name](
        num_classes=config.NUM_CLASSES,
        input_shape=config.INPUT_SHAPE,
        dropout_rate=config.DROPOUT_RATE,
        l2_reg=config.L2_REG
    )

    # Compile model
    print("Compiling model...")
    compile_model(model, learning_rate)

    # Print model summary
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")

    # Prepare configuration dictionary
    config_dict = {
        'model_name': model_name,
        'batch_size': batch_size,
        'epochs': epochs,
        'initial_learning_rate': learning_rate,
        'min_learning_rate': config.MIN_LEARNING_RATE,
        'l2_reg': config.L2_REG,
        'dropout_rate': config.DROPOUT_RATE,
        'use_class_weights': use_class_weights,
        'use_mixed_precision': config.USE_MIXED_PRECISION,
        'img_size': config.IMG_SIZE,
        'num_classes': config.NUM_CLASSES,
        'timestamp': timestamp
    }

    # Get callbacks
    print("\nSetting up callbacks...")
    callback_list = get_callbacks(
        model_name=model_name,
        timestamp=timestamp,
        run_dir=run_dir,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        checkpoint_epochs=[20, 30, 50],
        config_dict=config_dict
    )

    # Train model
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callback_list,
        class_weight=class_weights,
        verbose=1
    )

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"All results saved to: {run_dir}")
    print(f"  - Best model: model_best.keras")
    print(f"  - Epoch checkpoints: model_epoch20.keras, model_epoch30.keras, model_epoch50.keras")
    print(f"  - Configuration: config.json")
    print(f"  - Training history: training_history.json")
    print(f"  - Checkpoint metrics: epoch20_metrics.json, epoch30_metrics.json, epoch50_metrics.json")
    print("="*80)

    return model, history


def evaluate_model(model: tf.keras.Model, batch_size: int = None):
    """
    Evaluate a trained model on the test set.

    Args:
        model: Trained Keras model
        batch_size: Batch size (uses config.BATCH_SIZE if None)

    Returns:
        Dictionary of evaluation metrics
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    print("\n" + "="*80)
    print("Evaluating on Test Set")
    print("="*80 + "\n")

    # Load test dataset
    test_ds = breakhis_data_loader.create_dataset('test', is_training=False, batch_size=batch_size)

    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    print("\n" + "="*80)
    print("Test Results")
    print("="*80)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("="*80)

    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }


def main():
    """Main training script. All configuration is read from config.py."""
    # Train model using config settings
    model, history = train_model(
        model_name=config.DEFAULT_MODEL,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.INITIAL_LEARNING_RATE,
        use_class_weights=config.USE_CLASS_WEIGHTS
    )

    # Evaluate if configured
    if config.EVALUATE_AFTER_TRAINING:
        evaluate_model(model, batch_size=config.BATCH_SIZE)


if __name__ == "__main__":
    main()
