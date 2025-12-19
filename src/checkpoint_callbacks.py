"""
Custom callbacks for saving models at specific epochs and logging metrics.
"""
import os
import json
import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np
from typing import Dict, List


class EpochCheckpointCallback(callbacks.Callback):
    """
    Custom callback to save models at specific epochs and evaluate them.

    Saves:
    - Model checkpoint at specified epochs (20, 30, 50)
    - Evaluation metrics (train, val, test) for each checkpoint
    - Training configuration
    - Full training history
    """

    def __init__(self,
                 checkpoint_epochs: List[int],
                 model_name: str,
                 timestamp: str,
                 run_dir: str,
                 train_ds,
                 val_ds,
                 test_ds,
                 config_dict: Dict):
        """
        Initialize the checkpoint callback.

        Args:
            checkpoint_epochs: List of epochs at which to save checkpoints (e.g., [20, 30, 50])
            model_name: Name of the model being trained
            timestamp: Timestamp string for this training run
            run_dir: Directory for this specific training run
            train_ds: Training dataset for evaluation
            val_ds: Validation dataset for evaluation
            test_ds: Test dataset for evaluation
            config_dict: Dictionary containing model configuration
        """
        super().__init__()
        self.checkpoint_epochs = sorted(checkpoint_epochs)
        self.model_name = model_name
        self.timestamp = timestamp
        self.run_dir = run_dir
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.config_dict = config_dict

        # Store training history across all epochs
        self.full_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        # Store checkpoint metrics
        self.checkpoint_metrics = {}

        # Ensure run directory exists
        os.makedirs(run_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.
        Saves model and evaluates if epoch is in checkpoint_epochs.
        """
        logs = logs or {}
        current_epoch = epoch + 1  # Keras uses 0-indexed epochs

        # Store history for this epoch
        self.full_history['loss'].append(float(logs.get('loss', 0)))
        self.full_history['accuracy'].append(float(logs.get('accuracy', 0)))
        self.full_history['val_loss'].append(float(logs.get('val_loss', 0)))
        self.full_history['val_accuracy'].append(float(logs.get('val_accuracy', 0)))

        # Get learning rate (handle different optimizer types)
        try:
            lr = float(self.model.optimizer.learning_rate.numpy())
        except:
            lr = float(self.model.optimizer.learning_rate)
        self.full_history['learning_rate'].append(lr)

        # Check if we should save a checkpoint at this epoch
        if current_epoch in self.checkpoint_epochs:
            print(f"\n{'='*80}")
            print(f"Checkpoint at Epoch {current_epoch}")
            print(f"{'='*80}")

            # Save model
            model_path = os.path.join(
                self.run_dir,
                f'model_epoch{current_epoch}.keras'
            )
            self.model.save(model_path)
            print(f"Model saved to: {model_path}")

            # Evaluate on all datasets
            print("\nEvaluating on all datasets...")
            metrics = self._evaluate_all_datasets()

            # Store metrics for this checkpoint
            self.checkpoint_metrics[f'epoch_{current_epoch}'] = metrics

            # Save metrics to JSON
            self._save_checkpoint_metrics(current_epoch, metrics)

            print(f"{'='*80}\n")

    def on_train_end(self, logs=None):
        """
        Called at the end of training.
        Saves full training history and configuration.
        """
        print(f"\n{'='*80}")
        print("Saving Training Summary")
        print(f"{'='*80}")

        # Save full training history
        history_path = os.path.join(
            self.run_dir,
            'training_history.json'
        )
        with open(history_path, 'w') as f:
            json.dump(self.full_history, f, indent=2)
        print(f"Full training history saved to: {history_path}")

        # Save configuration
        config_path = os.path.join(
            self.run_dir,
            'config.json'
        )
        with open(config_path, 'w') as f:
            json.dump(self.config_dict, f, indent=2)
        print(f"Training configuration saved to: {config_path}")

        # Save summary of all checkpoint metrics
        summary_path = os.path.join(
            self.run_dir,
            'checkpoint_summary.json'
        )
        summary = {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'checkpoint_epochs': self.checkpoint_epochs,
            'checkpoints': self.checkpoint_metrics,
            'configuration': self.config_dict
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Checkpoint summary saved to: {summary_path}")

        print(f"{'='*80}\n")

    def _evaluate_all_datasets(self) -> Dict:
        """
        Evaluate model on train, validation, and test datasets.

        Returns:
            Dictionary containing metrics for all datasets
        """
        metrics = {}

        # Evaluate on training set
        print("  - Evaluating on training set...")
        train_loss, train_acc = self.model.evaluate(self.train_ds, verbose=0)
        metrics['train_loss'] = float(train_loss)
        metrics['train_accuracy'] = float(train_acc)
        print(f"    Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Evaluate on validation set
        print("  - Evaluating on validation set...")
        val_loss, val_acc = self.model.evaluate(self.val_ds, verbose=0)
        metrics['val_loss'] = float(val_loss)
        metrics['val_accuracy'] = float(val_acc)
        print(f"    Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Evaluate on test set
        print("  - Evaluating on test set...")
        test_loss, test_acc = self.model.evaluate(self.test_ds, verbose=0)
        metrics['test_loss'] = float(test_loss)
        metrics['test_accuracy'] = float(test_acc)
        print(f"    Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        return metrics

    def _save_checkpoint_metrics(self, epoch: int, metrics: Dict):
        """
        Save metrics for a specific checkpoint epoch to a JSON file.

        Args:
            epoch: The epoch number
            metrics: Dictionary of metrics for this checkpoint
        """
        metrics_path = os.path.join(
            self.run_dir,
            f'epoch{epoch}_metrics.json'
        )

        checkpoint_data = {
            'epoch': epoch,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'metrics': metrics
        }

        with open(metrics_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"  Metrics saved to: {metrics_path}")


class BestModelTracker(callbacks.Callback):
    """
    Tracks the best model based on validation accuracy and saves detailed metrics.
    Works in conjunction with standard ModelCheckpoint callback.
    """

    def __init__(self,
                 model_name: str,
                 timestamp: str,
                 run_dir: str,
                 monitor: str = 'val_accuracy',
                 mode: str = 'max'):
        """
        Initialize the best model tracker.

        Args:
            model_name: Name of the model
            timestamp: Timestamp for this training run
            run_dir: Directory for this specific training run
            monitor: Metric to monitor (default: 'val_accuracy')
            mode: 'max' or 'min' (default: 'max')
        """
        super().__init__()
        self.model_name = model_name
        self.timestamp = timestamp
        self.run_dir = run_dir
        self.monitor = monitor
        self.mode = mode

        self.best_epoch = 0
        self.best_value = -float('inf') if mode == 'max' else float('inf')

        os.makedirs(run_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """Track best model based on monitored metric."""
        logs = logs or {}
        current_value = logs.get(self.monitor)

        if current_value is None:
            return

        # Check if this is the best model so far
        is_better = (self.mode == 'max' and current_value > self.best_value) or \
                   (self.mode == 'min' and current_value < self.best_value)

        if is_better:
            self.best_epoch = epoch + 1
            self.best_value = current_value

    def on_train_end(self, logs=None):
        """Save information about the best model."""
        best_model_info = {
            'best_epoch': self.best_epoch,
            'monitored_metric': self.monitor,
            'best_value': float(self.best_value),
            'mode': self.mode
        }

        info_path = os.path.join(
            self.run_dir,
            'best_model_info.json'
        )

        with open(info_path, 'w') as f:
            json.dump(best_model_info, f, indent=2)

        print(f"\nBest model info saved to: {info_path}")
        print(f"Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch}")
