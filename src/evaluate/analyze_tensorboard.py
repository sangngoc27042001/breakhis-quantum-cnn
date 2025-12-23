"""
Analyze TensorBoard logs to extract training time and performance metrics.
This script parses TensorFlow event files to get comprehensive training statistics.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


@dataclass
class ModelTrainingStats:
    """Statistics for a single model training run."""
    model_name: str
    folder_name: str
    start_time: float
    end_time: float
    training_duration_seconds: float
    training_duration_minutes: float
    total_epochs: int

    # Best metrics
    best_val_acc: float
    best_val_acc_epoch: int

    # Final epoch metrics (epoch 20)
    final_train_loss: float
    final_train_acc: float
    final_val_loss: float
    final_val_acc: float
    final_test_loss: float
    final_test_acc: float

    # Early training metrics (epoch 5)
    early_train_acc: float
    early_val_acc: float

    # Learning characteristics
    avg_train_acc_gain_per_epoch: float
    avg_val_acc_gain_per_epoch: float

    # Model info
    total_params: int = 0
    trainable_params: int = 0

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


def load_tensorboard_events(logdir: str) -> Dict:
    """
    Load TensorBoard events from a log directory.

    Args:
        logdir: Path to TensorBoard log directory

    Returns:
        Dictionary containing scalars data
    """
    if not TENSORBOARD_AVAILABLE:
        raise ImportError("TensorBoard is not installed")

    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    # Get all scalar tags
    tags = ea.Tags().get('scalars', [])

    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = [(e.wall_time, e.step, e.value) for e in events]

    return data


def analyze_training_from_json(result_folder: Path) -> ModelTrainingStats:
    """
    Analyze training from JSON files (training_history.json and config.json).

    Args:
        result_folder: Path to the result folder

    Returns:
        ModelTrainingStats object
    """
    # Load training history
    history_file = result_folder / "training_history.json"
    config_file = result_folder / "config.json"

    if not history_file.exists():
        raise FileNotFoundError(f"training_history.json not found in {result_folder}")

    with open(history_file, 'r') as f:
        history = json.load(f)

    # Load config if available
    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    # Extract model name from folder
    folder_name = result_folder.name
    model_name = folder_name.split('_202')[0]  # Get model name before timestamp

    # Get epoch counts
    total_epochs = len(history.get('train_loss', []))

    # Calculate training time from folder timestamp if available
    # Format: modelname_YYYYMMDD_HHMMSS
    parts = folder_name.split('_')
    if len(parts) >= 3:
        # This is a rough estimate - actual time should come from timestamps
        training_duration = config.get('training_time_seconds', 0)
    else:
        training_duration = 0

    # Get wall time from TensorBoard if available
    start_time = 0
    end_time = 0

    tensorboard_dir = result_folder / "tensorboard"
    if tensorboard_dir.exists() and TENSORBOARD_AVAILABLE:
        try:
            tb_data = load_tensorboard_events(str(tensorboard_dir))
            # Get timestamps from any available metric
            for tag, events in tb_data.items():
                if events:
                    start_time = events[0][0]  # First event wall_time
                    end_time = events[-1][0]  # Last event wall_time
                    training_duration = end_time - start_time
                    break
        except Exception as e:
            print(f"Warning: Could not load TensorBoard data for {folder_name}: {e}")

    # Extract metrics
    train_acc = history.get('train_acc', [])
    val_acc = history.get('val_acc', [])
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])

    # Best validation accuracy
    best_val_acc = max(val_acc) if val_acc else 0
    best_val_acc_epoch = val_acc.index(best_val_acc) + 1 if val_acc else 0

    # Final epoch metrics
    final_train_loss = train_loss[-1] if train_loss else 0
    final_train_acc = train_acc[-1] if train_acc else 0
    final_val_loss = val_loss[-1] if val_loss else 0
    final_val_acc = val_acc[-1] if val_acc else 0

    # Early training metrics (epoch 5)
    early_epoch_idx = min(4, len(train_acc) - 1) if train_acc else 0
    early_train_acc = train_acc[early_epoch_idx] if train_acc else 0
    early_val_acc = val_acc[early_epoch_idx] if val_acc else 0

    # Learning characteristics
    if len(train_acc) > 1:
        avg_train_acc_gain = (train_acc[-1] - train_acc[0]) / (len(train_acc) - 1)
    else:
        avg_train_acc_gain = 0

    if len(val_acc) > 1:
        avg_val_acc_gain = (val_acc[-1] - val_acc[0]) / (len(val_acc) - 1)
    else:
        avg_val_acc_gain = 0

    # Get model parameters from config
    total_params = config.get('total_params', 0)
    trainable_params = config.get('trainable_params', 0)

    # Load final epoch metrics for test accuracy
    final_test_loss = 0
    final_test_acc = 0
    final_metrics_file = result_folder / f"epoch{total_epochs}_metrics.json"
    if final_metrics_file.exists():
        try:
            with open(final_metrics_file, 'r') as f:
                final_metrics = json.load(f)
                final_test_loss = final_metrics.get('test_loss', 0)
                final_test_acc = final_metrics.get('test_acc', 0)
        except Exception as e:
            print(f"Warning: Could not load final metrics for {folder_name}: {e}")

    return ModelTrainingStats(
        model_name=model_name,
        folder_name=folder_name,
        start_time=start_time,
        end_time=end_time,
        training_duration_seconds=training_duration,
        training_duration_minutes=training_duration / 60 if training_duration > 0 else 0,
        total_epochs=total_epochs,
        best_val_acc=best_val_acc,
        best_val_acc_epoch=best_val_acc_epoch,
        final_train_loss=final_train_loss,
        final_train_acc=final_train_acc,
        final_val_loss=final_val_loss,
        final_val_acc=final_val_acc,
        final_test_loss=final_test_loss,
        final_test_acc=final_test_acc,
        early_train_acc=early_train_acc,
        early_val_acc=early_val_acc,
        avg_train_acc_gain_per_epoch=avg_train_acc_gain,
        avg_val_acc_gain_per_epoch=avg_val_acc_gain,
        total_params=total_params,
        trainable_params=trainable_params
    )


def analyze_all_models(results_dir: str = "results") -> List[ModelTrainingStats]:
    """
    Analyze all model training runs in the results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        List of ModelTrainingStats objects
    """
    results_path = Path(results_dir)
    all_stats = []

    for folder in results_path.iterdir():
        if not folder.is_dir() or folder.name.startswith('.'):
            continue

        try:
            stats = analyze_training_from_json(folder)
            all_stats.append(stats)
        except Exception as e:
            print(f"Warning: Could not analyze {folder.name}: {e}")

    return all_stats


def print_training_summary(stats_list: List[ModelTrainingStats]):
    """Print a summary of all training runs."""
    if not stats_list:
        print("No training runs found!")
        return

    print("\n" + "=" * 100)
    print("TRAINING TIME AND PERFORMANCE SUMMARY")
    print("=" * 100)

    # Sort by training time
    sorted_by_time = sorted(stats_list, key=lambda x: x.training_duration_seconds)

    print("\nFASTEST TRAINING MODELS:")
    print("-" * 120)
    print(f"{'Rank':<6} {'Model':<25} {'Time (min)':<12} {'Final Val Acc':<15} {'Best Val Acc':<15} {'Test Acc':<15}")
    print("-" * 120)

    for i, stats in enumerate(sorted_by_time, 1):
        time_str = f"{stats.training_duration_minutes:.2f}" if stats.training_duration_minutes > 0 else "N/A"
        test_acc_str = f"{stats.final_test_acc:.2f}" if stats.final_test_acc > 0 else "N/A"
        print(f"{i:<6} {stats.model_name:<25} {time_str:<12} {stats.final_val_acc:<15.2f} {stats.best_val_acc:<15.2f} {test_acc_str:<15}")

    # Sort by best validation accuracy
    sorted_by_acc = sorted(stats_list, key=lambda x: x.best_val_acc, reverse=True)

    print("\n" + "=" * 120)
    print("BEST PERFORMING MODELS (by validation accuracy):")
    print("-" * 120)
    print(f"{'Rank':<6} {'Model':<25} {'Best Val Acc':<15} {'Epoch':<8} {'Final Val Acc':<15} {'Test Acc':<15}")
    print("-" * 120)

    for i, stats in enumerate(sorted_by_acc, 1):
        test_acc_str = f"{stats.final_test_acc:.2f}" if stats.final_test_acc > 0 else "N/A"
        print(f"{i:<6} {stats.model_name:<25} {stats.best_val_acc:<15.2f} {stats.best_val_acc_epoch:<8} {stats.final_val_acc:<15.2f} {test_acc_str:<15}")

    # Early learning speed
    sorted_by_early = sorted(stats_list, key=lambda x: x.early_val_acc, reverse=True)

    print("\n" + "=" * 120)
    print("FASTEST LEARNING MODELS (validation accuracy at epoch 5):")
    print("-" * 120)
    print(f"{'Rank':<6} {'Model':<25} {'Early Val Acc':<15} {'Early Train Acc':<15}")
    print("-" * 120)

    for i, stats in enumerate(sorted_by_early, 1):
        print(f"{i:<6} {stats.model_name:<25} {stats.early_val_acc:<15.2f} {stats.early_train_acc:<15.2f}")

    # Sort by test accuracy
    sorted_by_test = sorted([s for s in stats_list if s.final_test_acc > 0],
                           key=lambda x: x.final_test_acc, reverse=True)

    if sorted_by_test:
        print("\n" + "=" * 120)
        print("BEST TEST ACCURACY:")
        print("-" * 120)
        print(f"{'Rank':<6} {'Model':<25} {'Test Acc':<15} {'Val Acc':<15} {'Generalization Gap':<20}")
        print("-" * 120)

        for i, stats in enumerate(sorted_by_test, 1):
            gen_gap = stats.final_val_acc - stats.final_test_acc
            print(f"{i:<6} {stats.model_name:<25} {stats.final_test_acc:<15.2f} {stats.final_val_acc:<15.2f} {gen_gap:<20.2f}")

    print("\n" + "=" * 120)


def save_results_to_csv(stats_list: List[ModelTrainingStats], output_file: str = "model_comparison.csv"):
    """Save results to CSV file."""
    if not stats_list:
        print("No stats to save!")
        return

    # Convert to DataFrame
    data = [stats.to_dict() for stats in stats_list]
    df = pd.DataFrame(data)

    # Sort by best validation accuracy
    df = df.sort_values('best_val_acc', ascending=False)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return df


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze TensorBoard logs and training results')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Path to results directory')
    parser.add_argument('--output', type=str, default='model_comparison.csv',
                        help='Output CSV file path')

    args = parser.parse_args()

    print(f"Analyzing results from: {args.results_dir}")

    # Analyze all models
    stats_list = analyze_all_models(args.results_dir)

    if not stats_list:
        print("No training runs found!")
        return

    # Print summary
    print_training_summary(stats_list)

    # Save to CSV
    save_results_to_csv(stats_list, args.output)

    # Additional statistics
    print("\n" + "=" * 120)
    print("OVERALL STATISTICS:")
    print("-" * 120)

    if stats_list[0].training_duration_seconds > 0:
        avg_time = sum(s.training_duration_minutes for s in stats_list) / len(stats_list)
        fastest = min(stats_list, key=lambda x: x.training_duration_seconds)
        slowest = max(stats_list, key=lambda x: x.training_duration_seconds)

        print(f"Average training time:     {avg_time:.2f} minutes")
        print(f"Fastest model:             {fastest.model_name} ({fastest.training_duration_minutes:.2f} min)")
        print(f"Slowest model:             {slowest.model_name} ({slowest.training_duration_minutes:.2f} min)")

    avg_best_acc = sum(s.best_val_acc for s in stats_list) / len(stats_list)
    best_model = max(stats_list, key=lambda x: x.best_val_acc)

    print(f"Average best val acc:      {avg_best_acc:.2f}%")
    print(f"Best performing model:     {best_model.model_name} ({best_model.best_val_acc:.2f}%)")

    # Test accuracy statistics
    models_with_test = [s for s in stats_list if s.final_test_acc > 0]
    if models_with_test:
        avg_test_acc = sum(s.final_test_acc for s in models_with_test) / len(models_with_test)
        best_test_model = max(models_with_test, key=lambda x: x.final_test_acc)
        avg_gen_gap = sum(s.final_val_acc - s.final_test_acc for s in models_with_test) / len(models_with_test)

        print(f"Average test acc:          {avg_test_acc:.2f}%")
        print(f"Best test model:           {best_test_model.model_name} ({best_test_model.final_test_acc:.2f}%)")
        print(f"Avg generalization gap:    {avg_gen_gap:.2f}% (val - test)")

    print("=" * 120)


if __name__ == "__main__":
    main()
