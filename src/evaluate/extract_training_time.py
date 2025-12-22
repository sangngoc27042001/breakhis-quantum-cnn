"""
Extract precise training time from TensorBoard event files.
This provides the most accurate training duration by reading event timestamps.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


@dataclass
class TrainingTime:
    """Training time information for a model."""
    model_name: str
    folder_name: str
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    duration_minutes: float
    duration_hours: float
    total_epochs: int
    seconds_per_epoch: float

    def __str__(self):
        return (f"{self.model_name}: {self.duration_minutes:.2f} min "
                f"({self.seconds_per_epoch:.2f} sec/epoch)")


def get_event_file(tensorboard_dir: Path) -> Path:
    """Find the TensorBoard event file in a directory."""
    event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {tensorboard_dir}")
    return event_files[0]  # Usually there's only one


def extract_training_time(tensorboard_dir: Path, model_name: str = None) -> TrainingTime:
    """
    Extract training time from TensorBoard event files.

    Args:
        tensorboard_dir: Path to TensorBoard log directory
        model_name: Optional model name (derived from parent folder if not provided)

    Returns:
        TrainingTime object with timing information
    """
    if not TENSORBOARD_AVAILABLE:
        raise ImportError("TensorBoard is required. Install with: pip install tensorboard")

    # Load events
    ea = event_accumulator.EventAccumulator(str(tensorboard_dir))
    ea.Reload()

    # Get scalar tags
    tags = ea.Tags().get('scalars', [])
    if not tags:
        raise ValueError(f"No scalar data found in {tensorboard_dir}")

    # Use any available tag to get timestamps
    tag = tags[0]
    events = ea.Scalars(tag)

    if not events:
        raise ValueError(f"No events found for tag {tag}")

    # Extract timing information
    start_time = events[0].wall_time
    end_time = events[-1].wall_time
    duration = end_time - start_time
    total_steps = len(events)

    # Get model name from folder structure if not provided
    if model_name is None:
        folder_name = tensorboard_dir.parent.name
        model_name = folder_name.split('_202')[0]  # Remove timestamp
    else:
        folder_name = tensorboard_dir.parent.name

    return TrainingTime(
        model_name=model_name,
        folder_name=folder_name,
        start_timestamp=start_time,
        end_timestamp=end_time,
        duration_seconds=duration,
        duration_minutes=duration / 60,
        duration_hours=duration / 3600,
        total_epochs=total_steps,
        seconds_per_epoch=duration / total_steps if total_steps > 0 else 0
    )


def extract_all_training_times(results_dir: str = "results") -> List[TrainingTime]:
    """
    Extract training times for all models in the results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        List of TrainingTime objects
    """
    results_path = Path(results_dir)
    all_times = []

    for folder in results_path.iterdir():
        if not folder.is_dir() or folder.name.startswith('.'):
            continue

        tensorboard_dir = folder / "tensorboard"
        if not tensorboard_dir.exists():
            print(f"Warning: No tensorboard directory in {folder.name}")
            continue

        try:
            time_info = extract_training_time(tensorboard_dir)
            all_times.append(time_info)
        except Exception as e:
            print(f"Warning: Could not extract time from {folder.name}: {e}")

    return all_times


def print_training_times(times: List[TrainingTime]):
    """Print training times in a formatted table."""
    if not times:
        print("No training times found!")
        return

    print("\n" + "=" * 100)
    print("TRAINING TIME ANALYSIS")
    print("=" * 100)

    # Sort by duration
    sorted_times = sorted(times, key=lambda x: x.duration_seconds)

    print("\nMODELS RANKED BY TRAINING TIME (Fastest to Slowest):")
    print("-" * 100)
    print(f"{'Rank':<6} {'Model':<35} {'Total Time':<20} {'Time/Epoch':<20} {'Epochs':<10}")
    print("-" * 100)

    for i, t in enumerate(sorted_times, 1):
        time_str = f"{t.duration_minutes:.2f} min ({t.duration_hours:.2f} hr)"
        epoch_str = f"{t.seconds_per_epoch:.2f} sec"
        print(f"{i:<6} {t.model_name:<35} {time_str:<20} {epoch_str:<20} {t.total_epochs:<10}")

    # Statistics
    print("\n" + "=" * 100)
    print("STATISTICS:")
    print("-" * 100)

    avg_time = sum(t.duration_minutes for t in times) / len(times)
    fastest = sorted_times[0]
    slowest = sorted_times[-1]

    print(f"Average training time:    {avg_time:.2f} minutes")
    print(f"Fastest model:            {fastest.model_name} ({fastest.duration_minutes:.2f} min)")
    print(f"Slowest model:            {slowest.model_name} ({slowest.duration_minutes:.2f} min)")
    print(f"Speed difference:         {slowest.duration_minutes / fastest.duration_minutes:.2f}x")

    print("\n" + "=" * 100)


def save_to_csv(times: List[TrainingTime], output_file: str = "training_times.csv"):
    """Save training times to CSV file."""
    if not times:
        print("No times to save!")
        return

    data = []
    for t in times:
        data.append({
            'model_name': t.model_name,
            'folder_name': t.folder_name,
            'duration_minutes': t.duration_minutes,
            'duration_hours': t.duration_hours,
            'duration_seconds': t.duration_seconds,
            'total_epochs': t.total_epochs,
            'seconds_per_epoch': t.seconds_per_epoch
        })

    df = pd.DataFrame(data)
    df = df.sort_values('duration_seconds')
    df.to_csv(output_file, index=False)

    print(f"\nTraining times saved to: {output_file}")
    return df


def compare_speed_vs_accuracy(results_dir: str = "results"):
    """Compare training speed vs final accuracy."""
    import json

    times = extract_all_training_times(results_dir)
    if not times:
        print("No training data found!")
        return

    results_path = Path(results_dir)
    comparison = []

    for time_info in times:
        folder = results_path / time_info.folder_name
        history_file = folder / "training_history.json"

        if not history_file.exists():
            continue

        try:
            with open(history_file, 'r') as f:
                history = json.load(f)

            val_acc = history.get('val_acc', [])
            best_val_acc = max(val_acc) if val_acc else 0
            final_val_acc = val_acc[-1] if val_acc else 0

            comparison.append({
                'model_name': time_info.model_name,
                'training_minutes': time_info.duration_minutes,
                'best_val_acc': best_val_acc,
                'final_val_acc': final_val_acc,
                'efficiency': best_val_acc / time_info.duration_minutes if time_info.duration_minutes > 0 else 0
            })
        except Exception as e:
            print(f"Warning: Could not load history for {time_info.model_name}: {e}")

    if not comparison:
        print("No comparison data available!")
        return

    # Print comparison
    print("\n" + "=" * 110)
    print("TRAINING EFFICIENCY: SPEED vs ACCURACY")
    print("=" * 110)
    print(f"{'Model':<35} {'Time (min)':<15} {'Best Val Acc':<15} {'Final Val Acc':<15} {'Efficiency':<15}")
    print("-" * 110)

    # Sort by efficiency (accuracy per minute)
    sorted_comp = sorted(comparison, key=lambda x: x['efficiency'], reverse=True)

    for comp in sorted_comp:
        print(f"{comp['model_name']:<35} {comp['training_minutes']:<15.2f} "
              f"{comp['best_val_acc']:<15.2f} {comp['final_val_acc']:<15.2f} "
              f"{comp['efficiency']:<15.2f}")

    print("\n" + "=" * 110)
    print("Note: Efficiency = Best Validation Accuracy / Training Time (minutes)")
    print("      Higher efficiency means better accuracy achieved in less time")
    print("=" * 110)

    # Save to CSV
    df = pd.DataFrame(sorted_comp)
    df.to_csv('speed_vs_accuracy.csv', index=False)
    print("\nSpeed vs accuracy comparison saved to: speed_vs_accuracy.csv")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract training times from TensorBoard logs')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Path to results directory')
    parser.add_argument('--output', type=str, default='training_times.csv',
                        help='Output CSV file path')
    parser.add_argument('--compare', action='store_true',
                        help='Compare speed vs accuracy')

    args = parser.parse_args()

    if not TENSORBOARD_AVAILABLE:
        print("ERROR: TensorBoard is required for this script.")
        print("Install with: pip install tensorboard")
        return

    print(f"Extracting training times from: {args.results_dir}")

    # Extract times
    times = extract_all_training_times(args.results_dir)

    if not times:
        print("No training data found!")
        return

    # Print times
    print_training_times(times)

    # Save to CSV
    save_to_csv(times, args.output)

    # Compare if requested
    if args.compare:
        compare_speed_vs_accuracy(args.results_dir)


if __name__ == "__main__":
    main()
