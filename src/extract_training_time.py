"""
Script to extract training time from TensorFlow event files.
"""
import os
from pathlib import Path
from tensorflow.python.summary.summary_iterator import summary_iterator
import argparse


def get_training_time_from_events(event_file):
    """
    Extract training time from a TensorFlow event file.

    Args:
        event_file: Path to the TensorFlow event file

    Returns:
        Training time in seconds, or None if not found
    """
    try:
        events = list(summary_iterator(str(event_file)))

        if len(events) < 2:
            return None

        # Get first and last event timestamps
        start_time = events[0].wall_time
        end_time = events[-1].wall_time

        # Calculate duration in seconds
        duration = end_time - start_time

        return duration

    except Exception as e:
        print(f"Error reading {event_file}: {e}")
        return None


def format_time(seconds):
    """
    Format seconds into human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds is None:
        return "N/A"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def extract_all_training_times(results_dir='./results'):
    """
    Extract training times for all models in the results directory.

    Args:
        results_dir: Path to the results directory

    Returns:
        Dictionary mapping model names to training times
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        raise ValueError(f"Results directory not found: {results_dir}")

    training_times = {}

    # Iterate through all model directories
    for model_dir in sorted(results_path.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Look for train event files
        train_event_files = list(model_dir.glob("tensorboard_logs/train/events.out.tfevents.*"))

        if not train_event_files:
            print(f"No train event files found for {model_name}")
            continue

        # Use the first (or only) event file
        event_file = train_event_files[0]

        print(f"Processing {model_name}...")
        training_time = get_training_time_from_events(event_file)

        if training_time is not None:
            training_times[model_name] = {
                'seconds': training_time,
                'formatted': format_time(training_time)
            }
            print(f"  Training time: {format_time(training_time)} ({training_time:.2f} seconds)")
        else:
            training_times[model_name] = {
                'seconds': None,
                'formatted': 'N/A'
            }
            print(f"  Training time: N/A")

    return training_times


def generate_markdown_table(training_times):
    """
    Generate a markdown table of training times.

    Args:
        training_times: Dictionary of model names to training time info

    Returns:
        Markdown table string
    """
    lines = [
        "## Training Times",
        "",
        "| Model | Training Time | Duration (seconds) |",
        "|-------|--------------|-------------------|"
    ]

    for model_name in sorted(training_times.keys()):
        info = training_times[model_name]
        formatted = info['formatted']
        seconds = f"{info['seconds']:.2f}" if info['seconds'] is not None else "N/A"

        lines.append(f"| {model_name} | {formatted} | {seconds} |")

    lines.append("")

    return "\n".join(lines)


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Extract training times from TensorFlow event files.'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Path to the results directory (default: ./results)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for markdown table (default: print to stdout)'
    )

    args = parser.parse_args()

    try:
        print("Extracting training times from TensorFlow event files...")
        print("-" * 80)

        training_times = extract_all_training_times(args.results_dir)

        print("-" * 80)
        print(f"\nFound training times for {len(training_times)} models")
        print("")

        # Generate markdown table
        markdown = generate_markdown_table(training_times)

        if args.output:
            output_path = Path(args.output)
            output_path.write_text(markdown)
            print(f"Markdown table saved to: {args.output}")
        else:
            print(markdown)

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
