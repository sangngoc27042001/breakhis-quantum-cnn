"""
Script to compress the results folder into a zip archive.
"""
import os
import zipfile
from datetime import datetime
import argparse


def zip_results(results_dir='./results', output_dir='./archives', zip_name=None):
    """
    Compress the results folder into a zip archive.

    Args:
        results_dir: Path to the results directory to zip
        output_dir: Directory where the zip file will be saved
        zip_name: Name of the output zip file (without .zip extension).
                  If None, will use timestamp.

    Returns:
        Path to the created zip file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if results directory exists
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory not found: {results_dir}")

    # Generate zip file name
    if zip_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_name = f'results_{timestamp}'

    zip_path = os.path.join(output_dir, f'{zip_name}.zip')

    print(f"Compressing results folder: {results_dir}")
    print(f"Output zip file: {zip_path}")
    print("-" * 80)

    # Create zip file
    file_count = 0
    total_size = 0

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files in results directory
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Get relative path for the archive
                arcname = os.path.relpath(file_path, os.path.dirname(results_dir))

                # Add file to zip
                zipf.write(file_path, arcname)

                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1

                print(f"Added: {arcname} ({file_size:,} bytes)")

    # Get zip file size
    zip_size = os.path.getsize(zip_path)
    compression_ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0

    print("-" * 80)
    print(f"Compression complete!")
    print(f"  Total files: {file_count}")
    print(f"  Original size: {total_size:,} bytes ({total_size / (1024*1024):.2f} MB)")
    print(f"  Compressed size: {zip_size:,} bytes ({zip_size / (1024*1024):.2f} MB)")
    print(f"  Compression ratio: {compression_ratio:.1f}%")
    print(f"  Zip file saved to: {zip_path}")
    print("-" * 80)

    return zip_path


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Compress the results folder into a zip archive.'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Path to the results directory to zip (default: ./results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./archives',
        help='Directory where the zip file will be saved (default: ./archives)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Name of the output zip file without .zip extension (default: results_TIMESTAMP)'
    )

    args = parser.parse_args()

    try:
        zip_path = zip_results(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            zip_name=args.name
        )
        print(f"\nSuccess! Results compressed to: {zip_path}")

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
