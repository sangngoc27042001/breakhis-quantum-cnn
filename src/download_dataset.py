import os
import gdown
import zipfile
from pathlib import Path


def download_and_extract_dataset():
    """
    Download preprocessed dataset from Google Drive and extract to root folder.
    """
    # Google Drive file ID extracted from the URL
    # https://drive.google.com/file/d/1YYlMyzdnN6uMYRDUZsYfs-zkENZdjaCJ/view?usp=sharing
    # file_id = "1kledqy3CqTelZefA8D9YNtS1zUwb2T6V"
    file_id = "1YYlMyzdnN6uMYRDUZsYfs-zkENZdjaCJ"

    # Construct the direct download URL
    url = f"https://drive.google.com/uc?id={file_id}"

    # Get the root directory (where this script is located)
    root_dir = Path(".")

    # Output zip file path
    output_zip = root_dir / "processed_breakhis_balanced.zip"

    # Expected dataset directory after extraction
    dataset_dir = root_dir / "processed_breakhis_balanced"

    # Check if dataset already exists
    if dataset_dir.exists() and dataset_dir.is_dir():
        print(f"Dataset directory already exists at {dataset_dir}")
        print("Skipping download and extraction.")
        return

    # Check if zip file already exists
    if output_zip.exists():
        print(f"Zip file already exists at {output_zip}")
        print("Skipping download, proceeding to extraction...")
    else:
        print("Downloading dataset from Google Drive...")
        try:
            # Download the file using gdown
            gdown.download(url, str(output_zip), quiet=False)
            print(f"Download complete: {output_zip}")
        except Exception as e:
            print(f"Error during download: {e}")
            raise

    # Extract the zip file
    try:
        print(f"Extracting to {root_dir}...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        print("Extraction complete!")

        # Optionally, remove the zip file after extraction
        print("Cleaning up zip file...")
        output_zip.unlink()
        print("Done!")

    except Exception as e:
        print(f"Error during extraction: {e}")
        raise


if __name__ == "__main__":
    download_and_extract_dataset()
