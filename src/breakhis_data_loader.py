"""
Data loader utilities for BreakHis dataset.
Handles loading images from directory structure and creating tf.data pipelines.
"""
import os
import glob
import tensorflow as tf
from typing import Tuple, List
from src import config


def get_paths_and_labels(directory: str) -> Tuple[List[str], List[int]]:
    """
    Recursively get all image paths and their corresponding labels from directory.

    Args:
        directory: Path to directory containing class subdirectories

    Returns:
        Tuple of (file_paths, labels) where labels are integer indices
    """
    file_paths = []
    labels = []

    # Get all class directories
    classes = [d for d in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, d))]

    for class_name in classes:
        if class_name not in config.CLASS_MAP:
            print(f"Warning: Unknown class '{class_name}' found in {directory}")
            continue

        class_dir = os.path.join(directory, class_name)
        label_idx = config.CLASS_MAP[class_name]

        # Get all PNG images in this class directory
        paths = glob.glob(os.path.join(class_dir, "*.png"))

        file_paths.extend(paths)
        labels.extend([label_idx] * len(paths))

    return file_paths, labels


def process_image(file_path: str, label: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load and preprocess a single image.

    Args:
        file_path: Path to image file
        label: Integer label for the image

    Returns:
        Tuple of (processed_image, label)
    """
    # Load image
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels=config.IMG_CHANNELS)

    # Resize to target size
    img = tf.image.resize(img, config.IMG_SIZE)

    # Normalize to [0, 1]
    img = img / 255.0

    # Cast to appropriate dtype (float16 for mixed precision, float32 otherwise)
    if config.USE_MIXED_PRECISION:
        img = tf.cast(img, tf.float16)
    else:
        img = tf.cast(img, tf.float32)

    return img, label


def create_dataset(split_name: str,
                   is_training: bool = False,
                   batch_size: int = None) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset for a specific split (train/val/test).

    Args:
        split_name: Name of the split ('train', 'val', or 'test')
        is_training: Whether this is training data (enables shuffling)
        batch_size: Batch size (uses config.BATCH_SIZE if None)

    Returns:
        tf.data.Dataset ready for training/evaluation
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Determine directory path
    if split_name == 'train':
        full_path = config.TRAIN_DIR
    elif split_name == 'val':
        full_path = config.VAL_DIR
    elif split_name == 'test':
        full_path = config.TEST_DIR
    else:
        raise ValueError(f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'")

    # Get all file paths and labels
    paths, labels = get_paths_and_labels(full_path)
    print(f"{split_name.capitalize()} set: Found {len(paths)} images across {len(set(labels))} classes")

    # Create dataset from paths and labels
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Shuffle if training
    if is_training:
        ds = ds.shuffle(buffer_size=len(paths), seed=config.RANDOM_SEED)

    # Map preprocessing function
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the data
    ds = ds.batch(batch_size)

    # Prefetch for performance
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_class_distribution(dataset: tf.data.Dataset) -> dict:
    """
    Compute class distribution from a dataset.

    Args:
        dataset: tf.data.Dataset to analyze

    Returns:
        Dictionary mapping class indices to their counts
    """
    from collections import Counter

    labels = []
    for _, y in dataset.unbatch():
        labels.append(int(y.numpy()))

    return dict(Counter(labels))


def compute_class_weights(dataset: tf.data.Dataset, num_classes: int = None) -> dict:
    """
    Compute class weights for imbalanced dataset.
    Uses inverse frequency: weight_i = N_total / (N_classes * N_i)

    Args:
        dataset: tf.data.Dataset to compute weights from
        num_classes: Number of classes (uses config.NUM_CLASSES if None)

    Returns:
        Dictionary mapping class indices to their weights
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    # Get class distribution
    class_counts = get_class_distribution(dataset)
    N_total = sum(class_counts.values())

    # Compute weights
    class_weights = {}
    for class_idx in range(num_classes):
        N_i = class_counts.get(class_idx, 1)  # Avoid division by zero
        weight = N_total / (num_classes * N_i)
        class_weights[class_idx] = weight

    return class_weights


def create_all_datasets(batch_size: int = None) -> Tuple[tf.data.Dataset,
                                                           tf.data.Dataset,
                                                           tf.data.Dataset]:
    """
    Convenience function to create train, validation, and test datasets.

    Args:
        batch_size: Batch size for all datasets (uses config.BATCH_SIZE if None)

    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    print("Creating data pipelines...")

    train_ds = create_dataset('train', is_training=True, batch_size=batch_size)
    val_ds = create_dataset('val', is_training=False, batch_size=batch_size)
    test_ds = create_dataset('test', is_training=False, batch_size=batch_size)

    print("\nDataset creation complete!")
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Test the data loader
    print("Testing BreakHis data loader...")
    print(f"Configuration:")
    print(f"  Base directory: {config.BASE_DIR}")
    print(f"  Image size: {config.IMG_SIZE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Number of classes: {config.NUM_CLASSES}")
    print()

    # Create datasets
    train_ds, val_ds, test_ds = create_all_datasets()

    # Show class distribution
    print("\nClass distribution:")
    for split_name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        dist = get_class_distribution(ds)
        print(f"\n{split_name}:")
        for class_idx, count in sorted(dist.items()):
            class_name = [k for k, v in config.CLASS_MAP.items() if v == class_idx][0]
            print(f"  {class_name:20} (class {class_idx}): {count:5d} images")

    # Test batch shape
    print("\n\nTesting batch shapes:")
    for images, labels in train_ds.take(1):
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image dtype: {images.dtype}")
        print(f"Label dtype: {labels.dtype}")
