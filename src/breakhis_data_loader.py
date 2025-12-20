"""
Data loader utilities for BreakHis dataset using PyTorch.
Handles loading images from directory structure and creating DataLoaders.
"""
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, List, Dict
from collections import Counter
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


class BreakHisDataset(Dataset):
    """PyTorch Dataset for BreakHis images."""

    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        """
        Initialize the dataset.

        Args:
            file_paths: List of paths to image files
            labels: List of integer labels
            transform: Optional torchvision transforms to apply
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


def get_transforms(is_training: bool = False) -> transforms.Compose:
    """
    Get image transforms for preprocessing.

    Args:
        is_training: Whether this is for training (adds augmentation)

    Returns:
        torchvision transforms composition
    """
    if is_training:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    return transform


def create_dataloader(split_name: str,
                      is_training: bool = False,
                      batch_size: int = None,
                      num_workers: int = 4,
                      pin_memory: bool = True) -> DataLoader:
    """
    Create a PyTorch DataLoader for a specific split (train/val/test).

    Args:
        split_name: Name of the split ('train', 'val', or 'test')
        is_training: Whether this is training data (enables shuffling and augmentation)
        batch_size: Batch size (uses config.BATCH_SIZE if None)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        PyTorch DataLoader ready for training/evaluation
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

    # Get transforms
    transform = get_transforms(is_training=is_training)

    # Create dataset
    dataset = BreakHisDataset(paths, labels, transform=transform)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return dataloader


def get_class_distribution(dataloader: DataLoader) -> Dict[int, int]:
    """
    Compute class distribution from a dataloader.

    Args:
        dataloader: PyTorch DataLoader to analyze

    Returns:
        Dictionary mapping class indices to their counts
    """
    labels = []
    for _, batch_labels in dataloader:
        labels.extend(batch_labels.tolist())

    return dict(Counter(labels))


def compute_class_weights(dataloader: DataLoader, num_classes: int = None) -> Dict[int, float]:
    """
    Compute class weights for imbalanced dataset.
    Uses inverse frequency: weight_i = N_total / (N_classes * N_i)

    Args:
        dataloader: PyTorch DataLoader to compute weights from
        num_classes: Number of classes (uses config.NUM_CLASSES if None)

    Returns:
        Dictionary mapping class indices to their weights
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    # Get class distribution
    class_counts = get_class_distribution(dataloader)
    N_total = sum(class_counts.values())

    # Compute weights
    class_weights = {}
    for class_idx in range(num_classes):
        N_i = class_counts.get(class_idx, 1)  # Avoid division by zero
        weight = N_total / (num_classes * N_i)
        class_weights[class_idx] = weight

    return class_weights


def create_all_dataloaders(batch_size: int = None,
                          num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create train, validation, and test dataloaders.

    Args:
        batch_size: Batch size for all dataloaders (uses config.BATCH_SIZE if None)
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("Creating data loaders...")

    train_loader = create_dataloader('train', is_training=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = create_dataloader('val', is_training=False, batch_size=batch_size, num_workers=num_workers)
    test_loader = create_dataloader('test', is_training=False, batch_size=batch_size, num_workers=num_workers)

    print("\nDataLoader creation complete!")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing BreakHis data loader...")
    print(f"Configuration:")
    print(f"  Base directory: {config.BASE_DIR}")
    print(f"  Image size: {config.IMG_SIZE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Number of classes: {config.NUM_CLASSES}")
    print()

    # Create dataloaders
    train_loader, val_loader, test_loader = create_all_dataloaders()

    # Show class distribution
    print("\nClass distribution:")
    for split_name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        dist = get_class_distribution(loader)
        print(f"\n{split_name}:")
        for class_idx, count in sorted(dist.items()):
            class_name = [k for k, v in config.CLASS_MAP.items() if v == class_idx][0]
            print(f"  {class_name:20} (class {class_idx}): {count:5d} images")

    # Test batch shape
    print("\n\nTesting batch shapes:")
    for images, labels in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image dtype: {images.dtype}")
        print(f"Label dtype: {labels.dtype}")
        break
