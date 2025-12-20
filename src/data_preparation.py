"""
Data preparation utilities for BreakHis dataset.
Handles dataset splitting, augmentation, and balancing.
"""
import os
import glob
import itertools
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


def download_breakhis_dataset(kaggle_dataset: str = "ambarish/breakhis") -> str:
    """
    Download BreakHis dataset from Kaggle.

    Args:
        kaggle_dataset: Kaggle dataset identifier

    Returns:
        Path to downloaded dataset
    """
    try:
        import kagglehub
        path = kagglehub.dataset_download(kaggle_dataset)
        print(f"Dataset downloaded to: {path}")
        return path
    except ImportError:
        raise ImportError("kagglehub not installed. Install with: pip install kagglehub")


def load_breakhis_paths(base_path: str) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Load all image paths from BreakHis dataset directory structure.

    Args:
        base_path: Root path to BreakHis dataset

    Returns:
        Tuple of (dataset_map, X_all, y_all) where:
            - dataset_map: Dict mapping class names to list of image paths
            - X_all: List of all image paths
            - y_all: List of all labels (class names)
    """
    # Get all PNG files
    breast_img_paths = glob.glob(
        f'{base_path}/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/**/*.png',
        recursive=True
    )

    print(f"Found {len(breast_img_paths)} images in dataset")

    # Initialize storage for each class
    A, F, PT, TA = [], [], [], []  # Benign subtypes
    DC, LC, MC, PC = [], [], [], []  # Malignant subtypes

    # Classify images based on filename convention
    for img in breast_img_paths:
        img_name = Path(img).name

        # Extract class from filename (character at position 6)
        if img_name[6] == 'A':
            A.append(img)
        elif img_name[6] == 'F':
            F.append(img)
        elif img_name[6] == 'P' and img_name[7] == 'T':
            PT.append(img)
        elif img_name[6] == 'T':
            TA.append(img)
        elif img_name[6] == 'D':
            DC.append(img)
        elif img_name[6] == 'L':
            LC.append(img)
        elif img_name[6] == 'M':
            MC.append(img)
        elif img_name[6] == 'P':
            PC.append(img)

    # Create dataset map
    dataset_map = {
        'Adenosis': A,
        'Fibroadenoma': F,
        'Phyllodes_Tumor': PT,
        'Tubular_Adenoma': TA,
        'Ductal_Carcinoma': DC,
        'Lobular_Carcinoma': LC,
        'Mucinous_Carcinoma': MC,
        'Papillary_Carcinoma': PC
    }

    # Print statistics
    print("\nClass distribution:")
    print("Benign:")
    print(f"  Adenosis: {len(A)}")
    print(f"  Fibroadenoma: {len(F)}")
    print(f"  Phyllodes Tumor: {len(PT)}")
    print(f"  Tubular Adenoma: {len(TA)}")
    print("\nMalignant:")
    print(f"  Ductal Carcinoma: {len(DC)}")
    print(f"  Lobular Carcinoma: {len(LC)}")
    print(f"  Mucinous Carcinoma: {len(MC)}")
    print(f"  Papillary Carcinoma: {len(PC)}")

    # Consolidate into X and y arrays
    X_all = []
    y_all = []

    for class_name, file_list in dataset_map.items():
        X_all.extend(file_list)
        y_all.extend([class_name] * len(file_list))

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    return dataset_map, X_all, y_all


def split_dataset(X: np.ndarray,
                  y: np.ndarray,
                  train_ratio: float = 0.70,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        X: Array of file paths
        y: Array of labels
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    print(f"\nSplitting data: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")

    # First split: Train vs Temp (Val+Test)
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        stratify=y,
        random_state=random_state
    )

    # Second split: Val vs Test
    test_ratio_adjusted = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_ratio_adjusted,
        stratify=y_temp,
        random_state=random_state
    )

    print(f"Train size: {len(X_train)}")
    print(f"Val size:   {len(X_val)}")
    print(f"Test size:  {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split(X: np.ndarray,
               y: np.ndarray,
               split_name: str,
               output_dir: str,
               target_size: Tuple[int, int] = (224, 224)):
    """
    Save a dataset split to disk, organized by class.

    Args:
        X: Array of source file paths
        y: Array of labels
        split_name: Name of the split ('train', 'val', 'test')
        output_dir: Base output directory
        target_size: Target image size (width, height)
    """
    print(f"\nSaving {split_name} data to disk...")

    for src_path, label in tqdm(zip(X, y), total=len(X)):
        dest_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)

        try:
            img = Image.open(src_path)
            img = img.resize(target_size)
            img.save(os.path.join(dest_dir, Path(src_path).name))
        except Exception as e:
            print(f"Error saving {src_path}: {e}")


def balance_training_data(train_dir: str,
                          class_names: List[str],
                          target_count: int = None):
    """
    Balance training data using augmentation (flips and rotations).
    Augments minority classes to match the majority class count.

    Args:
        train_dir: Directory containing training data
        class_names: List of class names
        target_count: Target number of samples per class (uses max if None)
    """
    print("\nBalancing training data with augmentation...")

    # Count samples per class
    counts = {}
    for cls in class_names:
        cls_path = os.path.join(train_dir, cls)
        if os.path.exists(cls_path):
            counts[cls] = len([f for f in os.listdir(cls_path) if f.endswith('.png')])
        else:
            counts[cls] = 0

    # Determine target count
    if target_count is None:
        target_count = max(counts.values())

    print(f"Target count per class: {target_count}")

    # Define augmentation options
    flip_options = [
        None,
        [Image.FLIP_LEFT_RIGHT],
        [Image.FLIP_TOP_BOTTOM],
        [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
    ]

    rot_options = [
        None,
        Image.ROTATE_90,
        Image.ROTATE_180,
        Image.ROTATE_270
    ]

    # Generate all combinations (excluding no augmentation)
    aug_combinations = list(itertools.product(flip_options, rot_options))[1:]

    # Augment each class
    for cls in class_names:
        cls_path = os.path.join(train_dir, cls)
        current_count = counts[cls]
        needed = target_count - current_count

        if needed <= 0:
            print(f"  {cls}: Already balanced ({current_count} samples)")
            continue

        print(f"  {cls}: Generating {needed} new images")

        existing_files = [
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.endswith('.png')
        ]

        generated = 0
        num_files = len(existing_files)
        num_combos = len(aug_combinations)

        while generated < needed:
            # Cycle through files and augmentation combinations
            src_img_path = existing_files[generated % num_files]
            img = Image.open(src_img_path)

            # Get augmentation combination
            combo_index = generated % num_combos
            current_flips, current_rot = aug_combinations[combo_index]

            # Apply flips
            if current_flips is not None:
                for flip_op in current_flips:
                    img = img.transpose(flip_op)

            # Apply rotation
            if current_rot is not None:
                img = img.transpose(current_rot)

            # Create filename tags
            flip_tag = "noFlip"
            if current_flips == [Image.FLIP_LEFT_RIGHT]:
                flip_tag = "flipLR"
            elif current_flips == [Image.FLIP_TOP_BOTTOM]:
                flip_tag = "flipTB"
            elif len(current_flips or []) == 2:
                flip_tag = "flipBoth"

            rot_tag = "noRot"
            if current_rot == Image.ROTATE_90:
                rot_tag = "rot90"
            elif current_rot == Image.ROTATE_180:
                rot_tag = "rot180"
            elif current_rot == Image.ROTATE_270:
                rot_tag = "rot270"

            # Save augmented image
            new_filename = f"aug_{generated}_{flip_tag}_{rot_tag}_{Path(src_img_path).name}"
            img.save(os.path.join(cls_path, new_filename))

            generated += 1

    print("Balancing complete!")


def prepare_breakhis_dataset(kaggle_dataset: str = "ambarish/breakhis",
                              output_dir: str = "./processed_breakhis_balanced",
                              target_size: Tuple[int, int] = (224, 224),
                              train_ratio: float = 0.70,
                              val_ratio: float = 0.15,
                              test_ratio: float = 0.15,
                              balance_train: bool = True,
                              random_state: int = 42):
    """
    Complete pipeline to prepare BreakHis dataset.

    Args:
        kaggle_dataset: Kaggle dataset identifier
        output_dir: Output directory for processed dataset
        target_size: Target image size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        balance_train: Whether to balance training data
        random_state: Random seed
    """
    print("="*80)
    print("BreakHis Dataset Preparation Pipeline")
    print("="*80)

    # Step 1: Download dataset
    print("\n[1/5] Downloading dataset...")
    dataset_path = download_breakhis_dataset(kaggle_dataset)

    # Step 2: Load and organize paths
    print("\n[2/5] Loading dataset paths...")
    dataset_map, X_all, y_all = load_breakhis_paths(dataset_path)

    # Step 3: Split dataset
    print("\n[3/5] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X_all, y_all,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )

    # Step 4: Save splits
    print("\n[4/5] Saving splits to disk...")
    save_split(X_val, y_val, 'val', output_dir, target_size)
    save_split(X_test, y_test, 'test', output_dir, target_size)
    save_split(X_train, y_train, 'train', output_dir, target_size)

    # Step 5: Balance training data
    if balance_train:
        print("\n[5/5] Balancing training data...")
        train_dir = os.path.join(output_dir, 'train')
        class_names = list(dataset_map.keys())
        balance_training_data(train_dir, class_names)
    else:
        print("\n[5/5] Skipping data balancing")

    print("\n" + "="*80)
    print(f"Dataset preparation complete!")
    print(f"Processed dataset saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    # Run the complete preparation pipeline
    prepare_breakhis_dataset(
        kaggle_dataset="ambarish/breakhis",
        output_dir="./processed_breakhis_balanced_50_25_25",
        target_size=(224, 224),
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        balance_train=True,
        random_state=42
    )
