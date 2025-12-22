"""
Configuration file for BreakHis dataset training.
Contains all hyperparameters, paths, and model settings.
"""
import os

# ============================================================================
# DATA PATHS
# ============================================================================
BASE_DIR = "./processed_breakhis_balanced"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

# ============================================================================
# IMAGE SETTINGS
# ============================================================================
IMG_SIZE = (224, 224)
IMG_CHANNELS = 3
INPUT_SHAPE = (*IMG_SIZE, IMG_CHANNELS)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Class mappings for BreakHis 8-class classification
CLASS_MAP = {
    'Adenosis': 0,
    'Fibroadenoma': 1,
    'Phyllodes_Tumor': 2,
    'Tubular_Adenoma': 3,
    'Ductal_Carcinoma': 4,
    'Lobular_Carcinoma': 5,
    'Mucinous_Carcinoma': 6,
    'Papillary_Carcinoma': 7
}

NUM_CLASSES = len(CLASS_MAP)

# Class weights for handling class imbalance (computed from validation set)
CLASS_WEIGHTS = {
    0: 2.212686567164179,    # Adenosis
    1: 0.975328947368421,    # Fibroadenoma
    2: 2.1801470588235294,   # Phyllodes_Tumor
    3: 1.7441176470588236,   # Tubular_Adenoma
    4: 0.2867504835589942,   # Ductal_Carcinoma
    5: 1.577127659574468,    # Lobular_Carcinoma
    6: 1.245798319327731,    # Mucinous_Carcinoma
    7: 1.7648809523809523    # Papillary_Carcinoma
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 256
EPOCHS = 20
INITIAL_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-6
# ReduceLROnPlateau settings
LR_REDUCTION_FACTOR = 0.5
LR_REDUCTION_PATIENCE = 0

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Regularization
L2_REG = 0.001
DROPOUT_RATE = 0.45

# Available small models (<7M parameters) - biggest from each family
AVAILABLE_SMALL_MODELS = [
    "mobilenetv3_small_100",  # 1.53M params, 96.5ms
    "mnasnet_100",            # 3.11M params, 226.52ms
    "regnetx_002",            # 2.32M params, 24.35ms (second fastest)
    "regnety_002",            # 2.80M params, 23.33ms (fastest)
    "ghostnet_100",           # 3.91M params, 62.38ms (largest small model)
    "efficientnet_lite0",     # 3.38M params, 285.4ms
    "mobilevit_xs",           # 1.94M params, 70.36ms (hybrid CNN-Transformer)
]

# Current backbones available (for reference - can be used with small_models.py)
AVAILABLE_BACKBONES = [
    "vgg16",
    "efficientnetv2_rw_s",
    "densenet169",
    "mobilenetv3_large_100",
    "nasnetamobile",
]

# Default model to use (can be any small model or "cnn_quantum")
DEFAULT_MODEL = "cnn_quantum"  # Fastest small model with good accuracy

# ---------------------------------------------------------------------------
# CNN-Quantum Hybrid Model configuration
# Only used when DEFAULT_MODEL == "cnn_quantum" (or when training that model)
# ---------------------------------------------------------------------------
# Keep these defaults aligned with src/model_implementations/cnn_quantum.py
# Can use any model from AVAILABLE_SMALL_MODELS as quantum model backbone
QUANTUM_CNN_CONFIG_BACKBONE = "regnetx_002"  # Can be any model from AVAILABLE_SMALL_MODELS

# QuantumDenseLayer hyperparameters
QUANTUM_CNN_CONFIG_NO_QUBITS = 16  # Number of qubits in the quantum dense layer
QUANTUM_CNN_CONFIG_DENSE_ENCODING_METHOD = "rotation"  # "amplitude" | "rotation"
QUANTUM_CNN_CONFIG_DENSE_TEMPLATE = "strong"  # "strong" | "two_design" | "basic"
QUANTUM_CNN_CONFIG_DENSE_DEPTH = 3

# Combined name for result folder naming / experiment tracking.
# Example: cnn_quantum_mobilenetv3large_dense-rotation_strong_depth1
QUANTUM_CNN_CONFIG_COMBINED_NAME = (
    f"cnn_quantum_"
    f"{QUANTUM_CNN_CONFIG_BACKBONE}_"
    f"dense-{QUANTUM_CNN_CONFIG_DENSE_ENCODING_METHOD}_"
    f"{QUANTUM_CNN_CONFIG_DENSE_TEMPLATE}_"
    f"depth-{QUANTUM_CNN_CONFIG_DENSE_DEPTH}"
    f"qubits-{QUANTUM_CNN_CONFIG_NO_QUBITS}"
)

# Training options
USE_CLASS_WEIGHTS = True
EVALUATE_AFTER_TRAINING = True

# ============================================================================
# TRAINING CALLBACKS
# ============================================================================
# ModelCheckpoint settings
CHECKPOINT_FILEPATH = 'best_breakhis_model.keras'
CHECKPOINT_MONITOR = 'val_accuracy'
CHECKPOINT_MODE = 'max'


# EarlyStopping settings (optional)
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = 'val_loss'

# ============================================================================
# MIXED PRECISION TRAINING
# ============================================================================
USE_MIXED_PRECISION = False
MIXED_PRECISION_DTYPE = 'float16'

# ============================================================================
# DATA AUGMENTATION (for online augmentation if needed)
# ============================================================================
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotation_range': 20,
    'zoom_range': 0.1,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'fill_mode': 'nearest'
}

# ============================================================================
# RANDOM SEED
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# LOGGING & CHECKPOINTS
# ============================================================================
LOG_DIR = "./logs"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

# Create directories if they don't exist
for directory in [LOG_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
