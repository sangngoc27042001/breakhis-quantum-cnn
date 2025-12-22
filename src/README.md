# BreakHis Classification Pipeline

A complete pipeline for training deep learning models on the BreakHis breast cancer histopathological image dataset.

## Project Structure

```
src/
├── config.py                      # Configuration file with all hyperparameters
├── data_preparation.py            # Dataset download, split, and augmentation
├── breakhis_data_loader.py        # TensorFlow data pipeline utilities
├── train.py                       # Main training script
├── model_implementations/         # Model architecture definitions
│   ├── __init__.py
│   ├── convnext_tiny.py          # ConvNeXt Tiny model
│   ├── densenet121.py            # DenseNet121 with optional feature fusion
│   ├── efficientnetb0.py         # EfficientNetB0 model
│   ├── resnet50.py               # ResNet50 model
│   └── mobilenetv2.py            # MobileNetV2 lightweight model
└── README.md                      # This file
```

## Installation

Install required dependencies:

```bash
pip install tensorflow keras kagglehub numpy pandas scikit-learn Pillow tqdm
```

## Usage

### 1. Data Preparation

First, download and prepare the BreakHis dataset:

```python
# Run data preparation script
python data_preparation.py
```

This will:
- Download the BreakHis dataset from Kaggle
- Split into train (70%), validation (15%), and test (15%) sets
- Resize all images to 224×224
- Balance training data using augmentation (flips and rotations)
- Save processed dataset to `./processed_breakhis_balanced/`

### 2. Training

Train a model using the main training script:

```bash
# Train ConvNeXt Tiny (default)
python train.py

# Train a specific model
python train.py --model densenet121

# Custom hyperparameters
python train.py --model efficientnetb0 --epochs 100 --batch-size 64 --lr 0.0001

# Train and evaluate
python train.py --model resnet50 --evaluate
```

#### Available Models

- `convnext_tiny` - ConvNeXt Tiny (default, good balance of accuracy and speed)
- `densenet121` - DenseNet121 with multi-level feature fusion
- `efficientnetb0` - EfficientNetB0 (efficient and accurate)
- `resnet50` - ResNet50 (classic architecture)
- `mobilenetv2` - MobileNetV2 (lightweight, fast inference)

#### Training Arguments

- `--model`: Model architecture to use
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Initial learning rate (default: 1e-4)
- `--no-class-weights`: Disable class weights for imbalanced data
- `--evaluate`: Evaluate on test set after training

### 3. Configuration

All hyperparameters and settings can be modified in [config.py](config.py):

```python
# Key configurations
IMG_SIZE = (224, 224)           # Input image size
BATCH_SIZE = 32                 # Training batch size
EPOCHS = 50                     # Number of epochs
INITIAL_LEARNING_RATE = 1e-4    # Initial learning rate
DROPOUT_RATE = 0.45             # Dropout rate
L2_REG = 0.001                  # L2 regularization
USE_MIXED_PRECISION = True      # Enable mixed precision training
```

### 4. Testing Individual Components

Test the data loader:

```bash
python breakhis_data_loader.py
```

Test a model architecture:

```bash
python model_implementations/convnext_tiny.py
python model_implementations/densenet121.py
```

## Dataset Information

### BreakHis Dataset

The BreakHis dataset contains 7,909 microscopic images of breast tumor tissue collected from 82 patients. Images are classified into 8 classes:

**Benign (4 classes):**
- Adenosis
- Fibroadenoma
- Phyllodes Tumor
- Tubular Adenoma

**Malignant (4 classes):**
- Ductal Carcinoma
- Lobular Carcinoma
- Mucinous Carcinoma
- Papillary Carcinoma

### Class Distribution (After Processing)

After balancing, each class in the training set will have an equal number of samples (2,416 per class).

## Model Architectures

### ConvNeXt Tiny
- Base: ConvNeXt Tiny pretrained on ImageNet
- Head: GlobalAveragePooling2D → BatchNorm → Dropout → Dense(8)
- Parameters: ~28M
- Good balance of accuracy and efficiency

### DenseNet121
- Base: DenseNet121 pretrained on ImageNet
- Optional multi-level feature fusion from 3 dense blocks
- Head: Fusion → Dense layers → Dropout → Dense(8)
- Parameters: ~7-8M
- Excellent for feature extraction

### EfficientNetB0
- Base: EfficientNetB0 pretrained on ImageNet
- Head: GlobalAveragePooling2D → BatchNorm → Dropout → Dense(8)
- Parameters: ~5M
- Most efficient model

### ResNet50
- Base: ResNet50 pretrained on ImageNet
- Head: GlobalAveragePooling2D → BatchNorm → Dropout → Dense(8)
- Parameters: ~25M
- Classic and reliable

### MobileNetV2
- Base: MobileNetV2 pretrained on ImageNet
- Head: GlobalAveragePooling2D → BatchNorm → Dropout → Dense(8)
- Parameters: ~3M
- Fastest inference, suitable for deployment

## Training Features

- **Mixed Precision Training**: Faster training with FP16
- **Class Weights**: Handle class imbalance
- **Data Augmentation**: Balanced training set with augmented samples
- **Callbacks**:
  - ModelCheckpoint: Save best model
  - ReduceLROnPlateau: Adaptive learning rate
  - TensorBoard: Training visualization
  - CSVLogger: Save training history

## Output

Training produces the following outputs:

```
checkpoints/              # Saved models
├── convnext_tiny_YYYYMMDD_HHMMSS_best.keras
└── ...

logs/                     # TensorBoard logs
├── convnext_tiny_YYYYMMDD_HHMMSS/
└── ...

results/                  # Training history CSVs
├── convnext_tiny_YYYYMMDD_HHMMSS_history.csv
└── ...
```

## Example: Complete Workflow

```bash
# 1. Prepare dataset (run once)
python data_preparation.py

# 2. Train ConvNeXt Tiny model
python train.py --model convnext_tiny --epochs 50 --evaluate

# 3. Train other models for comparison
python train.py --model densenet121 --epochs 50 --evaluate
python train.py --model efficientnetb0 --epochs 50 --evaluate

# 4. View results in TensorBoard
tensorboard --logdir logs/
```

## Customization

### Adding a New Model

1. Create a new file in `model_implementations/`:

```python
# model_implementations/my_model.py
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from src import config

def build_model(num_classes=None, input_shape=None, **kwargs):
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if input_shape is None:
        input_shape = config.INPUT_SHAPE

    inputs = Input(shape=input_shape)

    # Your model architecture here
    base_model = tf.keras.applications.YourModel(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
```

2. Register it in `train.py`:

```python
from src.model_implementations.my_model import build_model as build_my_model

MODEL_REGISTRY = {
    ...
    'my_model': build_my_model,
}
```

3. Train it:

```bash
python train.py --model my_model
```

## Performance Tips

1. **Use Mixed Precision**: Already enabled by default in config
2. **Adjust Batch Size**: Larger batch sizes for faster training (if GPU memory allows)
3. **Learning Rate**: Start with 1e-4, reduce if training is unstable
4. **Early Stopping**: Monitor validation loss to avoid overfitting
5. **Class Weights**: Keep enabled for balanced performance across all classes

## Citation

If you use this code or the BreakHis dataset, please cite:

```
F. A. Spanhol, L. S. Oliveira, C. Petitjean and L. Heutte,
"A Dataset for Breast Cancer Histopathological Image Classification,"
in IEEE Transactions on Biomedical Engineering, vol. 63, no. 7, pp. 1455-1462, July 2016.
```

## License

This project is for research purposes. Please refer to the original BreakHis dataset license for data usage terms.
ok