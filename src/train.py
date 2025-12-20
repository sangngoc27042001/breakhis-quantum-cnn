"""
Training script for BreakHis classification models using PyTorch.
All configuration is controlled via config.py.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import json
import csv
from tqdm import tqdm

from src import config
from src import breakhis_data_loader
from src.model_implementations import (
    build_vgg16,
    build_efficientnetv2b3,
    build_densenet169,
    build_mobilenetv3large,
    build_nasnetmobile,
    build_cnn_quantum,
)


# Model registry
MODEL_REGISTRY = {
    "vgg16": build_vgg16,
    "efficientnetv2b3": build_efficientnetv2b3,
    "densenet169": build_densenet169,
    "mobilenetv3large": build_mobilenetv3large,
    "nasnetmobile": build_nasnetmobile,
    "cnn_quantum": build_cnn_quantum,
}


def get_device():
    """Auto-detect and return the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def train_epoch(model, train_loader, criterion, optimizer, device, class_weights_tensor=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Apply class weights if provided
        if class_weights_tensor is not None:
            weights = class_weights_tensor[labels]
            loss = criterion(outputs, labels)
            loss = (loss * weights).mean()
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def get_predictions(model, dataloader, device):
    """Get predictions for all samples in dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def save_checkpoint(model, optimizer, epoch, best_val_acc, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }, filepath)


def train_model(model_name: str,
                epochs: int = None,
                batch_size: int = None,
                learning_rate: float = None,
                use_class_weights: bool = True):
    """
    Train a model on the BreakHis dataset.
    
    Args:
        model_name: Name of model to train
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        use_class_weights: Whether to use class weights
        
    Returns:
        Tuple of (model, history dict)
    """
    # Use config defaults if not specified
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if learning_rate is None:
        learning_rate = config.INITIAL_LEARNING_RATE
    
    # Get device
    device = get_device()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name == "cnn_quantum":
        run_name = config.QUANTUM_CNN_CONFIG_COMBINED_NAME
    else:
        run_name = model_name
    run_dir = os.path.join(config.RESULTS_DIR, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Model: {model_name}")
    if model_name == "cnn_quantum":
        print(f"Quantum-CNN config: backbone={config.QUANTUM_CNN_CONFIG_BACKBONE}, "
              f"pooling_depth={config.QUANTUM_CNN_CONFIG_POOLING_DEPTH}, "
              f"dense_encoding_method={config.QUANTUM_CNN_CONFIG_DENSE_ENCODING_METHOD}, "
              f"dense_depth={config.QUANTUM_CNN_CONFIG_DENSE_DEPTH}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Use class weights: {use_class_weights}")
    print(f"Results directory: {run_dir}")
    print("=" * 80 + "\n")
    
    # Load datasets
    print("Loading datasets...")
    train_loader = breakhis_data_loader.create_dataloader('train', is_training=True, batch_size=batch_size)
    val_loader = breakhis_data_loader.create_dataloader('val', is_training=False, batch_size=batch_size)
    test_loader = breakhis_data_loader.create_dataloader('test', is_training=False, batch_size=batch_size)
    
    # Get class weights if requested
    class_weights_tensor = None
    if use_class_weights:
        class_weights = torch.tensor([config.CLASS_WEIGHTS[i] for i in range(config.NUM_CLASSES)], dtype=torch.float32)
        class_weights_tensor = class_weights.to(device)
        print("\nUsing class weights:")
        for class_idx, weight in config.CLASS_WEIGHTS.items():
            class_name = [k for k, v in config.CLASS_MAP.items() if v == class_idx][0]
            print(f"  {class_name:20} (class {class_idx}): {weight:.4f}")
        print()
    
    # Build model
    print(f"\nBuilding {model_name} model...")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    build_kwargs = dict(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        l2_reg=config.L2_REG,
    )
    
    if model_name == "cnn_quantum":
        build_kwargs.update(
            backbone=config.QUANTUM_CNN_CONFIG_BACKBONE,
            pooling_depth=config.QUANTUM_CNN_CONFIG_POOLING_DEPTH,
            dense_encoding_method=config.QUANTUM_CNN_CONFIG_DENSE_ENCODING_METHOD,
            dense_depth=config.QUANTUM_CNN_CONFIG_DENSE_DEPTH,
        )
    
    model = MODEL_REGISTRY[model_name](**build_kwargs)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(reduction='none' if use_class_weights else 'mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.L2_REG)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config.LR_REDUCTION_FACTOR,
                                                      patience=config.LR_REDUCTION_PATIENCE, )
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(run_dir, 'tensorboard'))
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, class_weights_tensor)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_checkpoint(model, optimizer, epoch, best_val_acc,
                          os.path.join(run_dir, 'model_best.pth'))
            print(f"✓ Saved best model (val_acc: {best_val_acc:.2f}%)")
        
        # Save checkpoints at specific epochs
        if (epoch + 1) in [20, 30, 50]:
            checkpoint_path = os.path.join(run_dir, f'model_epoch{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch + 1}")
            
            # Evaluate on all datasets
            print(f"\n  Evaluating on all datasets...")
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            }
            
            with open(os.path.join(run_dir, f'epoch{epoch + 1}_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best model: Epoch {best_epoch} with val_acc: {best_val_acc:.2f}%")
    print(f"All results saved to: {run_dir}")
    print("=" * 80)
    
    # Save training history
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config_dict = {
        'model_name': model_name,
        'run_name': run_name,
        'batch_size': batch_size,
        'epochs': epochs,
        'initial_learning_rate': learning_rate,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'timestamp': timestamp
    }
    
    if model_name == "cnn_quantum":
        config_dict.update({
            'quantum_cnn_backbone': config.QUANTUM_CNN_CONFIG_BACKBONE,
            'quantum_cnn_pooling_depth': config.QUANTUM_CNN_CONFIG_POOLING_DEPTH,
            'quantum_cnn_dense_encoding_method': config.QUANTUM_CNN_CONFIG_DENSE_ENCODING_METHOD,
            'quantum_cnn_dense_depth': config.QUANTUM_CNN_CONFIG_DENSE_DEPTH,
        })
    
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    writer.close()
    
    return model, history


def main():
    """Main training function."""
    # Use default model from config
    model_name = config.DEFAULT_MODEL
    
    print(f"\nTraining {model_name} model...")
    model, history = train_model(
        model_name=model_name,
        use_class_weights=config.USE_CLASS_WEIGHTS
    )
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
