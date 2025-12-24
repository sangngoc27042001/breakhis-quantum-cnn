"""
Script to load a trained model and evaluate it on the test set.
Prints model architecture summary and returns test set accuracy.

Usage:
    uv run python -m src.evaluate.evaluate_model --model_dir results/mobilenetv3_small_100_20251223_224717
    uv run python -m src.evaluate.evaluate_model --model_dir results/mobilenetv3_small_100_20251223_224717 --checkpoint model_epoch20.pth
"""
import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torchinfo import summary

from src import config
from src import breakhis_data_loader
from src.model_implementations import build_cnn_quantum, build_cnn_classical


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model, input_shape=(1, 3, 224, 224)):
    """Print detailed model architecture using torchinfo."""
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)

    try:
        summary(
            model,
            input_size=input_shape,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            depth=3,
            verbose=1
        )
    except Exception as e:
        print(f"Could not generate detailed summary: {e}")
        print("\nModel structure:")
        print(model)

    print("=" * 80 + "\n")


def load_model_from_checkpoint(model_dir, checkpoint_name='model_best.pth'):
    """
    Load a trained model from checkpoint.

    Args:
        model_dir: Path to the results directory containing the checkpoint
        checkpoint_name: Name of the checkpoint file (default: 'model_best.pth')

    Returns:
        Tuple of (model, config_dict)
    """
    # Load config to determine model type and architecture
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        saved_config = json.load(f)

    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Model name: {saved_config['model_name']}")

    # Determine if this is a quantum model
    is_quantum = (saved_config['model_name'] == 'cnn_quantum')

    # Build the model architecture
    if is_quantum:
        print("\nBuilding quantum CNN model...")
        print(f"  Backbone: {saved_config['quantum_cnn_backbone']}")
        print(f"  N_qubits: {saved_config['quantum_cnn_n_qubits']}")
        print(f"  Dense encoding: {saved_config['quantum_cnn_dense_encoding_method']}")
        print(f"  Dense template: {saved_config['quantum_cnn_dense_template']}")
        print(f"  Dense depth: {saved_config['quantum_cnn_dense_depth']}")

        model = build_cnn_quantum(
            num_classes=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE,
            l2_reg=config.L2_REG,
            backbone=saved_config['quantum_cnn_backbone'],
            n_qubits=saved_config['quantum_cnn_n_qubits'],
            dense_encoding_method=saved_config['quantum_cnn_dense_encoding_method'],
            dense_template=saved_config['quantum_cnn_dense_template'],
            dense_depth=saved_config['quantum_cnn_dense_depth'],
        )
    else:
        print("\nBuilding classical CNN model...")
        model = build_cnn_classical(
            model_name=saved_config['model_name'],
            num_classes=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE,
            l2_reg=config.L2_REG,
        )

    # Load checkpoint
    checkpoint_path = os.path.join(model_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\n✓ Model loaded successfully from epoch {checkpoint.get('epoch', 'N/A') + 1}")
    print(f"✓ Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    print("=" * 80 + "\n")

    return model, saved_config


def evaluate_on_test_set(model, device='cpu', batch_size=None):
    """
    Evaluate model on the test set.

    Args:
        model: Trained model
        device: Device to run evaluation on
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Load test data
    print("Loading test dataset...")
    test_loader = breakhis_data_loader.create_dataloader(
        'test',
        is_training=False,
        batch_size=batch_size
    )
    print(f"✓ Test set loaded: {len(test_loader.dataset)} samples\n")

    # Move model to device
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Evaluation
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    print("Evaluating on test set...")
    print("-" * 40)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(
        description='Load and evaluate a trained model on the test set'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Path to the model results directory (e.g., results/mobilenetv3_small_100_20251223_224717)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='model_best.pth',
        help='Name of the checkpoint file (default: model_best.pth)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run evaluation on (default: cpu)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size for evaluation (default: use config.BATCH_SIZE)'
    )
    parser.add_argument(
        '--no_summary',
        action='store_true',
        help='Skip printing model architecture summary'
    )

    args = parser.parse_args()

    # Verify model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)

    # Load model
    model, saved_config = load_model_from_checkpoint(
        args.model_dir,
        args.checkpoint
    )

    # Print parameter count
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Print model architecture summary
    if not args.no_summary:
        print_model_summary(model, input_shape=(1, 3, 224, 224))

    # Evaluate on test set
    test_loss, test_acc = evaluate_on_test_set(
        model,
        device=args.device,
        batch_size=args.batch_size
    )

    # Print results
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("=" * 80 + "\n")

    # Compare with training metrics if available
    metrics_file = os.path.join(args.model_dir, f'epoch{saved_config.get("best_epoch", 20)}_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        print("\n" + "=" * 80)
        print("COMPARISON WITH TRAINING METRICS")
        print("=" * 80)
        print(f"Best epoch: {saved_config.get('best_epoch', 'N/A')}")
        print(f"Train Accuracy: {metrics.get('train_acc', 'N/A'):.2f}%")
        print(f"Val Accuracy: {metrics.get('val_acc', 'N/A'):.2f}%")
        print(f"Test Accuracy (from training): {metrics.get('test_acc', 'N/A'):.2f}%")
        print(f"Test Accuracy (current): {test_acc:.2f}%")
        print("=" * 80 + "\n")

    return test_acc


if __name__ == "__main__":
    main()
