"""
Evaluate training and inference time for each backbone model.
This script measures:
- Training time per epoch (on real BreakHis dataset)
- Inference time (single sample and batch)
- Parameters count
- Memory usage

Results are compared across all backbone models.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import config
from src.model_implementations import build_cnn_classical


def get_device():
    """Auto-detect and return the best available device."""
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


def clear_gpu_memory():
    """Clear GPU cache and force garbage collection."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def measure_inference_time(model, device, num_runs=100, warmup=10, batch_size=1):
    """
    Measure average inference time for a model.

    Args:
        model: PyTorch model
        device: Device to run on
        num_runs: Number of inference runs to average
        warmup: Number of warmup runs
        batch_size: Batch size for inference

    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            times.append(time.time() - start)

    avg_time_ms = (sum(times) / len(times)) * 1000
    return avg_time_ms


def measure_training_epoch_time(model, device, batch_size=256, num_batches=100):
    """
    Measure time to train one epoch on dummy data.

    Args:
        model: PyTorch model
        device: Device to run on
        batch_size: Batch size for training
        num_batches: Number of batches per epoch

    Returns:
        Epoch time in seconds
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.time()

    for _ in range(num_batches):
        # Generate dummy data
        inputs = torch.randn(batch_size, 3, 224, 224).to(device)
        labels = torch.randint(0, config.NUM_CLASSES, (batch_size,)).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    epoch_time = time.time() - start_time
    return epoch_time


def get_model_memory(model, device):
    """
    Estimate model memory usage.

    Args:
        model: PyTorch model
        device: Device the model is on

    Returns:
        Memory usage in MB
    """
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Run a forward pass to measure memory
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(dummy_input)

        torch.cuda.synchronize()
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return memory_mb
    else:
        # For CPU/MPS, estimate based on parameter count
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        return param_size_mb


def evaluate_backbone_timing(model_name, device, batch_size=256, num_inference_runs=100):
    """
    Evaluate timing for a single backbone model.

    Args:
        model_name: Name of the model to evaluate
        device: Device to run on
        batch_size: Batch size for training
        num_inference_runs: Number of inference runs to average

    Returns:
        Dictionary with timing results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")

    try:
        # Build model
        print(f"Building model...")
        model = build_cnn_classical(
            model_name=model_name,
            num_classes=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE,
            l2_reg=config.L2_REG,
        )
        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

        # Measure inference time - single sample
        print(f"\nMeasuring inference time (single sample, {num_inference_runs} runs)...")
        inference_time_single = measure_inference_time(
            model, device, num_runs=num_inference_runs, batch_size=1
        )
        print(f"  Inference time (single): {inference_time_single:.2f} ms")

        # Measure inference time - batch
        print(f"\nMeasuring inference time (batch of {batch_size}, {num_inference_runs} runs)...")
        inference_time_batch = measure_inference_time(
            model, device, num_runs=num_inference_runs, batch_size=batch_size
        )
        print(f"  Inference time (batch): {inference_time_batch:.2f} ms")
        print(f"  Inference time (per sample): {inference_time_batch/batch_size:.2f} ms")

        # Measure memory
        print(f"\nMeasuring memory usage...")
        memory_mb = get_model_memory(model, device)
        print(f"  Memory: {memory_mb:.2f} MB")

        # Measure training epoch time with dummy data
        print(f"\nMeasuring training time for 1 epoch (dummy data, 100 batches)...")
        epoch_time = measure_training_epoch_time(model, device, batch_size=batch_size, num_batches=100)
        print(f"  Training epoch time: {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes)")

        # Cleanup
        del model
        clear_gpu_memory()

        results = {
            'Model': model_name,
            'Parameters (M)': round(total_params / 1e6, 3),
            'Trainable Params (M)': round(trainable_params / 1e6, 3),
            'Inference Time - Single (ms)': round(inference_time_single, 2),
            'Inference Time - Batch (ms)': round(inference_time_batch, 2),
            'Inference Time - Per Sample (ms)': round(inference_time_batch / batch_size, 3),
            'Training Epoch Time (sec)': round(epoch_time, 2),
            'Training Epoch Time (min)': round(epoch_time / 60, 2),
            'Memory (MB)': round(memory_mb, 2),
        }

        print(f"\n{'='*80}")
        print(f"✓ Evaluation complete for {model_name}")
        print(f"{'='*80}")

        return results

    except Exception as e:
        print(f"\n✗ Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main evaluation function."""
    print("="*80)
    print("BACKBONE MODELS TIMING EVALUATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get device
    device = get_device()

    # Evaluation settings
    batch_size = 256
    num_inference_runs = 100
    num_batches = 100  # Number of batches for training epoch simulation

    print(f"\nEvaluation Settings:")
    print(f"  Batch size: {batch_size}")
    print(f"  Inference runs: {num_inference_runs}")
    print(f"  Training batches per epoch: {num_batches}")
    print(f"  Input size: 224x224")
    print(f"  Number of classes: {config.NUM_CLASSES}")
    print(f"  Using dummy data for speed")
    print("="*80)

    # Models to evaluate
    models_to_evaluate = config.AVAILABLE_SMALL_MODELS

    print(f"\nEvaluating {len(models_to_evaluate)} models:")
    for i, model_name in enumerate(models_to_evaluate, 1):
        print(f"  {i}. {model_name}")
    print("="*80)

    # Evaluate all models
    results = []

    for i, model_name in enumerate(models_to_evaluate, 1):
        print(f"\n\n[{i}/{len(models_to_evaluate)}] Processing: {model_name}")

        result = evaluate_backbone_timing(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            num_inference_runs=num_inference_runs
        )

        if result:
            results.append(result)

        # Small delay between models
        time.sleep(2)

    # Create DataFrame
    if not results:
        print("\n✗ No results to display!")
        return

    df = pd.DataFrame(results)

    # Create output directory
    output_dir = Path("results/timing_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = output_dir / f"backbone_timing_{timestamp}.csv"
    df.to_csv(csv_file, index=False)

    # Print results sorted by different metrics
    print("\n\n" + "="*80)
    print("RESULTS: SORTED BY INFERENCE TIME (SINGLE SAMPLE)")
    print("="*80)
    df_sorted = df.sort_values('Inference Time - Single (ms)')
    print(df_sorted.to_string(index=False))

    print("\n\n" + "="*80)
    print("RESULTS: SORTED BY TRAINING EPOCH TIME")
    print("="*80)
    df_sorted = df.sort_values('Training Epoch Time (sec)')
    print(df_sorted.to_string(index=False))

    print("\n\n" + "="*80)
    print("RESULTS: SORTED BY PARAMETERS")
    print("="*80)
    df_sorted = df.sort_values('Parameters (M)')
    print(df_sorted.to_string(index=False))

    # Print insights
    print("\n\n" + "="*80)
    print("INSIGHTS")
    print("="*80)

    # Fastest inference
    fastest_inference_idx = df['Inference Time - Single (ms)'].idxmin()
    fastest_inference = df.loc[fastest_inference_idx]
    print(f"\n🏆 Fastest Inference (single sample):")
    print(f"   Model: {fastest_inference['Model']}")
    print(f"   Time: {fastest_inference['Inference Time - Single (ms)']} ms")
    print(f"   Parameters: {fastest_inference['Parameters (M)']}M")

    # Fastest training
    fastest_training_idx = df['Training Epoch Time (sec)'].idxmin()
    fastest_training = df.loc[fastest_training_idx]
    print(f"\n🏆 Fastest Training (per epoch):")
    print(f"   Model: {fastest_training['Model']}")
    print(f"   Time: {fastest_training['Training Epoch Time (min)']} min/epoch")
    print(f"   Parameters: {fastest_training['Parameters (M)']}M")

    # Smallest model
    smallest_idx = df['Parameters (M)'].idxmin()
    smallest = df.loc[smallest_idx]
    print(f"\n🏆 Smallest Model:")
    print(f"   Model: {smallest['Model']}")
    print(f"   Parameters: {smallest['Parameters (M)']}M")
    print(f"   Inference: {smallest['Inference Time - Single (ms)']} ms")
    print(f"   Training: {smallest['Training Epoch Time (min)']} min/epoch")

    # Most memory efficient
    most_efficient_idx = df['Memory (MB)'].idxmin()
    most_efficient = df.loc[most_efficient_idx]
    print(f"\n🏆 Most Memory Efficient:")
    print(f"   Model: {most_efficient['Model']}")
    print(f"   Memory: {most_efficient['Memory (MB)']} MB")
    print(f"   Parameters: {most_efficient['Parameters (M)']}M")

    # Speed comparison
    slowest_inference_idx = df['Inference Time - Single (ms)'].idxmax()
    slowest_inference = df.loc[slowest_inference_idx]
    inference_speedup = slowest_inference['Inference Time - Single (ms)'] / fastest_inference['Inference Time - Single (ms)']

    slowest_training_idx = df['Training Epoch Time (sec)'].idxmax()
    slowest_training = df.loc[slowest_training_idx]
    training_speedup = slowest_training['Training Epoch Time (sec)'] / fastest_training['Training Epoch Time (sec)']

    print(f"\n📊 Speed Comparison:")
    print(f"   Inference: {fastest_inference['Model']} is {inference_speedup:.2f}x faster than {slowest_inference['Model']}")
    print(f"   Training: {fastest_training['Model']} is {training_speedup:.2f}x faster than {slowest_training['Model']}")

    # Estimated training time for full 20 epochs
    print(f"\n⏱️  Estimated Full Training Time (20 epochs):")
    for _, row in df.sort_values('Training Epoch Time (sec)').iterrows():
        total_time_hours = (row['Training Epoch Time (sec)'] * 20) / 3600
        print(f"   {row['Model']:<30} {total_time_hours:6.2f} hours")

    print("\n" + "="*80)
    print(f"✓ Results saved to: {csv_file}")
    print("="*80)
    print("\nEVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
