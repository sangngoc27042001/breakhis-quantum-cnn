"""
Benchmark script to test training and inference speed of small models.
Measures:
- Model parameters
- Inference time (forward pass)
- Training time (forward + backward pass)
- Memory usage
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from datetime import datetime
import os

from src import config
from src.model_implementations import build_small_model


def get_device():
    """Get the best available device."""
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


def measure_inference_time(model, input_size, device, num_runs=100, warmup=10):
    """
    Measure average inference time for a model.

    Args:
        model: PyTorch model
        input_size: Tuple of (batch_size, channels, height, width)
        device: Device to run on
        num_runs: Number of inference runs to average
        warmup: Number of warmup runs

    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

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


def measure_training_time(model, input_size, device, num_runs=50, warmup=5):
    """
    Measure average training time (forward + backward pass) for a model.

    Args:
        model: PyTorch model
        input_size: Tuple of (batch_size, channels, height, width)
        device: Device to run on
        num_runs: Number of training runs to average
        warmup: Number of warmup runs

    Returns:
        Average training time in milliseconds
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    batch_size = input_size[0]
    dummy_input = torch.randn(input_size).to(device)
    dummy_labels = torch.randint(0, config.NUM_CLASSES, (batch_size,)).to(device)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()

    # Synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(num_runs):
        start = time.time()

        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        times.append(time.time() - start)

    avg_time_ms = (sum(times) / len(times)) * 1000
    return avg_time_ms


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


def benchmark_model(model_name, device, batch_size=32, num_inference_runs=100, num_training_runs=50):
    """
    Benchmark a single model.

    Args:
        model_name: Name of the model to benchmark
        device: Device to run on
        batch_size: Batch size for training benchmark
        num_inference_runs: Number of inference runs
        num_training_runs: Number of training runs

    Returns:
        Dictionary with benchmark results
    """
    print(f"\nBenchmarking {model_name}...")

    try:
        # Build model
        model = build_small_model(
            model_name=model_name,
            num_classes=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE,
            pretrained=False  # Use random weights for fair comparison
        )
        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

        # Measure inference time (single sample)
        print(f"  Measuring inference time (single sample)...")
        inference_time_single = measure_inference_time(
            model, (1, 3, 224, 224), device, num_runs=num_inference_runs
        )

        # Measure inference time (batch)
        print(f"  Measuring inference time (batch of {batch_size})...")
        inference_time_batch = measure_inference_time(
            model, (batch_size, 3, 224, 224), device, num_runs=num_inference_runs
        )

        # Measure training time
        print(f"  Measuring training time (batch of {batch_size})...")
        training_time = measure_training_time(
            model, (batch_size, 3, 224, 224), device, num_runs=num_training_runs
        )

        # Measure memory
        print(f"  Measuring memory usage...")
        memory_mb = get_model_memory(model, device)

        # Cleanup
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        results = {
            'Model': model_name,
            'Parameters (M)': round(total_params / 1e6, 2),
            'Trainable Params (M)': round(trainable_params / 1e6, 2),
            'Inference Time - Single (ms)': round(inference_time_single, 2),
            'Inference Time - Batch (ms)': round(inference_time_batch, 2),
            'Inference Time - Per Sample (ms)': round(inference_time_batch / batch_size, 2),
            'Training Time - Batch (ms)': round(training_time, 2),
            'Training Time - Per Sample (ms)': round(training_time / batch_size, 2),
            'Memory (MB)': round(memory_mb, 2),
        }

        print(f"  ✓ Benchmark complete")
        print(f"    Inference (single): {results['Inference Time - Single (ms)']} ms")
        print(f"    Inference (batch): {results['Inference Time - Batch (ms)']} ms")
        print(f"    Training (batch): {results['Training Time - Batch (ms)']} ms")
        print(f"    Memory: {results['Memory (MB)']} MB")

        return results

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def main():
    """Main benchmark function."""
    print("=" * 80)
    print("SMALL MODELS BENCHMARK")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get device
    device = get_device()

    # Benchmark settings
    batch_size = 32
    num_inference_runs = 100
    num_training_runs = 50

    print(f"\nBenchmark Settings:")
    print(f"  Batch size: {batch_size}")
    print(f"  Inference runs: {num_inference_runs}")
    print(f"  Training runs: {num_training_runs}")
    print(f"  Input size: 224x224")
    print(f"  Number of classes: {config.NUM_CLASSES}")
    print("=" * 80)

    # Benchmark all models
    results = []

    for i, model_name in enumerate(config.AVAILABLE_SMALL_MODELS, 1):
        print(f"\n[{i}/{len(config.AVAILABLE_SMALL_MODELS)}] {model_name}")
        result = benchmark_model(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            num_inference_runs=num_inference_runs,
            num_training_runs=num_training_runs
        )

        if result:
            results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by inference time (single)
    df = df.sort_values('Inference Time - Single (ms)')

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'benchmark_results_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Print insights
    print("\n" + "=" * 80)
    print("INSIGHTS")
    print("=" * 80)

    if not df.empty:
        fastest_inference = df.iloc[0]
        print(f"\nFastest Inference (single): {fastest_inference['Model']}")
        print(f"  Time: {fastest_inference['Inference Time - Single (ms)']} ms")
        print(f"  Parameters: {fastest_inference['Parameters (M)']}M")

        fastest_training_idx = df['Training Time - Batch (ms)'].idxmin()
        fastest_training = df.loc[fastest_training_idx]
        print(f"\nFastest Training: {fastest_training['Model']}")
        print(f"  Time: {fastest_training['Training Time - Batch (ms)']} ms/batch")
        print(f"  Parameters: {fastest_training['Parameters (M)']}M")

        smallest_idx = df['Parameters (M)'].idxmin()
        smallest = df.loc[smallest_idx]
        print(f"\nSmallest Model: {smallest['Model']}")
        print(f"  Parameters: {smallest['Parameters (M)']}M")
        print(f"  Inference: {smallest['Inference Time - Single (ms)']} ms")

        most_efficient_idx = df['Memory (MB)'].idxmin()
        most_efficient = df.loc[most_efficient_idx]
        print(f"\nMost Memory Efficient: {most_efficient['Model']}")
        print(f"  Memory: {most_efficient['Memory (MB)']} MB")
        print(f"  Parameters: {most_efficient['Parameters (M)']}M")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
