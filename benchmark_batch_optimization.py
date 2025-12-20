"""
Benchmark script to test batch processing optimization on GPU (V100).
Tests different batch sizes and chunk sizes to find optimal configuration.
"""
import torch
import time
import sys
from src.utils.quantum_dense_layer import QuantumDenseLayer

def benchmark_batch_size(
    encoding_method: str,
    batch_size: int,
    chunk_size: int | None,
    num_features: int = 1280,
    num_iterations: int = 3
):
    """Benchmark a specific configuration."""

    # Move to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create quantum layer
    qdense = QuantumDenseLayer(
        output_dim=8,
        embedding=encoding_method,
        template="strong",
        depth=1,
        batch_chunk_size=chunk_size
    )
    qdense = qdense.to(device)

    # Create dummy input
    x = torch.randn(batch_size, num_features, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = qdense(x)

    # Synchronize GPU before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            output = qdense(x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    throughput = batch_size / avg_time

    return avg_time, throughput


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("BATCH PROCESSING OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("=" * 80)

    encoding_method = "amplitude"

    # Test configurations
    batch_sizes = [8, 16, 32, 64, 128]
    chunk_sizes = [None, 0, 4, 8, 16, 32]  # None=auto, 0=no chunking

    print(f"\nTesting {encoding_method.upper()} encoding")
    print("=" * 80)

    results = []

    for batch_size in batch_sizes:
        print(f"\nBatch Size: {batch_size}")
        print("-" * 80)

        for chunk_size in chunk_sizes:
            # Skip invalid combinations
            if chunk_size is not None and chunk_size > 0 and chunk_size >= batch_size:
                continue

            try:
                chunk_label = "auto (8)" if chunk_size is None else ("no chunking" if chunk_size == 0 else str(chunk_size))

                avg_time, throughput = benchmark_batch_size(
                    encoding_method=encoding_method,
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    num_iterations=3
                )

                print(f"  Chunk size {chunk_label:15s}: {avg_time:6.3f}s  ({throughput:6.1f} samples/sec)")

                results.append({
                    'batch_size': batch_size,
                    'chunk_size': chunk_size,
                    'chunk_label': chunk_label,
                    'time': avg_time,
                    'throughput': throughput
                })

            except Exception as e:
                print(f"  Chunk size {chunk_label:15s}: ERROR - {e}")

    # Find best configurations
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)

    for batch_size in batch_sizes:
        batch_results = [r for r in results if r['batch_size'] == batch_size]
        if batch_results:
            best = min(batch_results, key=lambda r: r['time'])
            print(f"Batch {batch_size:3d}: chunk_size={best['chunk_label']:15s} → {best['time']:.3f}s ({best['throughput']:6.1f} samples/sec)")

    # Overall recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Find the configuration with best throughput for batch=128
    batch_128_results = [r for r in results if r['batch_size'] == 128]
    if batch_128_results:
        best_128 = min(batch_128_results, key=lambda r: r['time'])
        print(f"\nFor large batches (128), use chunk_size={best_128['chunk_size']}")
        print(f"  Time: {best_128['time']:.3f}s")
        print(f"  Throughput: {best_128['throughput']:.1f} samples/sec")

        # Compare to no chunking
        no_chunk = next((r for r in batch_128_results if r['chunk_size'] == 0), None)
        if no_chunk:
            speedup = no_chunk['time'] / best_128['time']
            print(f"  Speedup vs no chunking: {speedup:.2f}x")

    # Configuration for config.py
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIG.PY SETTING")
    print("=" * 80)

    # Analyze which chunk size works best across different batch sizes
    chunk_performance = {}
    for chunk_size in chunk_sizes:
        chunk_results = [r for r in results if r['chunk_size'] == chunk_size]
        if chunk_results:
            avg_throughput = sum(r['throughput'] for r in chunk_results) / len(chunk_results)
            chunk_performance[chunk_size] = avg_throughput

    if chunk_performance:
        best_chunk = max(chunk_performance.items(), key=lambda x: x[1])
        chunk_label = "None  # auto-detect (8 for amplitude)" if best_chunk[0] is None else f"{best_chunk[0]}     # fixed chunk size"
        print(f"\nQUANTUM_CNN_CONFIG_DENSE_BATCH_CHUNK_SIZE = {chunk_label}")
        print(f"# Average throughput: {best_chunk[1]:.1f} samples/sec across all batch sizes")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. This benchmark is designed for GPU (V100).")
        print("   Running on CPU will not show the optimization benefits.\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    main()
