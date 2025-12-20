"""
Quick test to verify batch chunking optimization on V100.
"""
import torch
import time
from src.utils.quantum_dense_layer import QuantumDenseLayer

def test_batch(batch_size, chunk_size, device):
    """Test a specific configuration."""
    qdense = QuantumDenseLayer(
        output_dim=8,
        embedding="amplitude",
        template="strong",
        depth=1,
        batch_chunk_size=chunk_size
    ).to(device)

    x = torch.randn(batch_size, 1280, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = qdense(x)

    # Time it
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        _ = qdense(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    return time.time() - start


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    print("Testing batch=8:")
    time_8_no_chunk = test_batch(8, 0, device)
    print(f"  No chunking: {time_8_no_chunk:.3f}s")

    time_8_chunk = test_batch(8, 8, device)
    print(f"  Chunk=8:     {time_8_chunk:.3f}s")

    print("\nTesting batch=128:")
    time_128_no_chunk = test_batch(128, 0, device)
    print(f"  No chunking: {time_128_no_chunk:.3f}s")

    time_128_chunk = test_batch(128, 8, device)
    print(f"  Chunk=8:     {time_128_chunk:.3f}s")

    speedup = time_128_no_chunk / time_128_chunk
    print(f"\n✅ Speedup with chunking: {speedup:.2f}x faster!")
