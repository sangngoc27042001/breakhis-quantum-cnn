"""Benchmark quantum layers performance (intended for V100 investigation).

Run on the target machine, for example:

  python -m src.bench_quantum_speed --layer dense --embedding amplitude --batch-size 32 --input-dim 960
  python -m src.bench_quantum_speed --layer dense --embedding rotation  --batch-size 32 --input-dim 960

It prints:
- detected torch device
- QuantumDenseLayer / QuantumPoolingLayer backend (lightning.gpu vs lightning.qubit)
- forward / backward time per step

This script uses synthetic inputs so it isolates layer performance.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch


@dataclass
class Timings:
    forward_s: float
    backward_s: float


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_one_step(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    do_backward: bool,
    optimizer: torch.optim.Optimizer | None,
) -> Timings:
    model.train(do_backward)

    _sync_if_cuda(device)
    t0 = time.perf_counter()
    y = model(x)
    _sync_if_cuda(device)
    t1 = time.perf_counter()

    backward_s = 0.0
    if do_backward:
        assert optimizer is not None
        loss = y.mean()
        optimizer.zero_grad(set_to_none=True)
        _sync_if_cuda(device)
        t2 = time.perf_counter()
        loss.backward()
        optimizer.step()
        _sync_if_cuda(device)
        t3 = time.perf_counter()
        backward_s = t3 - t2

    return Timings(forward_s=t1 - t0, backward_s=backward_s)


def bench_dense(args: argparse.Namespace) -> None:
    from src.utils import QuantumDenseLayer

    device = torch.device(args.device)
    print(f"[torch] device={device} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[torch] gpu={torch.cuda.get_device_name(0)}")

    print(
        f"[QuantumDenseLayer] config: embedding={args.embedding} depth={args.depth} input_dim={args.input_dim} "
        f"output_dim={args.output_dim} batch_size={args.batch_size}"
    )

    layer = QuantumDenseLayer(output_dim=args.output_dim, embedding=args.embedding, depth=args.depth).to(device)
    x = torch.randn(args.batch_size, args.input_dim, device=device, dtype=torch.float32)

    # Warmup / build
    for _ in range(args.warmup):
        _ = layer(x)
        if args.backward:
            opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
            _ = _time_one_step(model=layer, x=x, device=device, do_backward=True, optimizer=opt)

    # Report quantum backend after build
    qdtype = getattr(layer, "_quantum_device_type", "unknown")
    n_qubits = getattr(layer, "n_qubits_input", "unknown")
    print(f"[QuantumDenseLayer] backend={qdtype} n_qubits_input={n_qubits}")

    # Benchmark quantum layer
    opt = torch.optim.Adam(layer.parameters(), lr=1e-3) if args.backward else None
    timings = []
    for _ in range(args.iters):
        timings.append(
            _time_one_step(model=layer, x=x, device=device, do_backward=args.backward, optimizer=opt)
        )

    f_ms = sum(t.forward_s for t in timings) / len(timings) * 1000
    if args.backward:
        b_ms = sum(t.backward_s for t in timings) / len(timings) * 1000
    else:
        b_ms = 0.0

    print(f"\n[QuantumDenseLayer] avg forward:  {f_ms:.2f} ms/step")
    if args.backward:
        print(f"[QuantumDenseLayer] avg backward: {b_ms:.2f} ms/step")

    # Baseline: nn.Linear
    baseline = torch.nn.Linear(args.input_dim, args.output_dim).to(device)
    opt2 = torch.optim.Adam(baseline.parameters(), lr=1e-3) if args.backward else None

    timings2 = []
    for _ in range(args.iters):
        timings2.append(
            _time_one_step(model=baseline, x=x, device=device, do_backward=args.backward, optimizer=opt2)
        )

    f2_ms = sum(t.forward_s for t in timings2) / len(timings2) * 1000
    if args.backward:
        b2_ms = sum(t.backward_s for t in timings2) / len(timings2) * 1000
    else:
        b2_ms = 0.0

    print(f"\n[nn.Linear baseline] avg forward:  {f2_ms:.2f} ms/step")
    if args.backward:
        print(f"[nn.Linear baseline] avg backward: {b2_ms:.2f} ms/step")


def bench_pool(args: argparse.Namespace) -> None:
    from src.utils.quantum_pooling_layer.layer import QuantumPoolingLayer

    device = torch.device(args.device)
    print(f"[torch] device={device} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[torch] gpu={torch.cuda.get_device_name(0)}")

    # QuantumPoolingLayer chooses backend at import time / init.
    print(
        f"[QuantumPoolingLayer] config: depth={args.depth} n_qubits=4 "
        f"(fixed in layer) batch_size={args.batch_size} spatial={args.spatial} channels={args.channels}"
    )

    layer = QuantumPoolingLayer(depth=args.depth, n_qubits=4).to(device)

    # QuantumPoolingLayer expects channel-last (batch, n, n, m)
    x = torch.randn(args.batch_size, args.spatial, args.spatial, args.channels, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(args.warmup):
        _ = layer(x)

    opt = torch.optim.Adam(layer.parameters(), lr=1e-3) if args.backward else None
    timings = []
    for _ in range(args.iters):
        timings.append(_time_one_step(model=layer, x=x, device=device, do_backward=args.backward, optimizer=opt))

    f_ms = sum(t.forward_s for t in timings) / len(timings) * 1000
    b_ms = sum(t.backward_s for t in timings) / len(timings) * 1000 if args.backward else 0.0

    print(f"\n[QuantumPoolingLayer] avg forward:  {f_ms:.2f} ms/step")
    if args.backward:
        print(f"[QuantumPoolingLayer] avg backward: {b_ms:.2f} ms/step")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", choices=["dense", "pool"], default="dense")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--backward", action="store_true", help="also time backward+optimizer step")

    # dense
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--input-dim", type=int, default=960)
    p.add_argument("--output-dim", type=int, default=8)
    p.add_argument("--embedding", choices=["amplitude", "rotation"], default="amplitude")
    p.add_argument("--depth", type=int, default=1)

    # pool
    p.add_argument("--spatial", type=int, default=14)
    p.add_argument("--channels", type=int, default=128)

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.layer == "dense":
        bench_dense(args)
    else:
        bench_pool(args)


if __name__ == "__main__":
    main()
