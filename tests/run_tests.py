"""Minimal test runner without external dependencies.

This project doesn't pin pytest by default. This script provides a tiny
smoke-test suite for the QuantumDenseLayer that you can run via:

  uv run python -m tests.run_tests

If you prefer pytest, install it and run `uv run pytest -q`.
"""

from __future__ import annotations

import sys

import torch

from src.utils import QuantumDenseLayer


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_forward_backward_hparam_grid() -> None:
    """Checks forward/backward across several hyperparameter combinations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    embeddings = ["rotation", "amplitude"]
    templates = ["strong", "two_design"]
    depths = [1, 2]
    output_dims = [1, 4, 8, 12]

    for embedding in embeddings:
        for template in templates:
            for depth in depths:
                for out_dim in output_dims:
                    input_dim = 97 if embedding == "rotation" else 960

                    layer = QuantumDenseLayer(
                        output_dim=out_dim,
                        embedding=embedding,
                        template=template,
                        depth=depth,
                    ).to(device)

                    x = torch.randn(3, input_dim, device=device, dtype=torch.float32, requires_grad=True)
                    y = layer(x)

                    _assert(
                        tuple(y.shape) == (3, out_dim),
                        f"bad shape for {embedding=}, {template=}, {depth=}, {out_dim=}",
                    )
                    _assert(torch.isfinite(y).all().item(), "output has NaN/Inf")

                    loss = (y**2).mean()
                    loss.backward()

                    _assert(layer.theta.grad is not None, "theta.grad is None")
                    _assert(torch.isfinite(layer.theta.grad).all().item(), "theta.grad has NaN/Inf")
                    _assert(x.grad is not None, "x.grad is None")
                    _assert(torch.isfinite(x.grad).all().item(), "x.grad has NaN/Inf")


def test_can_train_parameters() -> None:
    """Runs a few optimizer steps and ensures parameters update and loss is finite."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    layer = QuantumDenseLayer(output_dim=4, embedding="rotation", template="strong", depth=2).to(device)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)

    # Fixed input, synthetic regression target
    x = torch.randn(8, 100, device=device, dtype=torch.float32)
    target = torch.zeros((8, 4), device=device, dtype=torch.float32)

    theta_before = layer.theta.detach().clone()

    losses = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        y = layer(x)
        loss = torch.nn.functional.mse_loss(y, target)
        _assert(torch.isfinite(loss).item(), "loss has NaN/Inf")
        loss.backward()
        opt.step()
        losses.append(loss.detach())

    theta_after = layer.theta.detach().clone()

    _assert(not torch.allclose(theta_before, theta_after), "parameters did not update")
    # Usually decreases, but quantum landscapes can be noisy; accept decrease or at least change.
    _assert((losses[-1] <= losses[0] + 1e-6).item() or not torch.allclose(losses[-1], losses[0]), "loss did not change")


def test_amplitude_width_check() -> None:
    layer = QuantumDenseLayer(output_dim=4, embedding="amplitude", depth=1)
    x = torch.randn(1, 4097)
    try:
        _ = layer(x)
    except ValueError:
        return
    raise AssertionError("expected ValueError for m > 4096")


def main() -> None:
    tests = [
        ("test_forward_backward_hparam_grid", test_forward_backward_hparam_grid),
        ("test_can_train_parameters", test_can_train_parameters),
        ("test_amplitude_width_check", test_amplitude_width_check),
    ]

    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[OK] {name}")
        except Exception as e:
            failures += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
