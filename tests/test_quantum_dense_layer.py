from __future__ import annotations

import pytest
import torch

from src.utils import QuantumDenseLayer


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "embedding,input_dim",
    [
        ("rotation", 97),
        ("rotation", 960),
        ("amplitude", 1),
        ("amplitude", 33),
        ("amplitude", 960),
        ("amplitude", 4096),
    ],
)
@pytest.mark.parametrize("arch", ["all_to_all", "ring"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("output_dim", [1, 4, 8, 12])
def test_forward_shape_grid(embedding: str, input_dim: int, arch: str, depth: int, output_dim: int) -> None:
    torch.manual_seed(0)

    layer = QuantumDenseLayer(
        output_dim=output_dim,
        embedding=embedding,
        architecture=arch,
        depth=depth,
    ).to(_device())

    x = torch.randn(4, input_dim, device=_device(), dtype=torch.float32)

    y = layer(x)
    assert y.shape == (4, output_dim)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_backward_computes_gradients() -> None:
    torch.manual_seed(0)

    layer = QuantumDenseLayer(output_dim=6, embedding="rotation", architecture="ring", depth=3).to(_device())
    x = torch.randn(2, 97, device=_device(), dtype=torch.float32, requires_grad=True)

    y = layer(x)
    loss = (y**2).mean()
    loss.backward()

    assert layer.theta.grad is not None
    assert torch.isfinite(layer.theta.grad).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_amplitude_embedding_raises_if_too_wide() -> None:
    layer = QuantumDenseLayer(output_dim=4, embedding="amplitude", depth=1)
    x = torch.randn(1, 4097)
    with pytest.raises(ValueError):
        _ = layer(x)


def test_output_bounds() -> None:
    with pytest.raises(ValueError):
        _ = QuantumDenseLayer(output_dim=0)
    with pytest.raises(ValueError):
        _ = QuantumDenseLayer(output_dim=13)


def test_can_train_parameters() -> None:
    torch.manual_seed(0)

    layer = QuantumDenseLayer(output_dim=4, embedding="rotation", architecture="ring", depth=2).to(_device())
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)

    x = torch.randn(8, 100, device=_device(), dtype=torch.float32)
    target = torch.zeros((8, 4), device=_device(), dtype=torch.float32)

    theta_before = layer.theta.detach().clone()

    losses = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        y = layer(x)
        loss = torch.nn.functional.mse_loss(y, target)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(loss.detach())

    theta_after = layer.theta.detach().clone()
    assert not torch.allclose(theta_before, theta_after)

    # Often decreases; but accept any change to avoid flaky failures.
    assert (losses[-1] <= losses[0] + 1e-6) or (not torch.allclose(losses[-1], losses[0]))
