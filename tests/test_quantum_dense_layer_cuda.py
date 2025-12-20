from __future__ import annotations

import pytest
import torch

from src.utils import QuantumDenseLayer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("template", ["strong", "two_design"])
@pytest.mark.parametrize("embedding,input_dim", [("rotation", 200), ("amplitude", 960)])
def test_forward_backward_on_cuda(template: str, embedding: str, input_dim: int) -> None:
    device = torch.device("cuda")

    layer = QuantumDenseLayer(output_dim=8, embedding=embedding, template=template, depth=2).to(device)

    x = torch.randn(4, input_dim, device=device, dtype=torch.float32, requires_grad=True)
    y = layer(x)

    assert y.is_cuda
    assert y.dtype == torch.float32

    loss = (y**2).mean()
    loss.backward()

    assert layer.theta.grad is not None
    assert layer.theta.grad.is_cuda
    assert torch.isfinite(layer.theta.grad).all()

    assert x.grad is not None
    assert x.grad.is_cuda
    assert torch.isfinite(x.grad).all()
