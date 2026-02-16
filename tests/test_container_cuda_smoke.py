import pytest
import torch


@pytest.mark.cuda
def test_cuda_smoke_tensor_ops() -> None:
    assert torch.cuda.is_available()

    x = torch.randn(32, device="cuda")
    y = (x * x).sum()

    assert y.is_cuda
    assert torch.isfinite(y)
