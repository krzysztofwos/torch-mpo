"""Gradient checking tests for TT layers."""

import pytest
import torch

from torch_mpo.layers import TTConv2d, TTLinear


def test_ttlinear_gradcheck():
    """Test gradient computation for TTLinear using gradcheck."""
    torch.manual_seed(0)
    m = TTLinear(
        12,
        9,
        inp_modes=[3, 4],
        out_modes=[3, 3],
        tt_ranks=[1, 3, 1],
        bias=False,
        dtype=torch.double,
    )
    m.double()
    x = torch.randn(5, 12, dtype=torch.double, requires_grad=True)

    def func(inp):
        return m(inp)

    assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_ttlinear_with_bias_gradcheck():
    """Test gradient computation for TTLinear with bias."""
    torch.manual_seed(0)
    m = TTLinear(
        8,
        6,
        inp_modes=[2, 2, 2],
        out_modes=[2, 3, 1],  # Fixed: must have same number of modes
        tt_ranks=[1, 2, 2, 1],  # Fixed: must have d+1 ranks
        bias=True,
        dtype=torch.double,
    )
    m.double()
    x = torch.randn(3, 8, dtype=torch.double, requires_grad=True)

    def func(inp):
        return m(inp)

    assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only gradcheck for conv may be slow; skip on CPU",
)
def test_ttconv2d_gradcheck_cuda():
    """Test gradient computation for TTConv2d on CUDA."""
    torch.manual_seed(0)
    m = TTConv2d(2, 3, kernel_size=3, padding=1, tt_ranks=2, bias=False).cuda().double()
    x = torch.randn(2, 2, 5, 5, device="cuda", dtype=torch.double, requires_grad=True)

    def func(inp):
        return m(inp)

    assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_ttconv2d_gradcheck_cpu():
    """Test gradient computation for small TTConv2d on CPU."""
    torch.manual_seed(0)
    # Use very small dimensions for CPU gradcheck
    m = TTConv2d(
        2, 2, kernel_size=2, padding=0, tt_ranks=2, bias=False, dtype=torch.double
    )
    x = torch.randn(1, 2, 3, 3, dtype=torch.double, requires_grad=True)

    def func(inp):
        return m(inp)

    assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)
