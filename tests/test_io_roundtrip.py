"""Test save/load roundtrip for TT layers."""

import copy
import io

import torch
import torch.nn as nn

from torch_mpo.layers import TTConv2d, TTLinear


def _roundtrip(m: nn.Module) -> nn.Module:
    """Save and load module state dict."""
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)

    # Create a new instance with same parameters
    m2 = copy.deepcopy(m)
    # Zero out parameters to ensure we're loading properly
    for p in m2.parameters():
        torch.nn.init.zeros_(p)

    # Load state dict
    m2.load_state_dict(torch.load(buf))
    return m2


def test_ttlinear_state_dict_roundtrip():
    """Test TTLinear state dict save/load."""
    m = TTLinear(64, 32, tt_ranks=4)
    x = torch.randn(3, 64)
    y = m(x)

    m2 = _roundtrip(m)
    y2 = m2(x)

    assert torch.allclose(y, y2, atol=1e-6, rtol=1e-5)


def test_ttconv2d_state_dict_roundtrip():
    """Test TTConv2d state dict save/load."""
    m = TTConv2d(3, 8, kernel_size=3, padding=1, tt_ranks=4)
    x = torch.randn(2, 3, 16, 16)
    y = m(x)

    m2 = _roundtrip(m)
    y2 = m2(x)

    assert torch.allclose(y, y2, atol=1e-6, rtol=1e-5)


def test_ttlinear_with_bias_roundtrip():
    """Test TTLinear with bias save/load."""
    m = TTLinear(32, 16, tt_ranks=3, bias=True)
    x = torch.randn(5, 32)
    y = m(x)

    m2 = _roundtrip(m)
    y2 = m2(x)

    assert torch.allclose(y, y2, atol=1e-6, rtol=1e-5)
    # Verify bias was loaded
    assert torch.allclose(m.bias, m2.bias)
