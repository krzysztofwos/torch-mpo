"""Test compress_model Conv2d handling."""

import torch
import torch.nn as nn

from torch_mpo.layers import TTConv2d
from torch_mpo.utils import compress_model


def test_conv_tuple_geometry_preserved():
    """Test that Conv2d tuple parameters are preserved during compression."""
    m = nn.Sequential(
        nn.Conv2d(
            3, 16, kernel_size=(3, 5), stride=(2, 3), padding=(1, 2), dilation=(1, 2)
        )
    )
    mc = compress_model(
        m, layers_to_compress=["0"], tt_ranks=4, verbose=False, compress_linear=False
    )
    tt = mc[0]
    assert isinstance(tt, TTConv2d)
    assert tt.kernel_size == (3, 5)
    assert tt.stride == (2, 3)
    assert tt.padding == (1, 2)
    assert tt.dilation == (1, 2)


def test_conv_same_padding_is_handled_or_skipped(capsys):
    """Test handling of 'same' padding in Conv2d compression."""
    m = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding="same"))
    mc = compress_model(
        m, layers_to_compress=["0"], tt_ranks=4, verbose=True, compress_linear=False
    )

    # Check if 'same' padding was translated to numeric
    out = capsys.readouterr().out
    if isinstance(mc[0], TTConv2d):
        # If successfully converted, padding should be (1, 1) for kernel_size=3, stride=1
        assert mc[0].padding == (1, 1)
    else:
        # If not supported, should remain Conv2d and print skip message
        assert isinstance(mc[0], nn.Conv2d)
        assert "padding" in out.lower() or "skip" in out.lower()


def test_conv_same_padding_with_stride_gt_1_is_skipped(capsys):
    """Test that 'same' padding with stride>1 is properly skipped."""
    # Create model with 'same' padding and stride=2
    m = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding="same"))
    mc = compress_model(
        m, layers_to_compress=["0"], tt_ranks=4, verbose=True, compress_linear=False
    )

    # Should remain Conv2d (not compressed) due to stride>1 with 'same' padding
    assert isinstance(mc[0], nn.Conv2d)
    assert not isinstance(mc[0], TTConv2d)

    # Check that appropriate skip message was printed
    out = capsys.readouterr().out
    assert "padding='same' with stride=" in out or "asymmetric" in out.lower()

    # Also test with tuple strides
    m2 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=(2, 3), padding="same"))
    mc2 = compress_model(
        m2, layers_to_compress=["0"], tt_ranks=4, verbose=True, compress_linear=False
    )

    assert isinstance(mc2[0], nn.Conv2d)
    assert not isinstance(mc2[0], TTConv2d)
    out2 = capsys.readouterr().out
    assert "padding='same' with stride=" in out2 or "asymmetric" in out2.lower()


def test_grouped_conv_is_skipped(capsys):
    """Test that grouped convolutions are properly skipped."""
    m = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, groups=4))
    mc = compress_model(
        m, layers_to_compress=["0"], tt_ranks=4, verbose=True, compress_linear=False
    )

    # Grouped conv should be skipped
    assert isinstance(mc[0], nn.Conv2d)
    out = capsys.readouterr().out
    assert "grouped" in out.lower()


def test_conv_rectangular_kernels():
    """Test compression of Conv2d with rectangular kernels."""
    m = nn.Sequential(nn.Conv2d(8, 16, kernel_size=(3, 5), padding=(1, 2)))
    mc = compress_model(
        m, layers_to_compress=["0"], tt_ranks=4, verbose=False, compress_linear=False
    )

    assert isinstance(mc[0], TTConv2d)
    assert mc[0].kernel_size == (3, 5)
    assert mc[0].padding == (1, 2)

    # Test forward pass shape preservation
    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        y_orig = m(x)
        y_comp = mc(x)
    assert y_orig.shape == y_comp.shape
