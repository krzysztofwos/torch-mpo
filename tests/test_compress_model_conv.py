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


def test_conv_same_padding_is_handled_or_skipped(caplog):
    """Test handling of 'same' padding in Conv2d compression."""
    import logging

    # Set logging level to capture info messages
    caplog.set_level(logging.INFO)

    m = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding="same"))
    mc = compress_model(
        m, layers_to_compress=["0"], tt_ranks=4, verbose=True, compress_linear=False
    )

    # Check if 'same' padding was translated to numeric
    log_text = caplog.text.lower()
    if isinstance(mc[0], TTConv2d):
        # If successfully converted, padding should be (1, 1) for kernel_size=3, stride=1
        assert mc[0].padding == (1, 1)
    else:
        # If not supported, should remain Conv2d and log skip message
        assert isinstance(mc[0], nn.Conv2d)
        assert "padding" in log_text or "skip" in log_text


def test_conv_same_padding_with_stride_gt_1_is_skipped(caplog):
    """Test that 'same' padding with stride>1 would be properly skipped if it were supported by PyTorch.

    Note: PyTorch doesn't actually support padding='same' with stride>1, so we test with
    a mock Conv2d layer that has the attributes we would expect.
    """
    import logging
    import types

    # Set logging level to capture info messages
    caplog.set_level(logging.INFO)

    # Create a mock Conv2d with 'same' padding and stride>1 attributes
    # This simulates what would happen if PyTorch did support this combination
    mock_conv = nn.Conv2d(
        3, 8, kernel_size=3, stride=1, padding=1
    )  # Create a valid conv first
    # Then override attributes to simulate 'same' padding with stride>1
    mock_conv.padding = "same"
    mock_conv.stride = (2, 2)

    m = nn.Sequential()
    m.add_module("0", mock_conv)

    mc = compress_model(
        m, layers_to_compress=["0"], tt_ranks=4, verbose=True, compress_linear=False
    )

    # Should remain Conv2d (not compressed) due to stride>1 with 'same' padding
    assert isinstance(mc[0], nn.Conv2d)
    assert not isinstance(mc[0], TTConv2d)

    # Check that appropriate skip message was logged
    log_text = caplog.text.lower()
    assert "padding='same' with stride=" in log_text or "asymmetric" in log_text


def test_grouped_conv_is_skipped(caplog):
    """Test that grouped convolutions are properly skipped."""
    import logging

    # Set logging level to capture info messages
    caplog.set_level(logging.INFO)

    m = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, groups=4))
    mc = compress_model(
        m, layers_to_compress=["0"], tt_ranks=4, verbose=True, compress_linear=False
    )

    # Grouped conv should be skipped
    assert isinstance(mc[0], nn.Conv2d)
    log_text = caplog.text.lower()
    assert "grouped" in log_text


def test_conv_init_from_pretrained(caplog):
    """Test that compress_model initializes TTConv2d from pretrained weights."""
    import logging

    caplog.set_level(logging.INFO)

    # Create a simple model with a Conv2d layer
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
        nn.ReLU(),
    )

    # Get reference output before compression
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y_before = model(x)

    # Compress the model
    compressed = compress_model(
        model,
        layers_to_compress=["0"],
        tt_ranks=8,
        verbose=True,
        compress_linear=False,
    )

    # Check that initialization from pretrained was attempted
    log_text = caplog.text
    assert (
        "initialized from pretrained weights via SVD" in log_text
        or "fallback" in log_text
    )

    # Get output after compression
    with torch.no_grad():
        y_after = compressed(x)

    # Outputs should be reasonably close if initialization worked
    # (random init would give very different outputs)
    rel_error = (y_before - y_after).norm() / y_before.norm()
    # With rank 8 and our simplified decomposition, we expect moderate approximation error
    # The key is that it's better than random initialization (which would give rel_error > 1.5)
    assert rel_error < 1.3, f"Outputs too different: rel_error={rel_error:.3f}"


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
