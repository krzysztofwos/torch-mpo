"""Test conv compression smoke tests."""

import torch
import torch.nn as nn

from torch_mpo.layers import TTConv2d
from torch_mpo.utils import compress_model


class TinyCNN(nn.Module):
    """Small CNN for testing compression."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 16, kernel_size=(3, 5), stride=(2, 1), padding=(1, 2))
        # Expected shape after conv on 16×16 input: [batch, 16, 8, 16]
        self.head = nn.Linear(16 * 8 * 16, 10)

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        return self.head(x.reshape(b, c * h * w))


def test_conv_compression_shape_and_paramcount():
    """Test conv compression preserves shapes and reduces parameters."""
    m = TinyCNN()
    cm = compress_model(
        m, layers_to_compress=["conv"], compression_ratio=0.2, verbose=False
    )
    assert isinstance(cm.conv, TTConv2d)

    x = torch.randn(2, 8, 16, 16)
    y = cm(x)

    # Check output shape
    assert y.shape == (2, 10)
    assert torch.isfinite(y).all()

    # Check we actually saved parameters on conv
    old = sum(p.numel() for p in m.conv.parameters())
    new = sum(p.numel() for p in cm.conv.parameters())
    assert new < old, f"Conv params not reduced: {old} -> {new}"


def test_conv_compression_forward_backward():
    """Test that compressed conv model can do forward and backward passes."""
    torch.manual_seed(0)
    m = TinyCNN()
    cm = compress_model(
        m, layers_to_compress=["conv"], compression_ratio=0.3, verbose=False
    )

    # Forward pass
    x = torch.randn(2, 8, 16, 16)
    target = torch.randn(2, 10)
    output = cm(x)

    # Loss and backward
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    # Check gradients exist for compressed conv
    assert cm.conv.cores[0].grad is not None
    if cm.conv.bias is not None:
        assert cm.conv.bias.grad is not None


def test_mixed_compression():
    """Test compressing both conv and linear layers."""

    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.fc = nn.Linear(16 * 8 * 8, 10)

        def forward(self, x):
            x = self.conv(x)
            x = torch.nn.functional.avg_pool2d(x, 2)
            x = x.reshape(x.size(0), -1)
            return self.fc(x)

    m = MixedModel()
    cm = compress_model(
        m,
        layers_to_compress=["conv", "fc"],
        compression_ratio=0.3,
        compress_linear=True,
        compress_conv=True,
        verbose=False,
    )

    # Check both layers were compressed
    assert isinstance(cm.conv, TTConv2d)
    from torch_mpo.layers import TTLinear

    assert isinstance(cm.fc, TTLinear)

    # Test forward pass
    x = torch.randn(2, 3, 16, 16)
    y = cm(x)
    assert y.shape == (2, 10)
    assert torch.isfinite(y).all()
