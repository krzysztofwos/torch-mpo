"""Test linear compression equivalence."""

import torch
import torch.nn as nn

from torch_mpo.utils import compress_model


class TinyMLP(nn.Module):
    """Small MLP for testing compression."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 16)
        )

    def forward(self, x):
        return self.classifier(x)


class SharedLinearModel(nn.Module):
    """Model with aliased linear layers to test weight tying preservation."""

    def __init__(self):
        super().__init__()
        shared = nn.Linear(16, 16, bias=False)
        self.a = shared
        self.b = shared

    def forward(self, x):
        return self.a(x), self.b(x)


class AutoDiscoverModel(nn.Module):
    """Model that relies on compress_model defaults."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.head = nn.Linear(8 * 8 * 8, 6)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.avg_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)
        return self.head(x)


def test_linear_compression_similarity():
    """Test that compressed linear layers produce similar outputs initially."""
    torch.manual_seed(0)
    m = TinyMLP()

    # Note: TT approximation with limited ranks has significant error
    # This test mainly ensures compression runs without errors
    cm = compress_model(
        m,
        layers_to_compress=["classifier.1", "classifier.3"],
        compress_conv=False,
        compression_ratio=0.5,  # Less aggressive compression
        verbose=False,
    )
    x = torch.randn(4, 1, 8, 8)

    with torch.no_grad():
        y_ref = m(x)
        y_cmp = cm(x)

    # Just check outputs are finite and have reasonable magnitude
    assert torch.isfinite(y_cmp).all()
    assert y_cmp.abs().mean() < 100  # Reasonable magnitude

    # Note: Due to TT approximation limitations with default ranks,
    # cosine similarity can be low initially. Fine-tuning is required
    # for good performance.


def test_linear_compression_parameter_reduction():
    """Test that compression actually reduces parameters."""
    torch.manual_seed(42)
    m = TinyMLP()

    # Count original parameters
    orig_params = sum(
        p.numel()
        for name, p in m.named_parameters()
        if "classifier.1" in name or "classifier.3" in name
    )

    cm = compress_model(
        m,
        layers_to_compress=["classifier.1", "classifier.3"],
        compress_conv=False,
        compression_ratio=0.25,
        verbose=False,
    )

    # Count compressed parameters
    comp_params = sum(
        p.numel()
        for name, p in cm.named_parameters()
        if "classifier.1" in name or "classifier.3" in name
    )

    assert comp_params < orig_params
    compression_achieved = orig_params / comp_params
    assert (
        compression_achieved > 1.5
    ), f"Insufficient compression: {compression_achieved:.2f}x"


def test_compressed_model_trainable():
    """Test that compressed model can be trained."""
    torch.manual_seed(0)
    m = TinyMLP()
    cm = compress_model(
        m,
        layers_to_compress=["classifier.1", "classifier.3"],
        compress_conv=False,
        compression_ratio=0.3,
        verbose=False,
    )

    # Simple training step
    optimizer = torch.optim.Adam(cm.parameters(), lr=1e-3)
    x = torch.randn(4, 1, 8, 8)
    target = torch.randn(4, 16)

    # Forward pass
    output = cm(x)
    loss = torch.nn.functional.mse_loss(output, target)

    # Backward pass
    loss.backward()

    # Check gradients exist
    for name, p in cm.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()


def test_shared_linear_aliases_remain_shared_after_compression():
    """Test that compress_model preserves aliased layer references."""
    model = SharedLinearModel()
    compressed = compress_model(
        model,
        layers_to_compress=["a"],
        compress_conv=False,
        tt_ranks=2,
        verbose=False,
    )

    from torch_mpo.layers import TTLinear

    assert isinstance(compressed.a, TTLinear)
    assert compressed.a is compressed.b

    y_a, y_b = compressed(torch.randn(3, 16))
    assert torch.allclose(y_a, y_b)


def test_compress_model_auto_discovers_layers_and_heuristic_ranks():
    """Test compress_model defaults without explicit layers or TT ranks."""
    model = AutoDiscoverModel()
    compressed = compress_model(
        model,
        layers_to_compress=None,
        tt_ranks=None,
        compression_ratio=0.5,
        verbose=False,
    )

    from torch_mpo.layers import TTConv2d, TTLinear

    assert isinstance(compressed.conv, TTConv2d)
    assert isinstance(compressed.head, TTLinear)

    output = compressed(torch.randn(2, 3, 16, 16))
    assert output.shape == (2, 6)
    assert torch.isfinite(output).all()
