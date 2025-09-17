"""Tests for gradient flow through TT-decomposed layers."""

import torch
import torch.nn as nn

from torch_mpo.layers import TTConv2d


class TestGradientFlow:
    """Test gradient flow through TT layers."""

    def test_ttconv_gradient_flow(self):
        """Test gradient flow through TTConv2d."""
        # Simple TTConv2d
        conv = TTConv2d(3, 16, kernel_size=3, padding=1, tt_ranks=4)

        # Input with gradient tracking
        x = torch.randn(1, 3, 8, 8, requires_grad=True)

        # Forward and backward
        y = conv(x)
        loss = y.sum()
        loss.backward()

        # Check input gradient
        assert x.grad is not None, "Input gradient is None"
        assert x.grad.norm() > 0, "Input gradient is zero"

        # Check spatial conv gradient
        assert (
            conv.spatial_conv.weight.grad is not None
        ), "Spatial conv gradient is None"
        assert conv.spatial_conv.weight.grad.norm() > 0, "Spatial conv gradient is zero"

        # Check core gradients
        for i, core in enumerate(conv.cores):
            assert core.grad is not None, f"Core {i} gradient is None"
            assert core.grad.norm() > 0, f"Core {i} gradient is zero"

    def test_ttconv_with_loss(self):
        """Test TTConv2d gradients with actual loss function."""
        # Conv that reduces to class scores
        conv = TTConv2d(3, 10, kernel_size=8, stride=8, tt_ranks=4)
        x = torch.randn(4, 3, 8, 8)
        y = conv(x).squeeze(-1).squeeze(-1)  # Should be [4, 10]

        # Compute loss
        target = torch.randint(0, 10, (4,))
        loss = nn.CrossEntropyLoss()(y, target)
        loss.backward()

        # Verify all cores have gradients
        all_have_grads = all(
            core.grad is not None and core.grad.norm() > 0 for core in conv.cores
        )
        assert all_have_grads, "Not all cores have non-zero gradients"

    def test_deep_network_gradients(self):
        """Test gradient flow through deep network with TT layers."""
        # Simple network with 3 TT layers
        model = nn.Sequential(
            TTConv2d(3, 32, kernel_size=3, padding=1, tt_ranks=4),
            nn.ReLU(),
            TTConv2d(32, 64, kernel_size=3, padding=1, tt_ranks=4),
            nn.ReLU(),
            TTConv2d(64, 10, kernel_size=3, padding=1, tt_ranks=4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Forward and backward
        x = torch.randn(8, 3, 32, 32)
        target = torch.randint(0, 10, (8,))

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Check all parameters have gradients
        params_with_grad = 0
        params_without_grad = 0

        for _name, param in model.named_parameters():
            if param.grad is not None and param.grad.norm() > 1e-8:
                params_with_grad += 1
            else:
                params_without_grad += 1

        assert (
            params_without_grad == 0
        ), f"{params_without_grad} parameters have no/zero gradients"
        assert params_with_grad > 0, "No parameters have gradients"

    def test_gradient_magnitude(self):
        """Test that gradient magnitudes are reasonable."""
        # Create a simple model
        model = nn.Sequential(
            TTConv2d(3, 16, kernel_size=3, padding=1, tt_ranks=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

        # Forward and backward
        x = torch.randn(4, 3, 8, 8)
        target = torch.randint(0, 10, (4,))

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Check gradient magnitudes
        tt_conv = model[0]

        # Spatial conv gradient should be reasonable
        spatial_grad_norm = tt_conv.spatial_conv.weight.grad.norm().item()
        assert (
            1e-6 < spatial_grad_norm < 1e3
        ), f"Spatial conv gradient norm {spatial_grad_norm} is out of range"

        # Core gradients should be reasonable
        for i, core in enumerate(tt_conv.cores):
            core_grad_norm = core.grad.norm().item()
            assert (
                1e-6 < core_grad_norm < 1e3
            ), f"Core {i} gradient norm {core_grad_norm} is out of range"

    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple batches."""
        conv = TTConv2d(3, 16, kernel_size=3, padding=1, tt_ranks=4)

        # First batch
        x1 = torch.randn(2, 3, 8, 8)
        y1 = conv(x1)
        loss1 = y1.sum()
        loss1.backward()

        # Store first gradients
        grads_1 = [core.grad.clone() for core in conv.cores]

        # Second batch (accumulate gradients) - use same input to ensure consistent gradients
        # This tests that gradients accumulate rather than replace
        y2 = conv(x1)  # Same input
        loss2 = y2.sum()
        loss2.backward()

        # Check gradients accumulated (should be approximately 2x)
        grads_2 = [core.grad for core in conv.cores]

        # All gradients should have approximately doubled
        for i, (g1, g2) in enumerate(zip(grads_1, grads_2, strict=False)):
            ratio = g2.norm() / g1.norm()
            assert 1.95 < ratio < 2.05, (
                f"Core {i} gradient should approximately double "
                f"(expected ~2.0, got {ratio:.4f})"
            )
