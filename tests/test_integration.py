"""Integration tests for TT-decomposed networks."""

import torch
import torch.nn as nn
import torch.optim as optim

from torch_mpo.layers import TTConv2d


class TestIntegration:
    """Integration tests for complete networks with TT layers."""

    def test_simple_conv_network(self):
        """Test a simple convolutional network."""
        model = nn.Sequential(
            TTConv2d(3, 32, kernel_size=3, padding=1, tt_ranks=4),
            nn.ReLU(),
            TTConv2d(32, 64, kernel_size=3, padding=1, tt_ranks=4),
            nn.ReLU(),
            TTConv2d(64, 10, kernel_size=3, padding=1, tt_ranks=4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Test forward pass
        x = torch.randn(8, 3, 32, 32)
        output = model(x)

        assert output.shape == (8, 10), f"Expected shape (8, 10), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN/Inf"

    def test_training_simple_network(self):
        """Test that a simple network can be trained."""
        # Simple 3-layer network
        model = nn.Sequential(
            TTConv2d(3, 32, kernel_size=3, padding=1, tt_ranks=4),
            nn.ReLU(),
            TTConv2d(32, 64, kernel_size=3, padding=1, tt_ranks=4),
            nn.ReLU(),
            TTConv2d(64, 10, kernel_size=3, padding=1, tt_ranks=4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Training setup
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Dummy data
        x = torch.randn(8, 3, 32, 32)
        target = torch.randint(0, 10, (8,))

        # Record initial loss
        model.eval()
        with torch.no_grad():
            initial_output = model(x)
            initial_loss = criterion(initial_output, target).item()

        # Train for a few steps
        model.train()
        losses = []
        for _ in range(10):
            output = model(x)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Check that loss decreased
        final_loss = losses[-1]
        assert (
            final_loss < initial_loss
        ), f"Loss did not decrease: {initial_loss} -> {final_loss}"
        assert all(
            torch.isfinite(p).all() for p in model.parameters()
        ), "Parameters contain NaN/Inf"

    def test_mixed_tt_standard_network(self):
        """Test network with both TT and standard layers."""
        model = nn.Sequential(
            # Standard conv
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # TT conv
            TTConv2d(32, 64, kernel_size=3, padding=1, tt_ranks=8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Another TT conv
            TTConv2d(64, 128, kernel_size=3, padding=1, tt_ranks=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # Standard linear
            nn.Linear(128, 10),
        )

        # Test forward pass
        x = torch.randn(4, 3, 32, 32)
        output = model(x)

        assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"

        # Test backward pass
        target = torch.randint(0, 10, (4,))
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_different_input_sizes(self):
        """Test TT layers with different input sizes."""
        model = TTConv2d(3, 16, kernel_size=3, padding=1, tt_ranks=4)

        input_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]

        for h, w in input_sizes:
            x = torch.randn(1, 3, h, w)
            output = model(x)

            assert output.shape == (
                1,
                16,
                h,
                w,
            ), f"Input size ({h}, {w}): expected output shape (1, 16, {h}, {w}), got {output.shape}"
            assert torch.isfinite(
                output
            ).all(), f"Input size ({h}, {w}): output contains NaN/Inf"

    def test_compression_benefits(self):
        """Test that TT layers actually provide compression."""
        in_channels, out_channels = 256, 512

        # Standard conv
        standard_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        standard_params = sum(p.numel() for p in standard_conv.parameters())

        # TT conv with reasonable rank
        tt_conv = TTConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, tt_ranks=16
        )
        tt_params = sum(p.numel() for p in tt_conv.parameters())

        compression_ratio = standard_params / tt_params
        assert compression_ratio > 5.0, (
            f"Expected compression ratio > 5, got {compression_ratio:.2f} "
            f"(standard: {standard_params}, TT: {tt_params})"
        )

    def test_deterministic_forward(self):
        """Test that forward pass is deterministic."""
        torch.manual_seed(42)
        model = TTConv2d(3, 16, kernel_size=3, padding=1, tt_ranks=4)

        x = torch.randn(2, 3, 8, 8)

        # Multiple forward passes should give same result
        output1 = model(x)
        output2 = model(x)

        assert torch.allclose(output1, output2), "Forward pass is not deterministic"
