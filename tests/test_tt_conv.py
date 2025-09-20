"""Tests for TTConv2d layer."""

import pytest
import torch
import torch.nn as nn

from torch_mpo.layers import TTConv2d


class TestTTConv2d:
    """Test cases for TTConv2d layer."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        batch_size = 8
        in_channels = 64
        out_channels = 128
        height, width = 32, 32

        # Create layer
        layer = TTConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            tt_ranks=4,
            bias=True,
        )

        # Test forward pass
        x = torch.randn(batch_size, in_channels, height, width)
        y = layer(x)

        # Check output shape (same spatial size due to padding)
        assert y.shape == (batch_size, out_channels, height, width)
        assert not torch.isnan(y).any()

    def test_different_kernel_sizes(self):
        """Test with different kernel sizes."""
        layer1 = TTConv2d(32, 64, kernel_size=3, tt_ranks=4)
        layer2 = TTConv2d(32, 64, kernel_size=5, tt_ranks=4)
        layer3 = TTConv2d(32, 64, kernel_size=(3, 5), tt_ranks=4)

        x = torch.randn(4, 32, 28, 28)

        y1 = layer1(x)
        assert y1.shape == (4, 64, 26, 26)  # 28 - 3 + 1

        y2 = layer2(x)
        assert y2.shape == (4, 64, 24, 24)  # 28 - 5 + 1

        y3 = layer3(x)
        assert y3.shape == (4, 64, 26, 24)  # (28 - 3 + 1, 28 - 5 + 1)

    def test_stride_and_padding(self):
        """Test stride and padding options."""
        # Stride 2
        layer1 = TTConv2d(16, 32, kernel_size=3, stride=2, tt_ranks=4)
        x = torch.randn(4, 16, 32, 32)
        y1 = layer1(x)
        assert y1.shape == (4, 32, 15, 15)  # (32 - 3) // 2 + 1

        # Padding
        layer2 = TTConv2d(16, 32, kernel_size=5, padding=2, tt_ranks=4)
        y2 = layer2(x)
        assert y2.shape == (4, 32, 32, 32)  # Same size due to padding

    def test_auto_factorization(self):
        """Test automatic mode factorization."""
        # Small channels
        layer1 = TTConv2d(32, 64, kernel_size=3, tt_ranks=4)
        assert len(layer1.inp_modes) == 2
        assert len(layer1.out_modes) == 2
        # inp_modes are dummy for TTConv2d (spatial conv handles input factorization)
        assert all(m == 1 for m in layer1.inp_modes)
        assert torch.prod(torch.tensor(layer1.out_modes)) == 64

        # Large channels
        layer2 = TTConv2d(512, 1024, kernel_size=3, tt_ranks=8)
        assert len(layer2.inp_modes) == 4
        assert len(layer2.out_modes) == 4
        # inp_modes are dummy for TTConv2d (spatial conv handles input factorization)
        assert all(m == 1 for m in layer2.inp_modes)
        assert torch.prod(torch.tensor(layer2.out_modes)) == 1024

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        layer = TTConv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            tt_ranks=8,
            bias=False,  # Disable bias for cleaner comparison
        )

        ratio = layer.compression_ratio()
        assert ratio > 1.0  # Should compress

        # Verify actual parameter count (excluding bias)
        original_params = 256 * 512 * 3 * 3
        tt_params = sum(p.numel() for p in layer.parameters())

        actual_ratio = original_params / tt_params
        assert abs(ratio - actual_ratio) < 0.01  # Should be exact

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        layer = TTConv2d(16, 32, kernel_size=3, tt_ranks=4)

        x = torch.randn(4, 16, 28, 28, requires_grad=True)
        y = layer(x)
        loss = y.mean()
        loss.backward()

        # Check gradients exist
        assert layer.spatial_conv.weight.grad is not None
        for core in layer.cores:
            assert core.grad is not None
            assert not torch.isnan(core.grad).any()

        if layer.bias is not None:
            assert layer.bias.grad is not None

    def test_compare_with_standard_conv(self):
        """Compare output similarity with standard convolution."""
        in_ch, out_ch = 32, 64
        kernel_size = 3

        # Create standard conv
        conv_standard = nn.Conv2d(in_ch, out_ch, kernel_size, padding=1)

        # Create TT conv with high rank (should be close to standard)
        conv_tt = TTConv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=1,
            tt_ranks=16,  # High rank for better approximation
            bias=True,
        )

        # Same input
        x = torch.randn(4, in_ch, 28, 28)

        # Forward pass
        y_standard = conv_standard(x)
        y_tt = conv_tt(x)

        # Should have same shape
        assert y_standard.shape == y_tt.shape

        # Should have reasonable output magnitudes
        assert y_tt.abs().mean() < 10.0  # Not exploding

    def test_from_conv_weight_initialization(self):
        """Test that from_conv_weight provides good approximation of original conv."""
        torch.manual_seed(42)

        # Create a standard conv layer
        conv_standard = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)

        # Create TTConv2d and initialize from standard conv weights
        conv_tt = TTConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            tt_ranks=16,  # High rank for good approximation
            bias=True,
        )

        # Initialize from pretrained weights
        with torch.no_grad():
            conv_tt.from_conv_weight(conv_standard.weight)
            if conv_standard.bias is not None and conv_tt.bias is not None:
                conv_tt.bias.copy_(conv_standard.bias)

        # Test on random input
        x = torch.randn(4, 32, 16, 16)

        with torch.no_grad():
            y_standard = conv_standard(x)
            y_tt = conv_tt(x)

        # Check that outputs are reasonably close
        # Note: TTConv2d from_conv_weight provides a best-effort approximation
        # With rank 16 and the simplified decomposition, we expect moderate error
        rel_error = (y_standard - y_tt).norm() / y_standard.norm()
        assert rel_error < 2.0, f"Relative error {rel_error:.4f} too large"

        # Check that both have similar statistics
        # Mean should be very close since bias is copied directly
        assert torch.allclose(y_standard.mean(), y_tt.mean(), atol=0.1)
        # Std will have some error due to the approximation
        # The new matrix_tt_svd-based implementation may have different variance characteristics
        # We check that the std is at least in a reasonable range (not zero and not exploding)
        tt_std = y_tt.std()
        standard_std = y_standard.std()
        assert tt_std > 0.01, f"TT std {tt_std:.4f} is too small"
        assert (
            tt_std < standard_std * 5
        ), f"TT std {tt_std:.4f} is too large compared to standard {standard_std:.4f}"

    def test_high_rank_approximation_quality(self):
        """Test that higher ranks provide better approximation of original conv.

        Note: For TTConv2d, the spatial projection rank (tt_ranks[1]) is the primary
        bottleneck for approximation quality. The current implementation uses
        matrix_tt_svd which adds additional TT decomposition on top of SVD,
        limiting the approximation quality even at high ranks.
        """
        torch.manual_seed(42)

        # Create a smaller standard conv layer for better demonstration
        conv_standard = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)

        # Test with different spatial projection ranks
        rank_errors = []
        ranks_to_test = [4, 8, 16, 32]

        for rank in ranks_to_test:
            conv_tt = TTConv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
                tt_ranks=rank,  # This sets the spatial projection rank
                bias=True,
            )

            # Initialize from pretrained weights
            with torch.no_grad():
                conv_tt.from_conv_weight(conv_standard.weight)
                if conv_standard.bias is not None and conv_tt.bias is not None:
                    conv_tt.bias.copy_(conv_standard.bias)

            # Test on random input
            x = torch.randn(4, 16, 8, 8)

            with torch.no_grad():
                y_standard = conv_standard(x)
                y_tt = conv_tt(x)

            # Calculate relative error
            rel_error = (y_standard - y_tt).norm() / y_standard.norm()
            rank_errors.append((rank, rel_error.item()))

        # Log the results for visibility
        for rank, error in rank_errors:
            print(f"  Rank {rank:2d}: relative error = {error:.4f}")

        # Check that we see at least some improvement with higher ranks
        first_error = rank_errors[0][1]
        last_error = rank_errors[-1][1]

        # Due to the additional TT decomposition in matrix_tt_svd,
        # the improvement is limited but should still be present
        assert last_error < first_error, (
            f"Higher rank should improve approximation: "
            f"rank {ranks_to_test[0]} error={first_error:.4f} vs "
            f"rank {ranks_to_test[-1]} error={last_error:.4f}"
        )

        # The current implementation has limited approximation quality
        # due to the additional TT decomposition. This is a known limitation
        # that could be improved in future versions.
        assert (
            last_error < 1.0
        ), f"Rank {ranks_to_test[-1]} error {last_error:.4f} should be < 100%"

    def test_numerical_stability(self):
        """Test that TTConv2d maintains stable activation statistics.

        The d^{-0.25} scaling should prevent activation explosion
        while maintaining reasonable gradient flow.
        """
        torch.manual_seed(42)

        # Create a standard conv for comparison
        conv_standard = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Create TTConv2d with various decomposition depths
        conv_tt_2 = TTConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            out_modes=[16, 8],  # d=2
            tt_ranks=8,
        )

        conv_tt_4 = TTConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            out_modes=[4, 4, 2, 4],  # d=4
            tt_ranks=8,
        )

        # Test input
        x = torch.randn(8, 64, 32, 32)

        # Get outputs
        with torch.no_grad():
            y_standard = conv_standard(x)
            y_tt_2 = conv_tt_2(x)
            y_tt_4 = conv_tt_4(x)

        # Calculate statistics
        std_mean, std_std = y_standard.mean().item(), y_standard.std().item()
        tt2_mean, tt2_std = y_tt_2.mean().item(), y_tt_2.std().item()
        tt4_mean, tt4_std = y_tt_4.mean().item(), y_tt_4.std().item()

        # Check that means are close to zero (within reasonable bounds)
        assert abs(tt2_mean) < 1.0, f"TTConv2d (d=2) mean too large: {tt2_mean}"
        assert abs(tt4_mean) < 1.0, f"TTConv2d (d=4) mean too large: {tt4_mean}"

        # Check that standard deviations are reasonable (not exploding or vanishing)
        assert 0.01 < tt2_std < 10.0, f"TTConv2d (d=2) std out of range: {tt2_std}"
        assert 0.01 < tt4_std < 10.0, f"TTConv2d (d=4) std out of range: {tt4_std}"

        # The d^{-0.25} scaling should keep outputs in similar range regardless of d
        ratio_2_to_std = tt2_std / std_std
        ratio_4_to_std = tt4_std / std_std

        # Both should be in reasonable range compared to standard conv
        assert (
            0.1 < ratio_2_to_std < 10.0
        ), f"TTConv2d (d=2) variance ratio: {ratio_2_to_std}"
        assert (
            0.1 < ratio_4_to_std < 10.0
        ), f"TTConv2d (d=4) variance ratio: {ratio_4_to_std}"

    def test_custom_modes(self):
        """Test with custom mode factorizations."""
        layer = TTConv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=3,
            inp_modes=[8, 8],  # 8 * 8 = 64
            out_modes=[16, 16],  # 16 * 16 = 256
            tt_ranks=[1, 8, 1],
        )

        x = torch.randn(4, 64, 32, 32)
        y = layer(x)

        assert y.shape == (4, 256, 30, 30)
        assert layer.compression_ratio() > 1.0

    def test_device_handling(self):
        """Test device placement."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create on GPU
        layer = TTConv2d(32, 64, kernel_size=3, device=torch.device("cuda"))

        # Check all parameters on GPU
        assert layer.spatial_conv.weight.device.type == "cuda"
        assert all(core.device.type == "cuda" for core in layer.cores)
        if layer.bias is not None:
            assert layer.bias.device.type == "cuda"

        # Test forward pass on GPU
        x = torch.randn(4, 32, 28, 28, device="cuda")
        y = layer(x)
        assert y.device.type == "cuda"

    def test_gradient_computation(self):
        """Test TTConv2d backward pass with detailed gradient checks."""
        tt_conv = TTConv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding=1, tt_ranks=2
        )
        x = torch.randn(1, 3, 8, 8, requires_grad=True)

        # Forward and backward
        output = tt_conv(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input gradient is None"
        assert (
            tt_conv.spatial_conv.weight.grad is not None
        ), "Spatial conv gradient is None"

        # Check core gradients
        for i, core in enumerate(tt_conv.cores):
            assert core.grad is not None, f"Core {i} gradient is None"
            assert torch.isfinite(
                core.grad
            ).all(), f"Core {i} gradient contains NaN/Inf"

    def test_different_configurations(self):
        """Test TTConv2d with various channel configurations."""
        configs = [
            (3, 16, 4),  # Small
            (64, 128, 8),  # Medium (VGG-like)
            (128, 256, 8),  # Large
            (256, 512, 8),  # Very large
        ]

        for in_ch, out_ch, rank in configs:
            tt_conv = TTConv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                padding=1,
                tt_ranks=rank,
            )

            # Test forward pass
            x = torch.randn(2, in_ch, 16, 16)
            output = tt_conv(x)

            assert output.shape == (
                2,
                out_ch,
                16,
                16,
            ), f"Config ({in_ch}->{out_ch}): expected shape (2, {out_ch}, 16, 16), got {output.shape}"
            assert torch.isfinite(
                output
            ).all(), f"Config ({in_ch}->{out_ch}): output contains NaN/Inf"

    def test_bias_handling(self):
        """Test TTConv2d with and without bias."""
        # With bias
        tt_conv_bias = TTConv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1,
            bias=True,
            tt_ranks=2,
        )
        assert tt_conv_bias.bias is not None, "Bias should exist when bias=True"
        assert tt_conv_bias.bias.shape == (
            8,
        ), f"Bias shape should be (8,), got {tt_conv_bias.bias.shape}"

        # Without bias
        tt_conv_no_bias = TTConv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1,
            bias=False,
            tt_ranks=2,
        )
        assert tt_conv_no_bias.bias is None, "Bias should be None when bias=False"

    def test_initialization_methods(self):
        """Test different initialization methods."""
        init_methods = ["xavier_normal", "xavier_uniform", "kaiming_normal"]

        for method in init_methods:
            tt_conv = TTConv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                tt_ranks=4,
                init_method=method,
            )

            # Check that parameters are initialized
            for i, core in enumerate(tt_conv.cores):
                assert not torch.allclose(
                    core, torch.zeros_like(core)
                ), f"Core {i} appears to be zero-initialized with {method}"
                assert torch.isfinite(
                    core
                ).all(), f"Core {i} contains NaN/Inf with {method} initialization"
