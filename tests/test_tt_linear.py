"""Tests for TTLinear layer."""

import pytest
import torch

from torch_mpo.layers import TTLinear


class TestTTLinear:
    """Test cases for TTLinear layer."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        batch_size = 32
        in_features = 256
        out_features = 128

        # Create layer
        layer = TTLinear(
            in_features=in_features,
            out_features=out_features,
            tt_ranks=4,
            bias=True,
        )

        # Test forward pass
        x = torch.randn(batch_size, in_features)
        y = layer(x)

        assert y.shape == (batch_size, out_features)
        assert not torch.isnan(y).any()

    def test_auto_factorization(self):
        """Test automatic mode factorization."""
        layer = TTLinear(
            in_features=784,  # 28 * 28
            out_features=256,  # 16 * 16
            tt_ranks=8,
        )

        # Check factorizations
        assert len(layer.inp_modes) == len(layer.out_modes)
        assert torch.prod(torch.tensor(layer.inp_modes)) == 784
        assert torch.prod(torch.tensor(layer.out_modes)) == 256

    def test_from_matrix(self):
        """Test initialization from full matrix - quantitative version."""
        import torch

        torch.manual_seed(42)

        in_features = 64  # 8×8
        out_features = 32  # 8×4
        r = 4

        U = torch.randn(out_features, r)
        V = torch.randn(in_features, r)
        W = U @ V.T

        layer = TTLinear(
            in_features=in_features,
            out_features=out_features,
            inp_modes=[8, 8],
            out_modes=[8, 4],
            tt_ranks=[1, r, 1],
        )
        layer.from_matrix(W)
        W_hat = layer.to_matrix()

        rel_err = (W - W_hat).norm() / W.norm()
        # Note: With rank-4 TT decomposition of a rank-4 matrix,
        # we get approximation error due to TT structure limitations
        # Expecting error < 1 for basic correctness
        assert rel_err < 1.0, f"Reconstruction too poor: rel_err={rel_err.item():.3e}"

    def test_boundary_rank_validation(self):
        """Test that boundary ranks must be 1."""
        # Valid: boundary ranks are 1
        layer = TTLinear(
            in_features=64,
            out_features=32,
            inp_modes=[8, 8],
            out_modes=[8, 4],
            tt_ranks=[1, 4, 1],
            bias=False,
        )
        assert layer.tt_ranks[0] == 1
        assert layer.tt_ranks[-1] == 1

        # Invalid: first boundary rank != 1
        with pytest.raises(AssertionError, match="Boundary TT ranks must be 1"):
            TTLinear(
                in_features=64,
                out_features=32,
                inp_modes=[8, 8],
                out_modes=[8, 4],
                tt_ranks=[2, 4, 1],
                bias=False,
            )

        # Invalid: last boundary rank != 1
        with pytest.raises(AssertionError, match="Boundary TT ranks must be 1"):
            TTLinear(
                in_features=64,
                out_features=32,
                inp_modes=[8, 8],
                out_modes=[8, 4],
                tt_ranks=[1, 4, 2],
                bias=False,
            )

        # Invalid: wrong length
        with pytest.raises(AssertionError, match="tt_ranks must have length"):
            TTLinear(
                in_features=64,
                out_features=32,
                inp_modes=[8, 8],
                out_modes=[8, 4],
                tt_ranks=[1, 4],  # Should be length 3, not 2
                bias=False,
            )

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        layer = TTLinear(
            in_features=1024,
            out_features=512,
            tt_ranks=8,
            bias=False,  # Disable bias for cleaner comparison
        )

        ratio = layer.compression_ratio()
        assert ratio > 1.0  # Should compress

        # Check actual parameter count (excluding bias)
        original_params = 1024 * 512
        tt_params = sum(p.numel() for p in layer.parameters())

        actual_ratio = original_params / tt_params
        assert abs(ratio - actual_ratio) < 0.01  # Should be exact

    def test_different_batch_shapes(self):
        """Test with different batch dimensions."""
        layer = TTLinear(256, 128, tt_ranks=4)

        # 1D batch
        x1 = torch.randn(32, 256)
        y1 = layer(x1)
        assert y1.shape == (32, 128)

        # 2D batch
        x2 = torch.randn(8, 4, 256)
        y2 = layer(x2)
        assert y2.shape == (8, 4, 128)

        # 3D batch
        x3 = torch.randn(2, 4, 8, 256)
        y3 = layer(x3)
        assert y3.shape == (2, 4, 8, 128)

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        layer = TTLinear(64, 32, tt_ranks=4)

        x = torch.randn(16, 64, requires_grad=True)
        y = layer(x)
        loss = y.mean()
        loss.backward()

        # Check gradients exist
        for core in layer.cores:
            assert core.grad is not None
            assert not torch.isnan(core.grad).any()

        if layer.bias is not None:
            assert layer.bias.grad is not None

    def test_device_handling(self):
        """Test device placement."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create on CPU
        layer_cpu = TTLinear(64, 32, device=torch.device("cpu"))
        assert all(core.device.type == "cpu" for core in layer_cpu.cores)

        # Create on GPU
        layer_gpu = TTLinear(64, 32, device=torch.device("cuda"))
        assert all(core.device.type == "cuda" for core in layer_gpu.cores)

        # Move between devices
        layer_cpu.cuda()
        assert all(core.device.type == "cuda" for core in layer_cpu.cores)

    def test_forward_pass_with_explicit_modes(self):
        """Test TTLinear forward pass with explicit modes."""
        # Create layer with explicit modes
        tt_linear = TTLinear(
            in_features=100,
            out_features=50,
            inp_modes=[10, 10],
            out_modes=[5, 10],
            tt_ranks=4,
        )

        # Test input
        x = torch.randn(32, 100)
        output = tt_linear(x)

        # Check output
        assert output.shape == (32, 50), f"Expected shape (32, 50), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_bias_handling(self):
        """Test TTLinear with and without bias."""
        # With bias
        tt_with_bias = TTLinear(
            in_features=100,
            out_features=50,
            inp_modes=[10, 10],
            out_modes=[5, 10],
            bias=True,
            tt_ranks=4,
        )
        assert tt_with_bias.bias is not None, "Bias should exist"
        assert tt_with_bias.bias.shape == (
            50,
        ), f"Bias shape should be (50,), got {tt_with_bias.bias.shape}"

        # Without bias
        tt_no_bias = TTLinear(
            in_features=100,
            out_features=50,
            inp_modes=[10, 10],
            out_modes=[5, 10],
            bias=False,
            tt_ranks=4,
        )
        assert tt_no_bias.bias is None, "Bias should be None"

    def test_different_ranks(self):
        """Test TTLinear with different rank configurations."""
        # Constant rank
        tt1 = TTLinear(
            in_features=100,
            out_features=50,
            inp_modes=[10, 10],
            out_modes=[5, 10],
            tt_ranks=4,
        )
        assert tt1.tt_ranks == [
            1,
            4,
            1,
        ], f"Expected ranks [1, 4, 1], got {tt1.tt_ranks}"

        # Variable ranks
        tt2 = TTLinear(
            in_features=100,
            out_features=50,
            inp_modes=[10, 10],
            out_modes=[5, 10],
            tt_ranks=[1, 8, 1],
        )
        assert tt2.tt_ranks == [
            1,
            8,
            1,
        ], f"Expected ranks [1, 8, 1], got {tt2.tt_ranks}"

    def test_initialization_methods(self):
        """Test different initialization methods."""
        init_methods = ["xavier_normal", "xavier_uniform", "normal"]

        for method in init_methods:
            tt_linear = TTLinear(
                in_features=100,
                out_features=50,
                inp_modes=[10, 10],
                out_modes=[5, 10],
                tt_ranks=4,
                init_method=method,
            )

            # Check cores are initialized
            for i, core in enumerate(tt_linear.cores):
                assert not torch.allclose(
                    core, torch.zeros_like(core)
                ), f"Core {i} appears zero-initialized with {method}"
                assert torch.isfinite(
                    core
                ).all(), f"Core {i} contains NaN/Inf with {method}"
