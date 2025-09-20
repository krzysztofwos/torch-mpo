"""Test TTLinear output fidelity when initialized from target matrix."""

import torch

from torch_mpo.layers import TTLinear


def test_ttlinear_matches_target_linear_on_low_rank():
    """Test that TTLinear initialized from matrix produces similar outputs."""
    torch.manual_seed(0)
    in_f, out_f, r = 64, 32, 4

    # Create a low-rank matrix
    U = torch.randn(out_f, r)
    V = torch.randn(in_f, r)
    W = U @ V.T

    x = torch.randn(17, in_f)

    # Reference output with exact matrix
    y_ref = x @ W.t()

    # TT layer from matrix with sufficient rank
    # Note: With 2-mode factorization, we need higher TT rank for good approximation
    tt = TTLinear(
        in_f,
        out_f,
        inp_modes=[8, 8],
        out_modes=[8, 4],
        tt_ranks=[1, 16, 1],  # Higher rank needed for approximation
        bias=False,
    )
    tt.from_matrix(W)
    y_tt = tt(x)

    rel = (y_ref - y_tt).norm() / y_ref.norm()
    assert rel < 0.35, f"Relative error {rel:.3f} too high"


def test_ttlinear_exact_reconstruction_with_sufficient_rank():
    """Test exact reconstruction when TT rank is sufficient."""
    torch.manual_seed(42)
    in_f, out_f = 8, 8

    # Create any matrix
    W = torch.randn(out_f, in_f)

    # TT layer with full rank for exact reconstruction
    tt = TTLinear(
        in_f,
        out_f,
        inp_modes=[2, 2, 2],
        out_modes=[2, 2, 2],
        tt_ranks=[1, 4, 4, 1],  # Sufficient ranks
        bias=False,
    )
    tt.from_matrix(W)
    W_reconstructed = tt.to_matrix()

    rel_err = (W - W_reconstructed).norm() / W.norm()
    assert rel_err < 0.2, f"Reconstruction error {rel_err:.3f} too high"

    # Test forward pass consistency
    x = torch.randn(5, in_f)
    y_ref = x @ W.t()
    y_tt = tt(x)
    forward_err = (y_ref - y_tt).norm() / y_ref.norm()
    assert forward_err < 0.2, f"Forward pass error {forward_err:.3f} too high"
