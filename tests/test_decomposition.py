"""Direct tests for TT decomposition helpers."""

import torch

from torch_mpo.decomposition.tt_svd import get_tt_ranks, matrix_tt_svd, tt_svd
from torch_mpo.layers import TTLinear


def _reconstruct_tt_tensor(cores: list[torch.Tensor]) -> torch.Tensor:
    """Reconstruct a tensor from TT cores."""
    result = cores[0]
    for core in cores[1:]:
        result = torch.tensordot(result, core, dims=([-1], [0]))
    return result.squeeze(0).squeeze(-1)


def test_tt_svd_reconstructs_tensor():
    """Test TT-SVD on a small tensor with sufficient ranks."""
    torch.manual_seed(0)
    tensor = torch.randn(2, 3, 4)

    cores = tt_svd(tensor, ranks=[1, 6, 4, 1], epsilon=0)
    reconstructed = _reconstruct_tt_tensor(cores)

    assert reconstructed.shape == tensor.shape
    assert torch.allclose(reconstructed, tensor, atol=1e-5, rtol=1e-5)


def test_tt_svd_truncates_with_epsilon():
    """Test TT-SVD epsilon-based truncation."""
    tensor = torch.zeros(2, 2, 2)
    tensor[0, 0, 0] = 3.0
    tensor[1, 1, 1] = 1e-6

    cores = tt_svd(tensor, ranks=[1, 4, 4, 1], epsilon=1e-3)

    assert cores[0].shape[-1] < 4
    reconstructed = _reconstruct_tt_tensor(cores)
    assert reconstructed.shape == tensor.shape


def test_matrix_tt_svd_reconstructs_matrix():
    """Test matrix TT-SVD reconstruction through TTLinear."""
    torch.manual_seed(0)
    matrix = torch.randn(8, 8)
    cores = matrix_tt_svd(
        matrix,
        inp_modes=[2, 4],
        out_modes=[2, 4],
        ranks=[1, 4, 1],
        epsilon=0,
    )

    layer = TTLinear(
        8,
        8,
        inp_modes=[2, 4],
        out_modes=[2, 4],
        tt_ranks=[1, 4, 1],
        bias=False,
    )

    with torch.no_grad():
        for param, core in zip(layer.cores, cores, strict=True):
            param.copy_(core)

    reconstructed = layer.to_matrix()
    assert torch.allclose(reconstructed, matrix, atol=1e-5, rtol=1e-5)


def test_get_tt_ranks_respects_bounds():
    """Test heuristic TT-rank generation."""
    ranks = get_tt_ranks([8, 8, 8], target_compression=0.25, max_rank=5)
    assert ranks[0] == 1
    assert ranks[-1] == 1
    assert all(1 <= rank <= 5 for rank in ranks)
    assert len(ranks) == 4
