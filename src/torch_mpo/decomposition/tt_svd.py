"""TT-SVD decomposition algorithms for tensors and matrices."""

import numpy as np
import torch


def tt_svd(
    tensor: torch.Tensor,
    ranks: list[int] | int,
    epsilon: float = 1e-10,
) -> list[torch.Tensor]:
    """
    Decompose a tensor into TT (Tensor Train) format using TT-SVD algorithm.

    Args:
        tensor: Input tensor to decompose
        ranks: TT-ranks for decomposition. If int, same rank for all modes.
               If list, must have length ndim+1 with boundary ranks=1
        epsilon: Relative threshold for SVD truncation. Singular values below
               epsilon * max(singular_value) are truncated. Set to 0 to disable

    Returns:
        List of TT cores
    """
    ndim = tensor.ndim
    shape = list(tensor.shape)

    # Handle rank specification
    if isinstance(ranks, int):
        ranks = [1] + [ranks] * (ndim - 1) + [1]
    else:
        assert len(ranks) == ndim + 1, f"ranks must have length {ndim + 1}"
        assert ranks[0] == 1 and ranks[-1] == 1, "Boundary ranks must be 1"

    # Initialize
    cores = []
    C = tensor

    for i in range(ndim - 1):
        # Reshape for SVD
        n_rows = ranks[i] * shape[i]
        n_cols = C.numel() // n_rows
        C = C.reshape(n_rows, n_cols)

        # Perform SVD using torch.linalg.svd
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)

        # Truncate based on rank and epsilon threshold
        # First, determine truncation based on epsilon (singular value threshold)
        if epsilon > 0:
            # Find where singular values drop below epsilon * max(S)
            s_max = S[0] if S.numel() > 0 else 1.0
            keep_indices = S > epsilon * s_max
            epsilon_rank = keep_indices.sum().item()
        else:
            epsilon_rank = S.shape[0]

        # Take minimum of requested rank and epsilon-based rank
        rank = min(ranks[i + 1], U.shape[1], epsilon_rank)
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        # Store core
        core = U.reshape(ranks[i], shape[i], rank)
        cores.append(core)

        # Update C for next iteration
        C = S[:, None] * Vh  # Efficient computation of S @ Vh

    # Last core
    cores.append(C.reshape(ranks[-2], shape[-1], ranks[-1]))

    return cores


def matrix_tt_svd(
    matrix: torch.Tensor,
    inp_modes: list[int],
    out_modes: list[int],
    ranks: list[int] | int,
    epsilon: float = 1e-10,
) -> list[torch.Tensor]:
    """
    Decompose a matrix into TT format using proper TT-SVD algorithm.

    This creates TT cores compatible with TTLinear layer format.
    Each core has shape [r_i * out_mode_i, r_{i+1} * inp_mode_i].

    Args:
        matrix: Input matrix of shape [out_dim, in_dim]
        inp_modes: Factorization of input dimension
        out_modes: Factorization of output dimension
        ranks: TT-ranks for decomposition (length should be d+1 where d=len(inp_modes))
        epsilon: Relative threshold for SVD truncation. Singular values below
               epsilon * max(singular_value) are truncated. Set to 0 to disable

    Returns:
        List of TT cores for the matrix
    """
    assert matrix.ndim == 2, "Input must be a matrix"
    out_dim, in_dim = matrix.shape

    # Verify factorizations
    assert (
        np.prod(inp_modes) == in_dim
    ), f"prod(inp_modes)={np.prod(inp_modes)} != {in_dim}"
    assert (
        np.prod(out_modes) == out_dim
    ), f"prod(out_modes)={np.prod(out_modes)} != {out_dim}"

    # Number of modes
    d = len(inp_modes)
    assert len(out_modes) == d, "inp_modes and out_modes must have same length"

    # Handle rank specification
    if isinstance(ranks, int):
        ranks = [1] + [ranks] * (d - 1) + [1]
    else:
        assert len(ranks) == d + 1, f"ranks must have length {d + 1}"
        assert ranks[0] == 1 and ranks[-1] == 1, "Boundary ranks must be 1"

    # Step 1: Reshape matrix into higher-order tensor
    # Matrix [out_total, in_total] -> [out_0, out_1, ..., in_0, in_1, ...]
    # Then permute to [out_0, in_0, out_1, in_1, ...]
    tensor = matrix.reshape(out_modes + inp_modes)

    # Create permutation to interleave output and input modes
    perm = [i for p in zip(range(d), range(d, 2 * d)) for i in p]
    C = tensor.permute(perm).contiguous()  # Safe to reshape next

    # Step 2: Apply TT-SVD to the permuted tensor
    cores = []
    r_left = ranks[0]  # == 1 by contract

    for i in range(d - 1):
        out_i, in_i = out_modes[i], inp_modes[i]
        left_size = r_left * out_i * in_i
        C_mat = C.reshape(left_size, -1)

        # Thin SVD using torch.linalg.svd
        U, S, Vh = torch.linalg.svd(C_mat, full_matrices=False)

        # Truncate based on rank and epsilon threshold
        if epsilon > 0:
            s_max = S[0] if S.numel() > 0 else 1.0
            keep_indices = S > epsilon * s_max
            epsilon_rank = keep_indices.sum().item()
        else:
            epsilon_rank = S.shape[0]

        r_right = min(ranks[i + 1], U.shape[1], epsilon_rank)
        U = U[:, :r_right]
        S = S[:r_right]
        Vh = Vh[:r_right, :]

        # Core_i: [r_left, out_i, r_right, in_i] -> flatten to [r_left*out_i, r_right*in_i]
        core = (
            U.reshape(r_left, out_i, in_i, r_right)
            .permute(0, 1, 3, 2)
            .reshape(r_left * out_i, r_right * in_i)
        )
        cores.append(core)

        # Carry remaining C = Sigma * Vh and reshape for next step
        C = S[:, None] * Vh  # [r_right, ...]
        # Remaining dims: [out_{i+1}, in_{i+1}, ..., out_{d-1}, in_{d-1}]
        remaining = []
        for j in range(i + 1, d):
            remaining.extend([out_modes[j], inp_modes[j]])
        C = C.reshape(r_right, *remaining)
        r_left = r_right

    # Last core is the remainder C: [r_left, out_{d-1}, in_{d-1}] (since r_right == 1)
    out_last, in_last = out_modes[-1], inp_modes[-1]
    C = C.reshape(r_left, out_last, in_last)  # ranks[-1] == 1
    core_last = C.reshape(r_left * out_last, 1 * in_last)
    cores.append(core_last)

    return cores


def get_tt_ranks(
    shape: list[int],
    target_compression: float = 0.1,
    max_rank: int = 50,
) -> list[int]:
    """
    Compute TT-ranks to achieve target compression ratio.

    Args:
        shape: Shape of the tensor to decompose
        target_compression: Target compression ratio (compressed_size / original_size)
        max_rank: Maximum allowed rank

    Returns:
        List of TT-ranks including boundary ranks
    """
    d = len(shape)

    # Simple heuristic: use same rank for all modes
    # Compression ratio ≈ (d * n * r^2) / (n^d) for tensor of shape [n, n, ..., n]
    avg_mode_size = np.mean(shape)
    r = int(np.sqrt(target_compression * np.prod(shape) / (d * avg_mode_size)))
    r = min(max(r, 1), max_rank)

    ranks = [1] + [r] * (d - 1) + [1]
    return ranks
