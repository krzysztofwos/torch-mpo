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
        epsilon: Threshold for SVD truncation

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

        # Perform SVD
        U, S, V = torch.svd(C)

        # Truncate to rank
        rank = min(ranks[i + 1], U.shape[1])
        U = U[:, :rank]
        S = S[:rank]
        V = V[:, :rank]

        # Store core
        core = U.reshape(ranks[i], shape[i], rank)
        cores.append(core)

        # Update C for next iteration
        C = torch.diag(S) @ V.t()

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
        epsilon: Threshold for SVD truncation

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

    # Step 1: Reshape matrix into higher-order tensor with mixed modes
    # Matrix [out_total, in_total] -> [out_0, out_1, ..., in_0, in_1, ...]
    # Then permute to [out_0, in_0, out_1, in_1, ...]
    tensor = matrix.reshape(out_modes + inp_modes)

    # Create permutation to interleave output and input modes
    perm = []
    for i in range(d):
        perm.append(i)  # out_mode[i]
        perm.append(i + d)  # inp_mode[i]
    tensor = tensor.permute(perm)

    # Step 2: Apply TT-SVD to the permuted tensor
    # The tensor now has shape [out_0, inp_0, out_1, inp_1, ..., out_{d-1}, inp_{d-1}]
    # We'll process pairs of modes at a time
    cores = []
    C = tensor

    for i in range(d):
        # Get current mode sizes
        out_mode = out_modes[i]
        inp_mode = inp_modes[i]

        # Reshape C for SVD
        if i == 0:
            # First core: no left rank
            # C shape: [out_0, inp_0, rest...]
            left_size = out_mode * inp_mode
            right_size = C.numel() // left_size
            C_mat = C.reshape(left_size, right_size)
        else:
            # Other cores: have left rank
            # C shape: [r_i, out_i, inp_i, rest...]
            left_size = ranks[i] * out_mode * inp_mode
            right_size = C.numel() // left_size
            C_mat = C.reshape(left_size, right_size)

        # Perform SVD
        U, S, V = torch.svd(C_mat)

        # Determine rank
        if i == d - 1:
            # Last core: no truncation needed, right rank must be 1
            rank = 1
        else:
            # Truncate to specified rank
            rank = min(ranks[i + 1], U.shape[1], V.shape[1])

        U = U[:, :rank]
        S = S[:rank]
        V = V[:, :rank]

        # Reshape U to get the core
        if i == 0:
            # First core: [out_0 * inp_0, r_1] -> [out_0, inp_0, r_1]
            U_tensor = U.reshape(out_mode, inp_mode, rank)
            # Permute and reshape to [r_0 * out_0, r_1 * inp_0] format
            # Since r_0 = 1: [out_0, r_1 * inp_0]
            core = U_tensor.permute(0, 2, 1).reshape(out_mode, rank * inp_mode)
        else:
            # Middle/last core: [r_i * out_i * inp_i, r_{i+1}]
            U_tensor = U.reshape(ranks[i], out_mode, inp_mode, rank)
            # Permute to [r_i, out_i, r_{i+1}, inp_i]
            U_tensor = U_tensor.permute(0, 1, 3, 2)
            # Reshape to [r_i * out_i, r_{i+1} * inp_i]
            core = U_tensor.reshape(ranks[i] * out_mode, rank * inp_mode)

        cores.append(core)

        # Update C for next iteration
        if i < d - 1:
            C = torch.diag(S) @ V.t()
            # C now has shape [r_{i+1}, right_size]
            # We need to reshape it back to tensor form for next iteration
            # The remaining dimensions are [out_{i+1}, inp_{i+1}, ..., out_{d-1}, inp_{d-1}]
            remaining_shape = []
            for j in range(i + 1, d):
                remaining_shape.extend([out_modes[j], inp_modes[j]])
            C = C.reshape([rank] + remaining_shape)

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
