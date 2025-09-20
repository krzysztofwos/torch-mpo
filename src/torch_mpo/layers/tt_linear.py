"""TT-decomposed fully connected layer implementation."""

import math

import numpy as np
import torch
import torch.nn as nn

from torch_mpo.decomposition import matrix_tt_svd


class TTLinear(nn.Module):
    """
    Tensor-Train decomposed linear layer.

    A linear transformation y = xW^T + b where the weight matrix W is
    decomposed into TT format to reduce parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        inp_modes: list[int] | None = None,
        out_modes: list[int] | None = None,
        tt_ranks: list[int] | int = 8,
        bias: bool = True,
        init_method: str = "xavier_normal",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize TT Linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            inp_modes: Factorization of input dimension. If None, auto-factorize.
            out_modes: Factorization of output dimension. If None, auto-factorize.
            tt_ranks: TT-ranks for decomposition. If int, same rank for all modes.
                      If list, must have length d+1 with boundary ranks (first and last) equal to 1.
            bias: Whether to include bias term
            init_method: Initialization method ("xavier_normal", "xavier_uniform", "normal", "from_matrix")
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method

        # Auto-factorize if modes not provided
        if inp_modes is None and out_modes is None:
            # Choose d based on the larger dimension
            d = None
            if max(in_features, out_features) <= 64:
                d = 2
            elif max(in_features, out_features) <= 256:
                d = 3
            else:
                d = 4
            inp_modes = self._auto_factorize(in_features, d)
            out_modes = self._auto_factorize(out_features, d)
        elif inp_modes is None:
            # Match the length of out_modes
            assert out_modes is not None
            inp_modes = self._auto_factorize(in_features, len(out_modes))
        elif out_modes is None:
            # Match the length of inp_modes
            out_modes = self._auto_factorize(out_features, len(inp_modes))

        assert inp_modes is not None
        assert out_modes is not None
        self.inp_modes = inp_modes
        self.out_modes = out_modes
        self.d = len(inp_modes)

        # Validate
        assert np.prod(inp_modes) == in_features
        assert np.prod(out_modes) == out_features
        assert len(out_modes) == self.d

        # Handle TT-ranks
        if isinstance(tt_ranks, int):
            self.tt_ranks = [1] + [tt_ranks] * (self.d - 1) + [1]
        else:
            # Validate list-based tt_ranks
            assert (
                len(tt_ranks) == self.d + 1
            ), f"tt_ranks must have length d+1={self.d+1}, got {len(tt_ranks)}"
            assert (
                tt_ranks[0] == 1 and tt_ranks[-1] == 1
            ), "Boundary TT ranks must be 1 (r0=rd=1)."
            self.tt_ranks = tt_ranks

        # Create TT cores as parameters
        self.cores = nn.ParameterList()
        for i in range(self.d):
            core_shape = (
                self.tt_ranks[i] * self.out_modes[i],
                self.tt_ranks[i + 1] * self.inp_modes[i],
            )
            core = nn.Parameter(torch.empty(core_shape, device=device, dtype=dtype))
            self.cores.append(core)

        # Bias term
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize TT cores and bias."""
        if self.init_method == "from_matrix":
            return

        # Initialize cores
        # The product of d matrices should preserve variance similar to a single matrix
        # For d cores, we want the product variance to be similar to standard initialization
        # This requires scaling each core by d^{-1/(2*d)} but in practice we use a simpler approach
        for i, core in enumerate(self.cores):
            fan_in = self.tt_ranks[i] * self.inp_modes[i]
            fan_out = self.tt_ranks[i + 1] * self.out_modes[i]

            if self.init_method == "xavier_normal":
                # Standard Xavier/Glorot initialization without extra scaling
                # The TT decomposition naturally provides some regularization
                std = math.sqrt(2.0 / (fan_in + fan_out))
                nn.init.normal_(core, mean=0, std=std)
            elif self.init_method == "xavier_uniform":
                # Standard Xavier/Glorot uniform initialization
                bound = math.sqrt(6.0 / (fan_in + fan_out))
                nn.init.uniform_(core, -bound, bound)
            elif self.init_method == "normal":
                # Simple normal initialization with fixed std
                nn.init.normal_(core, mean=0, std=0.02)

        # Initialize bias
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TT-decomposed linear layer.

        The algorithm contracts the input tensor with TT cores sequentially.
        For each core i, we contract over the input mode n_i and rank r_i.

        Args:
            input: Input tensor of shape [..., in_features]

        Returns:
            Output tensor of shape [..., out_features]
        """
        # Save batch shape
        batch_shape = input.shape[:-1]
        x = input.reshape(-1, self.in_features)
        batch_size = x.shape[0]

        # Reshape input to TT modes: [batch_size, n_1, n_2, ..., n_d]
        x = x.reshape(batch_size, *self.inp_modes)

        # Process cores one by one
        for i in range(self.d):
            # Get core and reshape
            core = self.cores[i]
            core_4d = core.reshape(
                self.tt_ranks[i],
                self.out_modes[i],
                self.tt_ranks[i + 1],
                self.inp_modes[i],
            )

            if i == 0:
                # First core (no left rank to contract)
                # x shape: [batch, n_0, n_1, ..., n_{d-1}]
                # core shape: [1, m_0, r_1, n_0]
                # We contract over n_0 (input mode 0)
                # Result shape: [batch, m_0, r_1, n_1, ..., n_{d-1}]

                x = torch.tensordot(x, core_4d.squeeze(0), dims=([1], [2]))
                # After tensordot, x has shape: [batch, n_1, ..., n_{d-1}, m_0, r_1]
                # We need to move m_0 and r_1 to positions 1 and 2
                perm = [0] + list(range(self.d, self.d + 2)) + list(range(1, self.d))
                x = x.permute(perm)
            elif i == self.d - 1:
                # Last core (no right rank in output)
                # x shape: [batch, m_0, ..., m_{d-2}, r_{d-1}, n_{d-1}]
                # core shape: [r_{d-1}, m_{d-1}, 1, n_{d-1}]
                # We contract over both r_{d-1} and n_{d-1}
                # Result shape: [batch, m_0, ..., m_{d-2}, m_{d-1}]

                # Flatten all dimensions except the last two
                x_shape = x.shape
                x = x.reshape(-1, x_shape[-2], x_shape[-1])

                # Reshape core for contraction: [r_{d-1}, n_{d-1}, m_{d-1}]
                core_3d = core_4d.squeeze(2).permute(0, 2, 1)

                # Contract using einsum
                x = torch.einsum("...ri, rij->...j", x, core_3d)

                # Reshape back to include all output modes
                x = x.reshape(*x_shape[:-2], self.out_modes[-1])
            else:
                # Middle cores (have both left and right ranks)
                # x shape: [batch, m_0, ..., m_{i-1}, r_i, n_i, n_{i+1}, ..., n_{d-1}]
                # core shape: [r_i, m_i, r_{i+1}, n_i]
                # We contract over r_i and n_i
                # Result shape: [batch, m_0, ..., m_{i-1}, m_i, r_{i+1}, n_{i+1}, ..., n_{d-1}]

                # Identify dimensions in current tensor
                # Position 0: batch
                # Positions 1 to i: output modes m_0, ..., m_{i-1}
                # Position i+1: left rank r_i
                # Position i+2: current input mode n_i
                # Positions i+3 to end: remaining input modes n_{i+1}, ..., n_{d-1}

                x_shape = x.shape
                pre_dims = x_shape[: i + 1]  # [batch, m_0, ..., m_{i-1}]
                r_i = x_shape[i + 1]  # r_i
                n_i = x_shape[i + 2]  # n_i
                post_dims = x_shape[i + 3 :]  # [n_{i+1}, ..., n_{d-1}]

                # Reshape to prepare for contraction
                # We need to group dimensions for efficient matrix multiplication
                x = x.reshape(-1, r_i, n_i, int(np.prod(post_dims)) if post_dims else 1)

                # Move the dimensions we're contracting to the end
                x = x.permute(0, 3, 1, 2)  # [..., post_dims_product, r_i, n_i]
                x = x.reshape(-1, r_i, n_i)

                # Prepare core for matrix multiplication
                # core: [r_i, m_i, r_{i+1}, n_i] -> [r_i * n_i, m_i * r_{i+1}]
                core_perm = core_4d.permute(0, 3, 1, 2).reshape(r_i * n_i, -1)

                # Contract via matrix multiplication
                x = x.reshape(-1, r_i * n_i) @ core_perm

                # Reshape to separate output mode and right rank
                x = x.reshape(-1, self.out_modes[i], self.tt_ranks[i + 1])

                # Restore all dimensions in the correct order
                if post_dims:
                    # First reshape to separate pre_dims and post_dims
                    x = x.reshape(
                        np.prod(pre_dims),
                        np.prod(post_dims),
                        self.out_modes[i],
                        self.tt_ranks[i + 1],
                    )
                    # Permute to correct order
                    x = x.permute(0, 2, 3, 1)
                    # Final reshape to restore all individual dimensions
                    x = x.reshape(
                        *pre_dims, self.out_modes[i], self.tt_ranks[i + 1], *post_dims
                    )
                else:
                    x = x.reshape(*pre_dims, self.out_modes[i], self.tt_ranks[i + 1])

        # Final reshape
        x = x.reshape(batch_size, self.out_features)

        # Add bias
        if self.bias is not None:
            x = x + self.bias

        # Restore batch shape
        x = x.reshape(*batch_shape, self.out_features)

        return x

    def from_matrix(self, matrix: torch.Tensor, epsilon: float = 1e-10) -> None:
        """Initialize TT cores from a full matrix using TT-SVD."""
        cores = matrix_tt_svd(
            matrix,
            self.inp_modes,
            self.out_modes,
            self.tt_ranks,
            epsilon,
        )

        for i, core in enumerate(cores):
            self.cores[i].data = core

    def to_matrix(self) -> torch.Tensor:
        """Reconstruct the full weight matrix from TT cores."""
        # Start with the first core and reshape it properly
        # Core 0 shape: [rank_0 * out_modes[0], rank_1 * inp_modes[0]]
        # Since rank_0 = 1, this is [out_modes[0], rank_1 * inp_modes[0]]
        result = self.cores[
            0
        ].t()  # Transpose to get [rank_1 * inp_modes[0], out_modes[0]]

        for i in range(1, self.d):
            # Current result shape: [rank_i * prod(inp_modes[:i]), prod(out_modes[:i])]
            # Next core shape: [rank_i * out_modes[i], rank_{i+1} * inp_modes[i]]

            # Get dimensions
            n_out_prev = np.prod(self.out_modes[:i])
            n_inp_prev = np.prod(self.inp_modes[:i])
            rank_i = self.tt_ranks[i]
            rank_ip1 = self.tt_ranks[i + 1]
            out_i = self.out_modes[i]
            inp_i = self.inp_modes[i]

            # Reshape result for contraction
            # [rank_i * prod(inp_modes[:i]), prod(out_modes[:i])]
            # -> [rank_i, prod(inp_modes[:i]), prod(out_modes[:i])]
            result = result.reshape(rank_i, n_inp_prev, n_out_prev)

            # Reshape core for contraction
            # [rank_i * out_modes[i], rank_{i+1} * inp_modes[i]]
            # -> [rank_i, out_modes[i], rank_{i+1}, inp_modes[i]]
            core = self.cores[i].reshape(rank_i, out_i, rank_ip1, inp_i)

            # Contract over rank_i
            # result: [rank_i, prod(inp_modes[:i]), prod(out_modes[:i])]
            # core: [rank_i, out_modes[i], rank_{i+1}, inp_modes[i]]
            # Contract over dimension 0 (rank_i)
            result = torch.tensordot(result, core, dims=([0], [0]))
            # Result shape: [prod(inp_modes[:i]), prod(out_modes[:i]), out_modes[i], rank_{i+1}, inp_modes[i]]

            # Reshape for next iteration
            # Want: [rank_{i+1} * prod(inp_modes[:i+1]), prod(out_modes[:i+1])]
            result = result.permute(
                3, 0, 4, 1, 2
            )  # [rank_{i+1}, prod(inp_modes[:i]), inp_modes[i], prod(out_modes[:i]), out_modes[i]]
            result = result.reshape(rank_ip1 * n_inp_prev * inp_i, n_out_prev * out_i)

        # Final transpose to get [out_features, in_features]
        result = result.t()
        return result

    def compression_ratio(self) -> float:
        """Calculate compression ratio of TT decomposition."""
        # Original parameters
        original_params = self.in_features * self.out_features

        # TT parameters
        tt_params = sum(
            self.tt_ranks[i]
            * self.out_modes[i]
            * self.tt_ranks[i + 1]
            * self.inp_modes[i]
            for i in range(self.d)
        )

        return original_params / tt_params

    def extra_repr(self) -> str:
        """String representation with layer details."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"inp_modes={self.inp_modes}, "
            f"out_modes={self.out_modes}, "
            f"tt_ranks={self.tt_ranks}, "
            f"compression_ratio={self.compression_ratio():.2f}x, "
            f"bias={self.bias is not None}"
        )

    @staticmethod
    def _auto_factorize(n: int, d: int | None = None) -> list[int]:
        """Auto-factorize a number into d factors."""
        if d is None:
            # Choose d based on n
            if n <= 64:
                d = 2
            elif n <= 256:
                d = 3
            else:
                d = 4

        # Simple factorization: try to make factors as equal as possible
        factors = []
        remaining = n

        for i in range(d - 1):
            # Find factor close to d-th root
            target = int(remaining ** (1.0 / (d - i)))

            # Find closest divisor
            for f in range(target, 0, -1):
                if remaining % f == 0:
                    factors.append(f)
                    remaining //= f
                    break

        factors.append(remaining)

        # Verify
        assert np.prod(factors) == n, f"Factorization failed: {factors} != {n}"

        return factors
