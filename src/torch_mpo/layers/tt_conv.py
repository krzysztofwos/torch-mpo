"""TT-decomposed convolutional layer implementation."""

import math

import numpy as np
import torch
import torch.nn as nn

from ..decomposition.tt_svd import matrix_tt_svd


class TTConv2d(nn.Module):
    """
    Tensor-Train decomposed 2D convolutional layer.

    IMPORTANT: Design and Limitations
    ----------------------------------
    This implementation uses a **spatial convolution → TT channel mixing** factorization:
    1. First, a spatial convolution projects input channels to a lower rank
    2. Then, TT cores mix the channels to produce output channels

    This is NOT a pure TT decomposition of the full 4D kernel tensor [out, in, kh, kw].
    Instead, it approximates channel interactions after a reduced spatial projection.
    This design is more efficient and stable in practice but differs from a theoretical
    full TT decomposition of the convolutional kernel.

    Current Limitations:
    - **Grouped/Depthwise convolutions**: Not supported (groups > 1 will raise ValueError)
    - **Spatial decomposition**: The decompose_spatial=True option is not implemented
    - **Padding='same'**: Only supported when it equals symmetric padding for stride=1

    Initialization from Pretrained Weights:
    The from_conv_weight() method allows initialization from pretrained conv weights
    by decomposing the kernel via SVD into spatial projection and TT channel mixing.
    Note: This requires decompose_spatial=False (the default).

    Compression vs. Accuracy Trade-off:
    The tt_ranks parameter controls the compression-accuracy trade-off:
    - Lower ranks (4-8): High compression but lower accuracy
    - Medium ranks (16-32): Balanced compression and accuracy
    - Higher ranks (64+): Lower compression but better accuracy
    For pretrained weight initialization, higher ranks preserve more information.

    Note on Initialization:
        TTConv2d requires careful initialization to maintain stable gradients.
        The TT cores are automatically scaled by 1/d^0.25 where d is the number
        of cores. This prevents activation explosion in deep networks while
        maintaining good gradient flow. Without this scaling, outputs can grow
        exponentially through layers (e.g., reaching 10^18 in deep ResNets).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        inp_modes: list[int] | None = None,
        out_modes: list[int] | None = None,
        tt_ranks: list[int] | int = 8,
        decompose_spatial: bool = False,
        init_method: str = "kaiming_normal",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize TT Conv2d layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides
            dilation: Spacing between kernel elements
            groups: Number of blocked connections (currently must be 1)
            bias: Whether to add a learnable bias
            inp_modes: Factorization of input channels. If None, auto-factorize.
            out_modes: Factorization of output channels. If None, auto-factorize.
            tt_ranks: TT-ranks for decomposition
            decompose_spatial: Whether to decompose spatial dimensions (experimental)
            init_method: Initialization method
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()

        if groups != 1:
            raise NotImplementedError("Grouped convolutions not yet supported")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.decompose_spatial = decompose_spatial
        self.init_method = init_method

        # Auto-factorize if modes not provided
        # For TTConv2d, inp_modes are not really used after spatial conv,
        # but we need them for consistency
        if out_modes is None:
            # Determine number of factors based on output channels
            if out_channels <= 64:
                d = 2
            elif out_channels <= 256:
                d = 3
            else:
                d = 4
            out_modes = self._auto_factorize(out_channels, d)

        # inp_modes can be dummy since we use spatial conv
        if inp_modes is None:
            inp_modes = [1] * len(out_modes)

        self.inp_modes = inp_modes
        self.out_modes = out_modes
        self.d = len(inp_modes)

        # Validate factorizations
        # For TTConv2d, inp_modes are dummy since spatial conv handles input factorization
        assert (
            np.prod(out_modes) == out_channels
        ), f"prod(out_modes)={np.prod(out_modes)} != {out_channels}"
        assert len(out_modes) == self.d, "inp_modes and out_modes must have same length"

        # Handle TT-ranks
        if isinstance(tt_ranks, int):
            self.tt_ranks = [1] + [tt_ranks] * (self.d - 1) + [1]
        else:
            self.tt_ranks = tt_ranks
            assert (
                len(self.tt_ranks) == self.d + 1
            ), f"tt_ranks must have length {self.d + 1}"
            assert (
                self.tt_ranks[0] == 1 and self.tt_ranks[-1] == 1
            ), "Boundary ranks must be 1"

        if not decompose_spatial:
            # Standard approach: spatial convolution + TT decomposition of channels
            # First conv: [H, W, in_channels, rank_1]
            self.spatial_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.tt_ranks[1],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=False,
                device=device,
                dtype=dtype,
            )

            # TT cores for channel transformation
            self.cores = nn.ParameterList()
            for i in range(self.d):
                if i == 0:
                    # First core: [rank_1, out_modes[0] * rank_2]
                    # Note: inp_modes[0] is conceptually 1 since spatial conv handles input channels
                    core_shape = (
                        self.tt_ranks[1],
                        (
                            self.out_modes[0] * self.tt_ranks[2]
                            if self.d > 1
                            else self.out_modes[0]
                        ),
                    )
                else:
                    # Regular cores: [rank_i+1, out_modes[i] * rank_{i+2}]
                    # For TTConv2d, inp_modes[i] is treated as 1 since spatial conv handles input factorization
                    core_shape = (
                        self.tt_ranks[i + 1],
                        (
                            self.out_modes[i] * self.tt_ranks[i + 2]
                            if i + 2 < len(self.tt_ranks)
                            else self.out_modes[i]
                        ),
                    )

                core = nn.Parameter(torch.empty(core_shape, device=device, dtype=dtype))
                self.cores.append(core)
        else:
            # Experimental: decompose spatial dimensions too
            raise NotImplementedError("Spatial decomposition not yet implemented")

        # Bias term
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        # Initialize spatial convolution
        # We need to scale down the spatial conv initialization because
        # the signal will be further multiplied by TT cores
        # The total variance should match a standard conv layer

        # For a standard conv: fan_in = in_channels * kernel_size^2
        # For TTConv: spatial conv produces rank_1 outputs, then TT cores map to out_channels
        # We need to distribute the variance properly

        # Initialize spatial convolution normally
        # We'll apply scaling to the TT cores instead
        nn.init.kaiming_normal_(
            self.spatial_conv.weight, mode="fan_out", nonlinearity="relu"
        )

        # Initialize TT cores
        # For TTConv2d, we need to be careful about initialization to maintain proper variance
        # The total transformation should preserve variance similar to a standard conv layer

        # Calculate the total number of cores to distribute the scaling
        total_cores = len(self.cores)

        # Scale factor to prevent variance explosion through multiple cores
        # We use a milder scaling to better match standard conv output variance
        # Empirically, 1/sqrt(sqrt(d)) works better than 1/sqrt(d)
        scale_factor = 1.0 / math.pow(total_cores, 0.25)

        for i, core in enumerate(self.cores):
            if i == 0:
                fan_in = self.tt_ranks[1]
                fan_out = self.out_modes[0] * (self.tt_ranks[2] if self.d > 1 else 1)
            else:
                fan_in = self.tt_ranks[i + 1]
                fan_out = self.out_modes[i] * (
                    self.tt_ranks[i + 2] if i + 2 < len(self.tt_ranks) else 1
                )

            if self.init_method == "xavier_normal":
                # Standard Xavier/Glorot initialization
                # Scale down to account for multiple cores
                std = math.sqrt(2.0 / (fan_in + fan_out)) * scale_factor
                nn.init.normal_(core, mean=0, std=std)
            elif self.init_method == "xavier_uniform":
                # Standard Xavier/Glorot uniform initialization
                bound = math.sqrt(6.0 / (fan_in + fan_out)) * scale_factor
                nn.init.uniform_(core, -bound, bound)
            elif self.init_method == "kaiming_normal":
                # Use fan_out mode which is better for ReLU networks
                # Scale down to prevent explosion through multiple cores
                std = math.sqrt(2.0 / fan_out) * scale_factor
                nn.init.normal_(core, mean=0, std=std)
            else:
                raise ValueError(f"Unknown init method: {self.init_method}")

        # Initialize bias
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def from_conv_weight(self, weight: torch.Tensor, epsilon: float = 1e-10) -> None:
        """
        Initialize TTConv2d from a pretrained conv weight tensor.

        This method decomposes the full 4D convolution kernel into:
        1. A spatial projection conv (tt_ranks[1] output channels)
        2. TT cores for channel mixing (tt_ranks[1] -> out_channels)

        Args:
            weight: Pretrained weight tensor of shape [out_channels, in_channels, kh, kw]
            epsilon: Relative threshold for SVD truncation in matrix_tt_svd

        Note:
            This provides a best-effort approximation of the original convolution.
            The approximation quality is limited by the TT structure and may require
            fine-tuning. The quality depends on tt_ranks[1] - higher ranks give
            better approximation but less compression. For production use, fine-tuning
            after initialization is strongly recommended.
        """
        if self.decompose_spatial:
            raise NotImplementedError(
                "from_conv_weight not supported with spatial decomposition"
            )

        # Get dimensions
        C_out, C_in, k_h, k_w = weight.shape
        assert (
            C_out == self.out_channels
        ), f"Weight has {C_out} output channels, expected {self.out_channels}"
        assert (
            C_in == self.in_channels
        ), f"Weight has {C_in} input channels, expected {self.in_channels}"
        assert (
            k_h,
            k_w,
        ) == self.kernel_size, f"Weight kernel size {(k_h, k_w)} != {self.kernel_size}"

        # Use tt_ranks[1] as the rank since that's what the spatial conv expects
        r = self.tt_ranks[1]

        # Flatten weight to matrix form [C_out, C_in * k_h * k_w]
        W = weight.reshape(C_out, C_in * k_h * k_w)

        # Thin SVD decomposition
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]  # truncate to rank r

        # Set spatial projection weights: Vh reshaped to [r, C_in, k_h, k_w]
        # Note: spatial_conv already has r output channels (tt_ranks[1])
        assert self.spatial_conv.out_channels == r
        self.spatial_conv.weight.copy_(Vh.reshape(r, C_in, k_h, k_w))

        # Channel mixer target matrix: U * S has shape [C_out, r]
        M = (U * S).to(self.cores[0].dtype)  # [C_out, r]

        # Use matrix_tt_svd to properly decompose M into TT cores
        # M is [C_out, r] where C_out = prod(out_modes) and r is the input dimension

        # Set up inp_modes to multiply to r
        # Simple choice: [r] + [1]*(d-1)
        inp_modes = [r] + [1] * (self.d - 1)
        out_modes = self.out_modes

        # Prepare ranks for matrix_tt_svd
        # The tt_ranks in TTConv are shifted by 1 compared to TTLinear
        # TTConv: [1, r1, r2, ..., rd, 1] where r1 is the spatial projection rank
        # For matrix_tt_svd we need ranks for the channel mixing part
        adjusted_ranks = [1] + list(self.tt_ranks[2:]) + [1]
        if len(adjusted_ranks) != self.d + 1:
            # If d=1, we have tt_ranks=[1, r, 1], need ranks=[1, 1]
            # If d=2, we have tt_ranks=[1, r, r2, 1], need ranks=[1, r2, 1]
            # Generally: use tt_ranks[2:] for internal ranks
            if self.d == 1:
                adjusted_ranks = [1, 1]
            else:
                # Ensure we have the right length
                adjusted_ranks = [1] + list(self.tt_ranks[2 : self.d + 1]) + [1]

        # Call matrix_tt_svd with correct dimensions
        tt_linear_cores = matrix_tt_svd(
            M,  # [C_out, r]
            inp_modes=inp_modes,  # product == r
            out_modes=out_modes,  # product == C_out
            ranks=adjusted_ranks,
            epsilon=epsilon,
        )

        # Now adapt TTLinear cores to TTConv cores
        # TTLinear cores are [r_i * out_i, r_{i+1} * in_i]
        # TTConv cores for i>0 are [r_{i+1}, out_i * r_{i+2}]
        # TTConv first core is [r, out_0 * r_2]

        for i, tt_linear_core in enumerate(tt_linear_cores):
            r_i = adjusted_ranks[i]
            r_ip1 = adjusted_ranks[i + 1]
            out_i = out_modes[i]
            in_i = inp_modes[i]  # in_0 == r, others == 1

            # TTLinear core is [r_i * out_i, r_{i+1} * in_i]
            # Reshape to [r_i, out_i, r_{i+1}, in_i]
            core4 = tt_linear_core.reshape(r_i, out_i, r_ip1, in_i)

            if i == 0:
                # First TTConv core expects shape [r, out_0 * r_2]
                # From TTLinear we have [r_i(=1), out_0, r_ip1, in_i(=r)]
                # Permute to [in_i, out_i, r_ip1, r_i] then squeeze r_i
                # Result: [r, out_0, r_ip1] -> flatten to [r, out_0 * r_ip1]
                if r_i == 1:
                    core3 = core4.squeeze(0).permute(2, 0, 1)  # [in_i, out_i, r_ip1]
                else:
                    core3 = core4.permute(3, 1, 2, 0).squeeze(
                        -1
                    )  # [in_i, out_i, r_ip1]
                conv_core = core3.reshape(in_i, out_i * r_ip1)  # [r, out_0 * r_ip1]
                self.cores[0].copy_(conv_core)
            else:
                # Regular TTConv core expects shape [r_{i+1}, out_i * r_{i+2}]
                # From TTLinear we have [r_i, out_i, r_ip1, in_i(=1)]
                # Since in_i == 1, squeeze it: [r_i, out_i, r_ip1]
                # Flatten to [r_i, out_i * r_ip1]
                core3 = core4.squeeze(-1)  # [r_i, out_i, r_ip1]
                conv_core = core3.reshape(r_i, out_i * r_ip1)
                self.cores[i].copy_(conv_core)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TT Conv2d layer.

        Args:
            input: Input tensor of shape [batch, in_channels, height, width]

        Returns:
            Output tensor of shape [batch, out_channels, height', width']
        """
        batch_size = input.shape[0]

        # Step 1: Apply spatial convolution
        # [batch, in_channels, H, W] -> [batch, rank_1, H', W']
        x = self.spatial_conv(input)

        # Get spatial dimensions after convolution
        h_out = x.shape[2]
        w_out = x.shape[3]

        # Step 2: Reshape for TT processing
        # [batch, rank_1, H', W'] -> [batch * H' * W', rank_1]
        x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, self.tt_ranks[1])

        # Step 3: Apply TT cores sequentially
        # This implements the contraction: x_{r_0 n_1 ... n_d r_d} = sum_r G^{(1)}_{r_0 n_1 r_1} ... G^{(d)}_{r_{d-1} n_d r_d}

        for i in range(self.d):
            core = self.cores[i]

            if i == 0:
                # First core: [rank_1, out_modes[0] * rank_2]
                # x: [spatial_points, rank_1] -> [spatial_points, out_modes[0] * rank_2]
                x = torch.matmul(x, core)

                if self.d > 1:
                    # Reshape to separate out_mode and next rank
                    # [spatial_points, out_modes[0] * rank_2] -> [spatial_points, out_modes[0], rank_2]
                    x = x.reshape(-1, self.out_modes[0], self.tt_ranks[2])
                    # Prepare for next core by flattening spatial and output modes
                    # [spatial_points, out_modes[0], rank_2] -> [spatial_points * out_modes[0], rank_2]
                    x = x.reshape(-1, self.tt_ranks[2])
                # If d==1, we're done with cores

            else:
                # Regular cores: [rank_i, out_modes[i] * rank_{i+1}]
                # x: [spatial_points * prod(out_modes[:i]), rank_i]
                x = torch.matmul(x, core)
                # -> [spatial_points * prod(out_modes[:i]), out_modes[i] * rank_{i+1}]

                if i < self.d - 1:
                    # Not the last core - prepare for next iteration
                    # [spatial_points * prod(out_modes[:i]), out_modes[i] * rank_{i+2}]
                    # -> [spatial_points * prod(out_modes[:i]), out_modes[i], rank_{i+2}]
                    x = x.reshape(-1, self.out_modes[i], self.tt_ranks[i + 2])
                    # -> [spatial_points * prod(out_modes[:i+1]), rank_{i+2}]
                    x = x.reshape(-1, self.tt_ranks[i + 2])
                else:
                    # Last core - just reshape to get final output channels
                    # [spatial_points * prod(out_modes[:-1]), out_modes[-1]]
                    x = x.reshape(-1, self.out_modes[i])

        # Step 4: Reshape back to conv output format
        # x: [batch * h_out * w_out, total_out_channels]
        x = x.reshape(batch_size, h_out, w_out, self.out_channels)
        x = x.permute(0, 3, 1, 2)

        # Step 5: Add bias
        if self.bias is not None:
            x = x + self.bias.reshape(1, -1, 1, 1)

        return x

    def compression_ratio(self) -> float:
        """Calculate compression ratio of TT decomposition."""
        # Original conv parameters
        original_params = (
            self.in_channels
            * self.out_channels
            * self.kernel_size[0]
            * self.kernel_size[1]
        )

        # TT conv parameters
        # Spatial conv: [H, W, in_channels, rank_1]
        spatial_params = self.spatial_conv.weight.numel()

        # TT cores
        tt_params = sum(core.numel() for core in self.cores)

        total_tt_params = spatial_params + tt_params

        return original_params / total_tt_params

    def extra_repr(self) -> str:
        """String representation with layer details."""
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != (0, 0):
            s += f", padding={self.padding}"
        if self.dilation != (1, 1):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        s += (
            f", inp_modes={self.inp_modes}, out_modes={self.out_modes}, "
            f"tt_ranks={self.tt_ranks}, "
            f"compression_ratio={self.compression_ratio():.2f}x"
        )
        return s

    @staticmethod
    def _auto_factorize(n: int, d: int | None = None) -> list[int]:
        """Auto-factorize a number into d factors."""
        if d is None:
            # For conv layers, use fewer modes since we handle spatial separately
            if n <= 64:
                d = 2
            elif n <= 256:
                d = 3
            else:
                d = 4

        # Handle special cases
        if d == 1:
            return [n]

        if n == 1:
            return [1] * d

        # Try to balance factors
        factors = []
        remaining = n

        for i in range(d - 1):
            target = int(remaining ** (1.0 / (d - i)))

            # Find the best divisor close to target
            best_f = 1
            for f in range(max(1, target - 2), target + 3):
                if f > 0 and remaining % f == 0:
                    # Prefer factors closer to target
                    if abs(f - target) < abs(best_f - target):
                        best_f = f

            factors.append(best_f)
            remaining //= best_f

        factors.append(remaining)

        # Verify the factorization is correct
        assert (
            np.prod(factors) == n
        ), f"Factorization failed: {factors} (product={np.prod(factors)}) != {n}"
        assert (
            len(factors) == d
        ), f"Wrong number of factors: got {len(factors)}, expected {d}"

        return factors


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Convert to pair of ints."""
    if isinstance(x, int):
        return (x, x)
    return x
