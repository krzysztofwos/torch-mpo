"""Model compression utilities for converting standard layers to TT format."""

from typing import cast

import torch
import torch.nn as nn

from torch_mpo.layers import TTConv2d, TTLinear


def compress_model(
    model: nn.Module,
    layers_to_compress: list[str] | None = None,
    compression_ratio: float = 0.1,
    tt_ranks: int | dict[str, int] | None = None,
    compress_linear: bool = True,
    compress_conv: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """
    Compress a PyTorch model by replacing layers with TT-decomposed versions.

    IMPORTANT: Conv2d Compression Limitations
    -----------------------------------------
    - Conv2d layers are randomly initialized (no weight reconstruction from pretrained)
    - Requires fine-tuning after compression for Conv2d layers
    - Linear layers can be initialized from pretrained weights via matrix_tt_svd

    Args:
        model: Model to compress
        layers_to_compress: list of layer names to compress. If None, compress all eligible layers.
        compression_ratio: Target compression ratio (0 < ratio < 1)
        tt_ranks: TT-ranks to use. Can be int (same for all) or dict mapping layer names to ranks.
        compress_linear: Whether to compress Linear layers (supports weight initialization)
        compress_conv: Whether to compress Conv2d layers (random init only, requires fine-tuning)
        verbose: Whether to print compression statistics

    Returns:
        Compressed model with TT layers
    """
    # Clone the model to avoid modifying the original
    compressed_model = _clone_model(model)

    # Find layers to compress
    if layers_to_compress is None:
        layers_to_compress = []
        if compress_linear:
            layers_to_compress.extend(_find_layers_by_type(compressed_model, nn.Linear))
        if compress_conv:
            layers_to_compress.extend(_find_layers_by_type(compressed_model, nn.Conv2d))

    # Compression statistics
    original_params = 0
    compressed_params = 0

    # Replace layers
    for layer_name in layers_to_compress:
        # Get the layer
        module_path = layer_name.split(".")
        parent = compressed_model
        for part in module_path[:-1]:
            parent = getattr(parent, part)

        old_layer = getattr(parent, module_path[-1])

        # Handle Linear layers
        if isinstance(old_layer, nn.Linear):
            # Get TT rank for this layer
            if isinstance(tt_ranks, dict):
                rank = tt_ranks.get(layer_name, 8)
            elif tt_ranks is not None:
                rank = tt_ranks
            else:
                # Auto-compute rank based on compression ratio
                rank = _compute_tt_rank_linear(
                    old_layer.in_features,
                    old_layer.out_features,
                    compression_ratio,
                )

            # Create TT layer
            tt_layer: TTLinear | TTConv2d = TTLinear(
                in_features=old_layer.in_features,
                out_features=old_layer.out_features,
                tt_ranks=rank,
                bias=old_layer.bias is not None,
                device=old_layer.weight.device,
                dtype=old_layer.weight.dtype,
            )

            # Initialize from original weights
            with torch.no_grad():
                if isinstance(tt_layer, TTLinear):
                    tt_layer.from_matrix(old_layer.weight.data)
                if old_layer.bias is not None:
                    tt_layer.bias.data = old_layer.bias.data.clone()

            # Update statistics
            old_params = old_layer.in_features * old_layer.out_features
            if old_layer.bias is not None:
                old_params += old_layer.out_features

            new_params = sum(p.numel() for p in tt_layer.parameters())

        # Handle Conv2d layers
        elif isinstance(old_layer, nn.Conv2d):
            # Skip depthwise/grouped convolutions for now
            if old_layer.groups != 1:
                if verbose:
                    print(f"Skipping {layer_name}: grouped convolution not supported")
                continue

            # Get TT rank for this layer
            if isinstance(tt_ranks, dict):
                rank = tt_ranks.get(layer_name, 8)
            elif tt_ranks is not None:
                rank = tt_ranks
            else:
                # Auto-compute rank based on compression ratio
                rank = _compute_tt_rank_conv(
                    old_layer.in_channels,
                    old_layer.out_channels,
                    old_layer.kernel_size,
                    compression_ratio,
                )

            # Create TT layer
            # Preserve tuple parameters (don't collapse to single int)
            # Cast to proper types for mypy
            kernel_size: int | tuple[int, int] = cast(
                int | tuple[int, int], old_layer.kernel_size
            )
            stride: int | tuple[int, int] = cast(
                int | tuple[int, int], old_layer.stride
            )
            dilation: int | tuple[int, int] = cast(
                int | tuple[int, int], old_layer.dilation
            )

            # Handle padding - can be tuple, int, or string
            padding: int | tuple[int, int]
            if isinstance(old_layer.padding, str):
                if old_layer.padding == "same":
                    # Only safe to translate to numeric padding for stride=1
                    s = stride if isinstance(stride, tuple) else (stride, stride)
                    if s != (1, 1):
                        if verbose:
                            print(
                                f"Skipping {layer_name}: padding='same' with stride={s} "
                                "requires asymmetric runtime padding; not supported."
                            )
                        continue
                    k = (
                        kernel_size
                        if isinstance(kernel_size, tuple)
                        else (kernel_size, kernel_size)
                    )
                    d = (
                        dilation
                        if isinstance(dilation, tuple)
                        else (dilation, dilation)
                    )
                    padding = ((d[0] * (k[0] - 1)) // 2, (d[1] * (k[1] - 1)) // 2)
                else:
                    if verbose:
                        print(
                            f"Skipping {layer_name}: padding='{old_layer.padding}' not supported"
                        )
                    continue
            else:
                padding = cast(int | tuple[int, int], old_layer.padding)

            tt_layer = TTConv2d(
                in_channels=old_layer.in_channels,
                out_channels=old_layer.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=old_layer.groups,
                bias=old_layer.bias is not None,
                tt_ranks=rank,
                device=old_layer.weight.device,
                dtype=old_layer.weight.dtype,
            )

            # Note: Conv2d weight initialization from pretrained is more complex
            # For now, we use random initialization which may need fine-tuning
            if verbose:
                print(
                    f"  Note: {layer_name} initialized randomly (Conv2d compression requires fine-tuning)"
                )

            # Copy bias if exists
            if old_layer.bias is not None:
                with torch.no_grad():
                    tt_layer.bias.data = old_layer.bias.data.clone()

            # Update statistics
            k_size = old_layer.kernel_size
            if isinstance(k_size, int):
                k_size = (k_size, k_size)
            old_params = (
                old_layer.in_channels * old_layer.out_channels * k_size[0] * k_size[1]
            )
            if old_layer.bias is not None:
                old_params += old_layer.out_channels

            new_params = sum(p.numel() for p in tt_layer.parameters())

        else:
            if verbose:
                print(
                    f"Skipping {layer_name}: unsupported layer type {type(old_layer)}"
                )
            continue

        # Replace the layer
        setattr(parent, module_path[-1], tt_layer)

        # Update total statistics
        original_params += old_params
        compressed_params += new_params

        if verbose:
            compression = old_params / new_params
            print(
                f"Compressed {layer_name}: {old_params:,} -> {new_params:,} params ({compression:.2f}x)"
            )

    if verbose and original_params > 0:
        total_compression = original_params / compressed_params
        print(
            f"\nTotal compression: {original_params:,} -> {compressed_params:,} params ({total_compression:.2f}x)"
        )

    return compressed_model


def _clone_model(model: nn.Module) -> nn.Module:
    """Create a deep copy of a model."""
    import copy

    return copy.deepcopy(model)


def _find_layers_by_type(
    model: nn.Module, layer_type: type, prefix: str = ""
) -> list[str]:
    """Find all layers of a specific type in a model."""
    layers = []

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(module, layer_type):
            layers.append(full_name)
        else:
            # Recurse
            layers.extend(_find_layers_by_type(module, layer_type, full_name))

    return layers


def _find_linear_layers(model: nn.Module, prefix: str = "") -> list[str]:
    """Find all Linear layers in a model."""
    return _find_layers_by_type(model, nn.Linear, prefix)


def _compute_tt_rank_linear(
    in_features: int,
    out_features: int,
    compression_ratio: float,
    max_rank: int = 50,
) -> int:
    """
    Compute TT-rank to achieve target compression ratio for Linear layer.

    Simple heuristic that assumes 4 modes for both input and output.
    """
    # Assume 4 modes (this could be made smarter)
    d = 4

    # Original parameters
    original = in_features * out_features

    # Target compressed parameters
    target = original * compression_ratio

    # Approximate: compressed ≈ d * sqrt(in_features) * sqrt(out_features) * r^2
    # where r is the TT-rank
    r_squared = target / (d * (in_features * out_features) ** 0.5)
    r = int(max(1, min(r_squared**0.5, max_rank)))

    return r


def _compute_tt_rank_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple,
    compression_ratio: float,
    max_rank: int = 50,
) -> int:
    """
    Compute TT-rank to achieve target compression ratio for Conv2d layer.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # Original parameters
    original = in_channels * out_channels * kernel_size[0] * kernel_size[1]

    # Target compressed parameters
    target = original * compression_ratio

    # For conv, we have spatial conv + TT cores
    # Spatial conv: in_channels * rank * kernel_size
    # Approximate best rank
    d = 3  # Typical decomposition depth for conv

    # Solve for rank (simplified heuristic)
    import math

    r = int(
        math.sqrt(
            target
            / (
                in_channels * kernel_size[0] * kernel_size[1]
                + d * math.sqrt(in_channels * out_channels)
            )
        )
    )
    r = max(1, min(r, max_rank))

    return r
