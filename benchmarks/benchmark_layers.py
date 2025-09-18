"""Benchmark TT layers against standard PyTorch layers."""

import argparse
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from torch_mpo.layers import TTConv2d, TTLinear


def benchmark_forward_pass(
    layer: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """Benchmark forward pass of a layer."""
    # Move to device
    layer = layer.to(device)
    layer.eval()

    # Create random input
    x = torch.randn(input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = layer(x)

    # Synchronize for accurate timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Time forward pass
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = layer(x)

    if device == "cuda":
        torch.cuda.synchronize()

    forward_time = (time.time() - start_time) / num_iterations * 1000  # ms

    # Memory usage
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = layer(x)
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = 0

    return {
        "forward_time_ms": forward_time,
        "memory_mb": memory_mb,
    }


def benchmark_backward_pass(
    layer: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """Benchmark backward pass of a layer."""
    # Move to device
    layer = layer.to(device)
    layer.train()

    # Create optimizer
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

    # Warmup
    for _ in range(warmup_iterations):
        x = torch.randn(input_shape, device=device, requires_grad=True)
        y = layer(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Synchronize for accurate timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Time backward pass
    start_time = time.time()
    for _ in range(num_iterations):
        x = torch.randn(input_shape, device=device, requires_grad=True)
        y = layer(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()

    backward_time = (time.time() - start_time) / num_iterations * 1000  # ms

    return {
        "backward_time_ms": backward_time,
    }


def compare_linear_layers(
    in_features: int,
    out_features: int,
    batch_size: int = 32,
    tt_ranks: List[int] = [4, 8, 16],
    device: str = "cuda",
) -> None:
    """Compare standard Linear with TTLinear."""
    print(f"\n{'='*80}")
    print(f"Linear Layer Comparison: {in_features} -> {out_features}")
    print(f"Batch size: {batch_size}, Device: {device}")
    print(f"{'='*80}")

    # Standard layer
    standard_layer = nn.Linear(in_features, out_features)
    input_shape = (batch_size, in_features)

    # Benchmark standard layer
    standard_forward = benchmark_forward_pass(
        standard_layer, input_shape, device=device
    )
    standard_backward = benchmark_backward_pass(
        standard_layer, input_shape, device=device
    )

    standard_params = sum(p.numel() for p in standard_layer.parameters())

    print("\nStandard Linear:")
    print(f"  Parameters: {standard_params:,}")
    print(f"  Forward time: {standard_forward['forward_time_ms']:.3f} ms")
    print(f"  Backward time: {standard_backward['backward_time_ms']:.3f} ms")
    if device == "cuda":
        print(f"  Memory usage: {standard_forward['memory_mb']:.2f} MB")

    # TT layers with different ranks
    for rank in tt_ranks:
        tt_layer = TTLinear(in_features, out_features, tt_ranks=rank)

        # Benchmark TT layer
        tt_forward = benchmark_forward_pass(tt_layer, input_shape, device=device)
        tt_backward = benchmark_backward_pass(tt_layer, input_shape, device=device)

        tt_params = sum(p.numel() for p in tt_layer.parameters())
        compression_ratio = standard_params / tt_params

        print(f"\nTTLinear (rank={rank}):")
        print(f"  Parameters: {tt_params:,} ({compression_ratio:.2f}x compression)")
        print(
            f"  Forward time: {tt_forward['forward_time_ms']:.3f} ms "
            f"({standard_forward['forward_time_ms'] / tt_forward['forward_time_ms']:.2f}x speedup)"
        )
        print(
            f"  Backward time: {tt_backward['backward_time_ms']:.3f} ms "
            f"({standard_backward['backward_time_ms'] / tt_backward['backward_time_ms']:.2f}x speedup)"
        )
        if device == "cuda":
            print(
                f"  Memory usage: {tt_forward['memory_mb']:.2f} MB "
                f"({standard_forward['memory_mb'] / tt_forward['memory_mb']:.2f}x reduction)"
            )


def compare_conv_layers(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    image_size: int = 32,
    batch_size: int = 32,
    tt_ranks: List[int] = [4, 8, 16],
    device: str = "cuda",
) -> None:
    """Compare standard Conv2d with TTConv2d."""
    print(f"\n{'='*80}")
    print(
        f"Conv2d Layer Comparison: {in_channels} -> {out_channels}, kernel={kernel_size}"
    )
    print(
        f"Image size: {image_size}x{image_size}, Batch size: {batch_size}, Device: {device}"
    )
    print(f"{'='*80}")

    # Standard layer
    standard_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
    input_shape = (batch_size, in_channels, image_size, image_size)

    # Benchmark standard layer
    standard_forward = benchmark_forward_pass(
        standard_layer, input_shape, device=device
    )
    standard_backward = benchmark_backward_pass(
        standard_layer, input_shape, device=device
    )

    standard_params = sum(p.numel() for p in standard_layer.parameters())

    print("\nStandard Conv2d:")
    print(f"  Parameters: {standard_params:,}")
    print(f"  Forward time: {standard_forward['forward_time_ms']:.3f} ms")
    print(f"  Backward time: {standard_backward['backward_time_ms']:.3f} ms")
    if device == "cuda":
        print(f"  Memory usage: {standard_forward['memory_mb']:.2f} MB")

    # TT layers with different ranks
    for rank in tt_ranks:
        tt_layer = TTConv2d(
            in_channels, out_channels, kernel_size, padding=1, tt_ranks=rank
        )

        # Benchmark TT layer
        tt_forward = benchmark_forward_pass(tt_layer, input_shape, device=device)
        tt_backward = benchmark_backward_pass(tt_layer, input_shape, device=device)

        tt_params = sum(p.numel() for p in tt_layer.parameters())
        compression_ratio = standard_params / tt_params

        print(f"\nTTConv2d (rank={rank}):")
        print(f"  Parameters: {tt_params:,} ({compression_ratio:.2f}x compression)")
        print(
            f"  Forward time: {tt_forward['forward_time_ms']:.3f} ms "
            f"({standard_forward['forward_time_ms'] / tt_forward['forward_time_ms']:.2f}x speedup)"
        )
        print(
            f"  Backward time: {tt_backward['backward_time_ms']:.3f} ms "
            f"({standard_backward['backward_time_ms'] / tt_backward['backward_time_ms']:.2f}x speedup)"
        )
        if device == "cuda":
            print(
                f"  Memory usage: {tt_forward['memory_mb']:.2f} MB "
                f"({standard_forward['memory_mb'] / tt_forward['memory_mb']:.2f}x reduction)"
            )


def benchmark_model_compression(device: str = "cuda") -> None:
    """Benchmark model compression on popular architectures."""
    import torchvision.models as models

    from torch_mpo.models import resnet50_mpo, vgg16_mpo

    print(f"\n{'='*80}")
    print("Model Compression Comparison")
    print(f"{'='*80}")

    # VGG-16
    print("\nVGG-16:")
    vgg_standard = models.vgg16(pretrained=False)
    vgg_mpo = vgg16_mpo(
        pretrained=False,
        tt_ranks_conv=8,
        tt_ranks_fc=16,
        compress_conv=True,
        compress_fc=True,
    )

    standard_params = sum(p.numel() for p in vgg_standard.parameters())
    mpo_params = sum(p.numel() for p in vgg_mpo.parameters())

    print(f"  Standard parameters: {standard_params:,}")
    print(f"  MPO parameters: {mpo_params:,}")
    print(f"  Compression ratio: {standard_params / mpo_params:.2f}x")

    # ResNet-50
    print("\nResNet-50:")
    resnet_standard = models.resnet50(pretrained=False)
    resnet_mpo = resnet50_mpo(
        pretrained=False,
        tt_ranks_conv=16,
        tt_ranks_fc=32,
        use_mpo_conv=True,
        use_mpo_fc=True,
    )

    standard_params = sum(p.numel() for p in resnet_standard.parameters())
    mpo_params = sum(p.numel() for p in resnet_mpo.parameters())

    print(f"  Standard parameters: {standard_params:,}")
    print(f"  MPO parameters: {mpo_params:,}")
    print(f"  Compression ratio: {standard_params / mpo_params:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MPO layers")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--linear-only", action="store_true", help="Only benchmark linear layers"
    )
    parser.add_argument(
        "--conv-only", action="store_true", help="Only benchmark conv layers"
    )
    parser.add_argument(
        "--model-only", action="store_true", help="Only benchmark model compression"
    )

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Running benchmarks on {args.device}")

    if not args.conv_only and not args.model_only:
        # Linear layer benchmarks
        compare_linear_layers(1024, 512, device=args.device)
        compare_linear_layers(4096, 4096, device=args.device)
        compare_linear_layers(25088, 4096, device=args.device)  # VGG classifier

    if not args.linear_only and not args.model_only:
        # Conv layer benchmarks
        compare_conv_layers(64, 128, device=args.device)
        compare_conv_layers(256, 512, device=args.device)
        compare_conv_layers(512, 512, device=args.device)

    if not args.linear_only and not args.conv_only:
        # Model compression benchmarks
        benchmark_model_compression(device=args.device)


if __name__ == "__main__":
    main()
