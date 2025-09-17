"""Example of compressing a pretrained VGG model using MPO."""

import argparse

import torch
import torchvision.models as models

from torch_mpo import compress_model


def main():
    parser = argparse.ArgumentParser(description="Compress VGG with MPO")
    parser.add_argument(
        "--model", type=str, default="vgg16", choices=["vgg16", "vgg19"]
    )
    parser.add_argument(
        "--compression-ratio", type=float, default=0.1, help="Target compression ratio"
    )
    parser.add_argument(
        "--tt-rank", type=int, default=None, help="TT-rank (if None, auto-compute)"
    )

    args = parser.parse_args()

    # Load pretrained model
    print(f"Loading pretrained {args.model}...")
    if args.model == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)

    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {original_params:,}")

    # Compress classifier layers (they have most parameters)
    layers_to_compress = [
        "classifier.0",  # First FC layer (most parameters)
        "classifier.3",  # Second FC layer
        "classifier.6",  # Third FC layer
    ]

    print(f"\nCompressing layers: {layers_to_compress}")
    print(f"Target compression ratio: {args.compression_ratio}")

    # Compress model
    compressed_model = compress_model(
        model,
        layers_to_compress=layers_to_compress,
        compression_ratio=args.compression_ratio,
        tt_ranks=args.tt_rank,
        verbose=True,
    )

    # Count compressed parameters
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    overall_compression = original_params / compressed_params

    print(f"\nCompressed model parameters: {compressed_params:,}")
    print(f"Overall compression: {overall_compression:.2f}x")
    print(f"Parameters saved: {original_params - compressed_params:,}")

    # Test inference speed
    print("\nTesting inference speed...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    compressed_model = compressed_model.to(device).eval()

    # Dummy input
    x = torch.randn(1, 3, 224, 224).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
            _ = compressed_model(x)

    # Time original model
    import time

    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    original_time = time.time() - start

    # Time compressed model
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = compressed_model(x)
    compressed_time = time.time() - start

    print(f"\nOriginal model: {original_time:.3f}s for 100 iterations")
    print(f"Compressed model: {compressed_time:.3f}s for 100 iterations")
    print(f"Speedup: {original_time / compressed_time:.2f}x")

    # Verify outputs are similar
    with torch.no_grad():
        out_original = model(x)
        out_compressed = compressed_model(x)

    # Check similarity (cosine similarity of logits)
    cos_sim = torch.nn.functional.cosine_similarity(
        out_original.view(-1), out_compressed.view(-1), dim=0
    )
    print(f"\nOutput similarity (cosine): {cos_sim.item():.4f}")

    # Save compressed model
    save_path = f"{args.model}_mpo_compressed.pth"
    torch.save(compressed_model.state_dict(), save_path)
    print(f"\nCompressed model saved to: {save_path}")


if __name__ == "__main__":
    main()
