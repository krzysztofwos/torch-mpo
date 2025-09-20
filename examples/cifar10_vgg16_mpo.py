"""CIFAR-10 classification with VGG-16 MPO."""

import argparse
import time

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch_mpo.models import VGG16_MPO


def train(model, device, train_loader, optimizer, epoch):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 50 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item():.6f}\t"
                f"Acc: {100. * correct / total:.2f}%"
            )

    accuracy = 100.0 * correct / total
    avg_loss = train_loss / len(train_loader)
    print(f"Train set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy


def test(model, device, test_loader):
    """Evaluate on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= total
    accuracy = 100.0 * correct / total

    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

    return accuracy


def print_compression_stats(model):
    """Print detailed compression statistics."""
    stats = model.compression_stats()

    print("\n" + "=" * 60)
    print("MPO Compression Statistics")
    print("=" * 60)

    print(f"\nTotal parameters: {stats['total_params']:,}")

    if stats["conv_compression"]:
        print("\nConvolutional layers:")
        for layer in stats["conv_compression"]:
            print(
                f"  {layer['name']}: {layer['in_channels']} -> {layer['out_channels']} "
                f"(compression: {layer['compression_ratio']:.2f}x)"
            )

    if stats["fc_compression"]:
        print("\nFully connected layers:")
        for layer in stats["fc_compression"]:
            print(
                f"  {layer['name']}: {layer['in_features']} -> {layer['out_features']} "
                f"(compression: {layer['compression_ratio']:.2f}x)"
            )

    # Compare with standard VGG-16 for same configuration
    # Note: Standard ImageNet VGG-16 has 138,357,544 parameters
    # For CIFAR-10 with 10 classes, we need to calculate the actual baseline
    # Conv layers: same as ImageNet version
    # FC layers: 512*7*7*4096 + 4096*4096 + 4096*10 (vs 4096*1000 for ImageNet)
    conv_params = 14714688  # Fixed for VGG-16 conv layers
    fc_params = 512 * 7 * 7 * 4096 + 4096 * 4096 + 4096 * 10  # FC layers for 10 classes
    standard_vgg16_params = conv_params + fc_params
    compression_ratio = standard_vgg16_params / stats["total_params"]
    print(f"\nStandard VGG-16 params (10 classes): {standard_vgg16_params:,}")
    print(f"Compressed model params: {stats['total_params']:,}")
    print(f"Overall compression ratio: {compression_ratio:.2f}x")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VGG-16 MPO on CIFAR-10")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument(
        "--tt-rank-conv", type=int, default=32, help="TT-rank for conv layers"
    )
    parser.add_argument(
        "--tt-rank-fc", type=int, default=64, help="TT-rank for FC layers"
    )
    parser.add_argument(
        "--no-compress-conv", action="store_true", help="disable conv compression"
    )
    parser.add_argument(
        "--no-compress-fc", action="store_true", help="disable FC compression"
    )
    parser.add_argument(
        "--batch-norm", action="store_true", help="use batch normalization"
    )
    parser.add_argument("--no-cuda", action="store_true", help="disable CUDA")
    parser.add_argument("--seed", type=int, default=1, help="random seed")

    args = parser.parse_args()

    # Setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # Data loading
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # Model
    print("Creating VGG-16 MPO with:")
    print(f"  Conv TT-rank: {args.tt_rank_conv}")
    print(f"  FC TT-rank: {args.tt_rank_fc}")
    print(f"  Compress conv: {not args.no_compress_conv}")
    print(f"  Compress FC: {not args.no_compress_fc}")
    print(f"  Batch norm: {args.batch_norm}")

    model = VGG16_MPO(
        num_classes=10,  # CIFAR-10 has 10 classes
        tt_ranks_conv=args.tt_rank_conv,
        tt_ranks_fc=args.tt_rank_fc,
        compress_conv=not args.no_compress_conv,
        compress_fc=not args.no_compress_fc,
        batch_norm=args.batch_norm,
    ).to(device)

    # Print compression statistics
    print_compression_stats(model)

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[8, 14, 18], gamma=0.5
    )

    # Training
    print("\nStarting training...")
    start_time = time.time()
    best_accuracy = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}")
        train(model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader)
        scheduler.step()

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), "vgg16_mpo_cifar10_best.pth")

    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")
    print(f"Best test accuracy: {best_accuracy:.2f}%")

    # Final model analysis
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Memory usage estimate
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size = (param_size + buffer_size) / 1024 / 1024

    print(f"  Model size: {model_size:.2f} MB")


if __name__ == "__main__":
    main()
