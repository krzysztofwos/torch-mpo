"""MNIST classification with LeNet-5 using MPO layers."""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch_mpo import TTLinear


class LeNet5MPO(nn.Module):
    """LeNet-5 with MPO-compressed fully connected layers."""

    def __init__(self, tt_rank: int = 8):
        super().__init__()

        # Convolutional layers (standard)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Fully connected layers (MPO-compressed)
        # 16*5*5 = 400 -> factorize as [20, 20]
        # 120 -> factorize as [12, 10]
        # 84 -> factorize as [12, 7]
        self.fc1 = TTLinear(
            in_features=400,
            out_features=120,
            inp_modes=[20, 20],
            out_modes=[12, 10],
            tt_ranks=[1, tt_rank, 1],
            bias=True,
        )

        self.fc2 = TTLinear(
            in_features=120,
            out_features=84,
            inp_modes=[12, 10],
            out_modes=[12, 7],
            tt_ranks=[1, tt_rank, 1],
            bias=True,
        )

        # Output layer (standard - small enough)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Convolutional part
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected part with MPO layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def compression_stats(self):
        """Print compression statistics."""
        print("\nCompression Statistics:")
        print(f"FC1: {self.fc1.compression_ratio():.2f}x compression")
        print(f"FC2: {self.fc2.compression_ratio():.2f}x compression")

        # Total parameters
        total_params = sum(p.numel() for p in self.parameters())

        # Original parameters (if using standard Linear layers)
        original_fc1 = 400 * 120 + 120  # weights + bias
        original_fc2 = 120 * 84 + 84
        compressed_fc1 = sum(p.numel() for p in self.fc1.parameters())
        compressed_fc2 = sum(p.numel() for p in self.fc2.parameters())

        saved_params = (original_fc1 + original_fc2) - (compressed_fc1 + compressed_fc2)

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Parameters saved: {saved_params:,}")
        print(f"Reduction: {saved_params / (original_fc1 + original_fc2) * 100:.1f}%")


def train(model, device, train_loader, optimizer, epoch):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

    accuracy = 100.0 * correct / len(train_loader.dataset)
    avg_loss = train_loss / len(train_loader)
    print(
        f"Train set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)"
    )


def test(model, device, test_loader):
    """Evaluate on test set."""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="LeNet-5 MPO on MNIST")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--tt-rank", type=int, default=8, help="TT-rank for MPO layers")
    parser.add_argument("--no-cuda", action="store_true", help="disable CUDA")
    parser.add_argument("--seed", type=int, default=1, help="random seed")

    args = parser.parse_args()

    # Setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # Data loading
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model
    print(f"Creating LeNet-5 with TT-rank={args.tt_rank}")
    model = LeNet5MPO(tt_rank=args.tt_rank).to(device)
    model.compression_stats()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training
    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        scheduler.step()

    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Final test accuracy: {accuracy:.2f}%")

    # Compare with standard LeNet-5 size
    standard_params = sum(
        p.numel()
        for p in nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.Conv2d(6, 16, 5),
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        ).parameters()
    )

    mpo_params = sum(p.numel() for p in model.parameters())

    print(f"\nStandard LeNet-5 parameters: {standard_params:,}")
    print(f"MPO LeNet-5 parameters: {mpo_params:,}")
    print(f"Overall compression: {standard_params / mpo_params:.2f}x")


if __name__ == "__main__":
    main()
