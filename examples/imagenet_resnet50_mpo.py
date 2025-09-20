"""ImageNet training with ResNet-50 MPO."""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch_mpo.models import resnet50_mpo


def main():
    parser = argparse.ArgumentParser(description="ResNet-50 MPO ImageNet Training")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training"
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

    # MPO specific arguments
    parser.add_argument(
        "--tt-rank-conv", default=16, type=int, help="TT-rank for convolutional layers"
    )
    parser.add_argument(
        "--tt-rank-fc", default=32, type=int, help="TT-rank for fully connected layers"
    )
    parser.add_argument(
        "--no-mpo-conv", action="store_true", help="disable MPO for conv layers"
    )
    parser.add_argument(
        "--no-mpo-fc", action="store_true", help="disable MPO for FC layers"
    )

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    # Create model
    print("=> creating ResNet-50 MPO model")
    print(f"   Conv TT-rank: {args.tt_rank_conv}")
    print(f"   FC TT-rank: {args.tt_rank_fc}")
    print(f"   Use MPO conv: {not args.no_mpo_conv}")
    print(f"   Use MPO FC: {not args.no_mpo_fc}")

    model = resnet50_mpo(
        num_classes=1000,
        use_mpo_conv=not args.no_mpo_conv,
        use_mpo_fc=not args.no_mpo_fc,
        tt_ranks_conv=args.tt_rank_conv,
        tt_ranks_fc=args.tt_rank_fc,
    )

    # Print compression statistics
    print_compression_stats(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # Data loading
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # Training loop
    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # Evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # Remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "resnet50_mpo",
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )


def train(train_loader, model, criterion, optimizer, epoch, args):
    """Train for one epoch."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # Compute output
        output = model(images)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    """Validate on test set."""
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # Compute output
            output = model(images)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(f" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")

    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """Save checkpoint."""
    torch.save(state, filename)
    if is_best:
        import shutil

        shutil.copyfile(filename, "model_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs."""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def print_compression_stats(model):
    """Print detailed compression statistics."""
    stats = model.compression_stats()

    print("\n" + "=" * 60)
    print("ResNet-50 MPO Compression Statistics")
    print("=" * 60)

    print(f"\nTotal parameters: {stats['total_params']:,}")

    if stats["conv_compression"]:
        print(f"\nCompressed convolutional layers: {len(stats['conv_compression'])}")
        total_conv_compression = sum(
            layer["compression_ratio"] for layer in stats["conv_compression"]
        ) / len(stats["conv_compression"])
        print(f"Average conv compression: {total_conv_compression:.2f}x")

    if stats["fc_compression"]:
        print(f"\nCompressed FC layers: {len(stats['fc_compression'])}")
        for layer in stats["fc_compression"]:
            print(f"  {layer['name']}: {layer['compression_ratio']:.2f}x compression")

    # Compare with standard ResNet-50
    standard_resnet50_params = 25557032  # Standard ResNet-50 parameters
    compression_ratio = standard_resnet50_params / stats["total_params"]
    print(f"\nOverall compression vs standard ResNet-50: {compression_ratio:.2f}x")
    print("=" * 60 + "\n")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # Ensure float for consistent accumulation/formatting
        if hasattr(val, "item"):
            val = float(val.item())
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Progress display utility."""

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
