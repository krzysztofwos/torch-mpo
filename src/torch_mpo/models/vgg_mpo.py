"""VGG models with MPO-compressed layers."""

from typing import Any

import torch
import torch.nn as nn

from torch_mpo.layers import TTConv2d, TTLinear


class VGG_MPO(nn.Module):
    """VGG model with MPO-compressed layers."""

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
        tt_ranks_conv: int | list[int] = 8,
        tt_ranks_fc: int | list[int] = 16,
        compress_conv: bool = True,
        compress_fc: bool = True,
    ):
        """
        Initialize VGG with MPO layers.

        Args:
            features: Feature extraction layers
            num_classes: Number of output classes
            init_weights: Whether to initialize weights
            dropout: Dropout probability
            tt_ranks_conv: TT-ranks for convolutional layers
            tt_ranks_fc: TT-ranks for fully connected layers
            compress_conv: Whether to use MPO for conv layers
            compress_fc: Whether to use MPO for FC layers
        """
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Classifier with optional MPO compression
        if compress_fc:
            self.classifier = nn.Sequential(
                TTLinear(512 * 7 * 7, 4096, tt_ranks=tt_ranks_fc),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                TTLinear(4096, 4096, tt_ranks=tt_ranks_fc),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, num_classes),  # Keep last layer standard
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, num_classes),
            )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, TTConv2d):
                # TTConv2d has its own initialization in __init__
                # Don't override it
                pass
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, TTLinear):
                # TTLinear has its own initialization in __init__
                # Don't override it
                pass

    def compression_stats(self) -> dict:
        """Calculate compression statistics."""
        stats: dict[str, Any] = {
            "total_params": sum(p.numel() for p in self.parameters()),
            "conv_compression": [],
            "fc_compression": [],
        }

        # Conv layer stats
        for name, module in self.features.named_modules():
            if isinstance(module, TTConv2d):
                stats["conv_compression"].append(
                    {
                        "name": f"features.{name}",
                        "compression_ratio": module.compression_ratio(),
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                    }
                )

        # FC layer stats
        for i, module in enumerate(self.classifier):
            if isinstance(module, TTLinear):
                stats["fc_compression"].append(
                    {
                        "name": f"classifier.{i}",
                        "compression_ratio": module.compression_ratio(),
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                    }
                )

        return stats


def make_layers(
    cfg: list[str | int],
    batch_norm: bool = False,
    tt_ranks: int | list[int] = 8,
    compress_conv: bool = True,
) -> nn.Sequential:
    """
    Create VGG feature layers from configuration.

    Args:
        cfg: Configuration list (numbers are channels, 'M' is maxpool)
        batch_norm: Whether to use batch normalization
        tt_ranks: TT-ranks for MPO layers
        compress_conv: Whether to use MPO for conv layers
    """
    layers: list[nn.Module] = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # v is an int (number of channels)
            assert isinstance(v, int)
            # Decide whether to compress this layer
            # Don't compress first few layers (small channel count)
            use_mpo = compress_conv and in_channels >= 64 and v >= 128

            if use_mpo:
                conv2d: TTConv2d | nn.Conv2d = TTConv2d(
                    in_channels,
                    v,
                    kernel_size=3,
                    padding=1,
                    tt_ranks=tt_ranks,
                )
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    return nn.Sequential(*layers)


# VGG configurations
cfgs: dict[str, list[int | str]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def VGG16_MPO(
    num_classes: int = 1000,
    tt_ranks_conv: int | list[int] = 8,
    tt_ranks_fc: int | list[int] = 16,
    compress_conv: bool = True,
    compress_fc: bool = True,
    batch_norm: bool = False,
    **kwargs,
) -> VGG_MPO:
    """VGG-16 with MPO compression."""
    model = VGG_MPO(
        make_layers(
            cfgs["D"],
            batch_norm=batch_norm,
            tt_ranks=tt_ranks_conv,
            compress_conv=compress_conv,
        ),
        num_classes=num_classes,
        tt_ranks_conv=tt_ranks_conv,
        tt_ranks_fc=tt_ranks_fc,
        compress_conv=compress_conv,
        compress_fc=compress_fc,
        **kwargs,
    )
    return model


def VGG19_MPO(
    num_classes: int = 1000,
    tt_ranks_conv: int | list[int] = 8,
    tt_ranks_fc: int | list[int] = 16,
    compress_conv: bool = True,
    compress_fc: bool = True,
    batch_norm: bool = False,
    **kwargs,
) -> VGG_MPO:
    """VGG-19 with MPO compression."""
    model = VGG_MPO(
        make_layers(
            cfgs["E"],
            batch_norm=batch_norm,
            tt_ranks=tt_ranks_conv,
            compress_conv=compress_conv,
        ),
        num_classes=num_classes,
        tt_ranks_conv=tt_ranks_conv,
        tt_ranks_fc=tt_ranks_fc,
        compress_conv=compress_conv,
        compress_fc=compress_fc,
        **kwargs,
    )
    return model


def vgg16_mpo(pretrained: bool = False, **kwargs) -> VGG_MPO:
    """
    VGG-16 with MPO compression.

    Args:
        pretrained: If True, initializes with standard VGG-16 weights then compresses
        **kwargs: Additional arguments for VGG16_MPO
    """
    model = VGG16_MPO(**kwargs)

    if pretrained:
        # Load standard VGG-16 weights and compress
        import torchvision.models as models

        # Load pretrained standard model
        vgg_standard = models.vgg16(pretrained=True)

        # Copy weights where possible
        # Note: This is a simplified version. Full implementation would
        # properly initialize MPO layers from pretrained weights
        model.load_state_dict(vgg_standard.state_dict(), strict=False)

        print(
            "Warning: Pretrained MPO initialization is simplified. Fine-tuning recommended."
        )

    return model


def vgg19_mpo(pretrained: bool = False, **kwargs) -> VGG_MPO:
    """
    VGG-19 with MPO compression.

    Args:
        pretrained: If True, initializes with standard VGG-19 weights then compresses
        **kwargs: Additional arguments for VGG19_MPO
    """
    model = VGG19_MPO(**kwargs)

    if pretrained:
        # Similar to vgg16_mpo
        import torchvision.models as models

        vgg_standard = models.vgg19(pretrained=True)
        model.load_state_dict(vgg_standard.state_dict(), strict=False)

        print(
            "Warning: Pretrained MPO initialization is simplified. Fine-tuning recommended."
        )

    return model
