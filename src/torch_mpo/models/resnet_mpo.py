"""ResNet models with MPO-compressed layers."""

from typing import Any

import torch
import torch.nn as nn

from torch_mpo.layers import TTConv2d, TTLinear


class BasicBlock(nn.Module):
    """Basic building block for ResNet-18/34 with optional MPO compression."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: type[nn.Module] | None = None,
        use_mpo: bool = True,
        tt_ranks: int | list[int] = 8,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # First convolution
        self.conv1: TTConv2d | nn.Conv2d
        if use_mpo and in_planes >= 64 and planes >= 64:
            self.conv1 = TTConv2d(
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                tt_ranks=tt_ranks,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
        self.bn1 = norm_layer(planes)

        # Second convolution
        self.conv2: TTConv2d | nn.Conv2d
        if use_mpo and planes >= 64:
            self.conv2 = TTConv2d(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                tt_ranks=tt_ranks,
            )
        else:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152 with optional MPO compression."""

    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: type[nn.Module] | None = None,
        use_mpo: bool = True,
        tt_ranks: int | list[int] = 8,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        # 1x1 convolution
        self.conv1: TTConv2d | nn.Conv2d
        if use_mpo and in_planes >= 128 and width >= 128:
            self.conv1 = TTConv2d(
                in_planes, width, kernel_size=1, bias=False, tt_ranks=tt_ranks
            )
        else:
            self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        # 3x3 convolution
        # Don't use MPO for grouped/dilated convolutions
        self.conv2: TTConv2d | nn.Conv2d
        if use_mpo and width >= 128 and groups == 1 and dilation == 1:
            self.conv2 = TTConv2d(
                width,
                width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=groups,
                dilation=dilation,
                bias=False,
                tt_ranks=tt_ranks,
            )
        else:
            self.conv2 = nn.Conv2d(
                width,
                width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=groups,
                dilation=dilation,
                bias=False,
            )
        self.bn2 = norm_layer(width)

        # 1x1 convolution
        self.conv3: TTConv2d | nn.Conv2d
        if use_mpo and width >= 128 and planes * self.expansion >= 128:
            self.conv3 = TTConv2d(
                width,
                planes * self.expansion,
                kernel_size=1,
                bias=False,
                tt_ranks=tt_ranks,
            )
        else:
            self.conv3 = nn.Conv2d(
                width, planes * self.expansion, kernel_size=1, bias=False
            )
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_MPO(nn.Module):
    """ResNet with optional MPO compression."""

    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: type[nn.Module] | None = None,
        use_mpo_conv: bool = True,
        use_mpo_fc: bool = True,
        tt_ranks_conv: int | list[int] = 8,
        tt_ranks_fc: int | list[int] = 16,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group
        self.use_mpo_conv = use_mpo_conv
        self.tt_ranks_conv = tt_ranks_conv

        # Initial convolution - don't compress (small)
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final classifier
        self.fc: TTLinear | nn.Linear
        if use_mpo_fc and 512 * block.expansion >= 256:
            self.fc = TTLinear(512 * block.expansion, num_classes, tt_ranks=tt_ranks_fc)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, TTConv2d):
                # TTConv2d initializes itself in __init__
                pass
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            # Downsample - could use MPO here too
            if (
                self.use_mpo_conv
                and self.inplanes >= 128
                and planes * block.expansion >= 128
            ):
                downsample = nn.Sequential(
                    TTConv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        tt_ranks=self.tt_ranks_conv,
                    ),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                use_mpo=self.use_mpo_conv,
                tt_ranks=self.tt_ranks_conv,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    use_mpo=self.use_mpo_conv,
                    tt_ranks=self.tt_ranks_conv,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def compression_stats(self) -> dict:
        """Calculate compression statistics."""
        stats: dict[str, Any] = {
            "total_params": sum(p.numel() for p in self.parameters()),
            "conv_compression": [],
            "fc_compression": [],
        }

        # Conv layer stats
        for name, module in self.named_modules():
            if isinstance(module, TTConv2d):
                stats["conv_compression"].append(
                    {
                        "name": name,
                        "compression_ratio": module.compression_ratio(),
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                    }
                )

        # FC layer stats
        if isinstance(self.fc, TTLinear):
            stats["fc_compression"].append(
                {
                    "name": "fc",
                    "compression_ratio": self.fc.compression_ratio(),
                    "in_features": self.fc.in_features,
                    "out_features": self.fc.out_features,
                }
            )

        return stats


def _resnet(
    arch: str,
    block: type[BasicBlock | Bottleneck],
    layers: list[int],
    pretrained: bool,
    progress: bool,
    **kwargs,
) -> ResNet_MPO:
    model = ResNet_MPO(block, layers, **kwargs)

    if pretrained:
        # Note: Loading pretrained weights for MPO models requires special handling
        print(
            f"Warning: Pretrained weights for {arch} MPO not implemented. Training from scratch."
        )

    return model


def resnet18_mpo(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> ResNet_MPO:
    """ResNet-18 with MPO compression."""
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34_mpo(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> ResNet_MPO:
    """ResNet-34 with MPO compression."""
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50_mpo(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> ResNet_MPO:
    """ResNet-50 with MPO compression."""
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101_mpo(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> ResNet_MPO:
    """ResNet-101 with MPO compression."""
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152_mpo(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> ResNet_MPO:
    """ResNet-152 with MPO compression."""
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )
