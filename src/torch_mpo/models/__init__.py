"""MPO-compressed model architectures."""

from torch_mpo.models.resnet_mpo import (
    resnet18_mpo,
    resnet34_mpo,
    resnet50_mpo,
    resnet101_mpo,
    resnet152_mpo,
)
from torch_mpo.models.vgg_mpo import VGG16_MPO, VGG19_MPO, vgg16_mpo, vgg19_mpo

__all__ = [
    # VGG models
    "VGG16_MPO",
    "VGG19_MPO",
    "vgg16_mpo",
    "vgg19_mpo",
    # ResNet models
    "resnet18_mpo",
    "resnet34_mpo",
    "resnet50_mpo",
    "resnet101_mpo",
    "resnet152_mpo",
]
