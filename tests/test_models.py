"""Tests for public model constructors and wrappers."""

import pytest
import torch
import torch.nn as nn

import torch_mpo.models.resnet_mpo as resnet_module
import torch_mpo.models.vgg_mpo as vgg_module
from torch_mpo.layers import TTConv2d, TTLinear
from torch_mpo.models import (
    VGG16_MPO,
    VGG19_MPO,
    resnet18_mpo,
    resnet34_mpo,
    resnet50_mpo,
    resnet101_mpo,
    resnet152_mpo,
    vgg16_mpo,
    vgg19_mpo,
)
from torch_mpo.models.resnet_mpo import BasicBlock, Bottleneck
from torch_mpo.models.vgg_mpo import VGG_MPO, make_layers


def test_make_layers_selects_ttconv_for_large_layers():
    """Test VGG feature construction with and without MPO convs."""
    features = make_layers([64, 128, "M"], batch_norm=True, compress_conv=True)

    assert isinstance(features[0], nn.Conv2d)
    assert isinstance(features[1], nn.BatchNorm2d)
    assert isinstance(features[3], TTConv2d)
    assert isinstance(features[-1], nn.MaxPool2d)


def test_vgg16_mpo_forward_and_compression_stats():
    """Test VGG-16 MPO forward pass and stats."""
    model = vgg16_mpo(num_classes=10, compress_conv=False, compress_fc=True)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))

    stats = model.compression_stats()
    assert isinstance(model, VGG_MPO)
    assert output.shape == (1, 10)
    assert stats["total_params"] == sum(p.numel() for p in model.parameters())
    assert stats["fc_compression"]
    assert not stats["conv_compression"]


def test_vgg19_factory_builds_model():
    """Test the explicit VGG-19 factory."""
    model = VGG19_MPO(num_classes=5, compress_conv=False, compress_fc=True)
    assert isinstance(model, VGG_MPO)
    assert isinstance(model.classifier[0], TTLinear)


def test_resnet18_mpo_forward_and_compression_stats():
    """Test ResNet-18 MPO forward pass and stats."""
    model = resnet18_mpo(num_classes=10, use_mpo_conv=True, use_mpo_fc=True)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))

    stats = model.compression_stats()
    assert output.shape == (1, 10)
    assert isinstance(model.fc, TTLinear)
    assert stats["fc_compression"]
    assert stats["conv_compression"]


def test_resnet50_uses_bottleneck_blocks():
    """Test ResNet-50 constructor wiring."""
    model = resnet50_mpo(num_classes=7, use_mpo_conv=False, use_mpo_fc=True)
    assert isinstance(model.layer1[0], Bottleneck)
    assert isinstance(model.fc, TTLinear)


def test_resnet18_pretrained_warns(capsys):
    """Test the ResNet pretrained warning path."""
    model = resnet18_mpo(
        pretrained=True,
        num_classes=10,
        use_mpo_conv=False,
        use_mpo_fc=False,
    )
    captured = capsys.readouterr()

    assert (
        "Warning: Pretrained weights for resnet18 MPO not implemented" in captured.out
    )
    assert model.fc.out_features == 10


@torch.no_grad()
def test_vgg16_pretrained_wrapper_loads_state_dict(monkeypatch, capsys):
    """Test the vgg16 pretrained wrapper without downloading weights."""

    class DummyMPO:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state_dict, strict=False):
            self.loaded = (state_dict, strict)

    class DummyStandard:
        def state_dict(self):
            return {"dummy": torch.tensor(1.0)}

    mpo_model = DummyMPO()
    monkeypatch.setattr(vgg_module, "VGG16_MPO", lambda **kwargs: mpo_model)

    import torchvision.models as tv_models

    monkeypatch.setattr(tv_models, "vgg16", lambda pretrained=True: DummyStandard())

    model = vgg16_mpo(pretrained=True, num_classes=10)
    captured = capsys.readouterr()

    assert model is mpo_model
    assert mpo_model.loaded is not None
    state_dict, strict = mpo_model.loaded
    assert strict is False
    assert torch.equal(state_dict["dummy"], torch.tensor(1.0))
    assert "Fine-tuning recommended." in captured.out


@torch.no_grad()
def test_vgg19_pretrained_wrapper_loads_state_dict(monkeypatch, capsys):
    """Test the vgg19 pretrained wrapper without downloading weights."""

    class DummyMPO:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state_dict, strict=False):
            self.loaded = (state_dict, strict)

    class DummyStandard:
        def state_dict(self):
            return {"dummy": torch.tensor(2.0)}

    mpo_model = DummyMPO()
    monkeypatch.setattr(vgg_module, "VGG19_MPO", lambda **kwargs: mpo_model)

    import torchvision.models as tv_models

    monkeypatch.setattr(tv_models, "vgg19", lambda pretrained=True: DummyStandard())

    model = vgg19_mpo(pretrained=True, num_classes=10)
    captured = capsys.readouterr()

    assert model is mpo_model
    assert mpo_model.loaded is not None
    state_dict, strict = mpo_model.loaded
    assert strict is False
    assert torch.equal(state_dict["dummy"], torch.tensor(2.0))
    assert "Fine-tuning recommended." in captured.out


def test_basicblock_validates_unsupported_arguments():
    """Test BasicBlock validation branches."""
    with pytest.raises(ValueError, match="groups=1 and base_width=64"):
        BasicBlock(64, 64, groups=2)

    with pytest.raises(NotImplementedError, match="Dilation > 1 not supported"):
        BasicBlock(64, 64, dilation=2)


def test_resnet_constructor_wrapper_arguments(monkeypatch):
    """Test ResNet wrapper layer mappings without constructing all variants."""
    calls = []

    def fake_resnet(arch, block, layers, pretrained, progress, **kwargs):
        calls.append((arch, block, layers, pretrained, progress, kwargs))
        return arch

    monkeypatch.setattr(resnet_module, "_resnet", fake_resnet)

    assert resnet18_mpo(num_classes=3, progress=False) == "resnet18"
    assert resnet34_mpo(num_classes=3, progress=False) == "resnet34"
    assert resnet50_mpo(num_classes=3, progress=False) == "resnet50"
    assert resnet101_mpo(num_classes=3, progress=False) == "resnet101"
    assert resnet152_mpo(num_classes=3, progress=False) == "resnet152"

    expected = [
        ("resnet18", BasicBlock, [2, 2, 2, 2]),
        ("resnet34", BasicBlock, [3, 4, 6, 3]),
        ("resnet50", Bottleneck, [3, 4, 6, 3]),
        ("resnet101", Bottleneck, [3, 4, 23, 3]),
        ("resnet152", Bottleneck, [3, 8, 36, 3]),
    ]

    assert [(arch, block, layers) for arch, block, layers, *_ in calls] == expected


def test_vgg_factory_constructors_return_models():
    """Test explicit VGG factory constructors."""
    assert isinstance(
        VGG16_MPO(num_classes=3, compress_conv=False, compress_fc=True), VGG_MPO
    )
    assert isinstance(
        VGG19_MPO(num_classes=3, compress_conv=False, compress_fc=True), VGG_MPO
    )
