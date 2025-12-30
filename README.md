# PyTorch Matrix Product Operators

[![Tests]][Python Actions] [![Latest Version PyPI]][PyPI]

[Tests]: https://img.shields.io/github/actions/workflow/status/krzysztofwos/torch-mpo/tests.yml?branch=main&label=Tests
[Python Actions]: https://github.com/krzysztofwos/torch-mpo/actions/workflows/tests.yml?query=branch%3Amain
[Latest Version PyPI]: https://img.shields.io/pypi/v/torch-mpo?label=PyPI
[PyPI]: https://pypi.org/project/torch-mpo/

A modern PyTorch implementation of Matrix Product Operators (MPO) for neural network compression, based on the paper "Compressing deep neural networks by matrix product operators" by Ze-Feng Gao et al.

## Overview

This library provides PyTorch implementations of tensor-train decomposed neural network layers that can significantly reduce the number of parameters in deep neural networks while maintaining accuracy.

## Features

- **TT-decomposed layers**: `TTLinear` and `TTConv2d` for compressed fully-connected and convolutional layers
- **Modern PyTorch**: Full compatibility with PyTorch 2.0+, type hints, device-agnostic
- **Pretrained model compression**: Convert existing PyTorch models to MPO format
- **Multiple architectures**: VGG-16/19, ResNet-18/34/50/101/152, and custom models
- **Automatic factorization**: Smart dimension factorization for optimal compression
- **Comprehensive examples**: MNIST, CIFAR-10, ImageNet training scripts
- **Analysis tools**: Compression ratio calculation, performance benchmarks

## Installation

```bash
# Clone the repository
git clone https://github.com/krzysztofwos/torch-mpo
cd torch-mpo

# Install with uv (recommended)
uv sync                  # Install base dependencies
uv sync --all-extras     # Install with all extras (dev, docs)

# Or install with pip (development mode)
pip install -e .
pip install -e ".[dev]"  # With development dependencies
```

## Quick Start

### Basic Usage

```python
import torch
from torch_mpo import TTLinear, TTConv2d

# Create a TT-decomposed linear layer
linear = TTLinear(
    in_features=1024,
    out_features=512,
    tt_ranks=8,  # Higher rank = better accuracy, more parameters
    bias=True
)

# Create a TT-decomposed convolutional layer
conv = TTConv2d(
    in_channels=128,
    out_channels=256,
    kernel_size=3,
    padding=1,
    tt_ranks=8
)

# Use them like standard PyTorch layers
x = torch.randn(32, 1024)
y = linear(x)  # Shape: [32, 512]

x = torch.randn(32, 128, 32, 32)
y = conv(x)  # Shape: [32, 256, 32, 32]
```

### Compress Existing Models

```python
from torch_mpo import compress_model
import torchvision.models as models

# Load a pretrained model
model = models.vgg16(pretrained=True)

# Compress it with MPO
compressed_model = compress_model(
    model,
    compression_ratio=0.1,  # Target 10x compression
    compress_linear=True,   # Compress Linear layers
    compress_conv=True,     # Compress Conv2d layers
    verbose=True
)

# Fine-tune the compressed model
optimizer = torch.optim.Adam(compressed_model.parameters(), lr=1e-4)
# ... continue with training
```

### Use Pre-built Architectures

```python
from torch_mpo.models import vgg16_mpo, resnet50_mpo

# VGG-16 with MPO compression
model = vgg16_mpo(
    num_classes=10,
    tt_ranks_conv=8,   # TT-rank for conv layers
    tt_ranks_fc=16,    # TT-rank for FC layers
    compress_conv=True,
    compress_fc=True
)

# ResNet-50 with MPO compression
model = resnet50_mpo(
    num_classes=1000,
    tt_ranks_conv=16,
    tt_ranks_fc=32,
    use_mpo_conv=True,
    use_mpo_fc=True
)
```

## Examples

The `examples/` directory contains complete training scripts:

### MNIST with LeNet-5 MPO

```bash
python examples/mnist_lenet5_mpo.py --tt-rank 8 --epochs 10
```

### CIFAR-10 with VGG-16 MPO

```bash
python examples/cifar10_vgg16_mpo.py --tt-rank-conv 8 --tt-rank-fc 16 --epochs 20
```

### ImageNet with ResNet-50 MPO

```bash
python examples/imagenet_resnet50_mpo.py /path/to/imagenet \
    --tt-rank-conv 16 --tt-rank-fc 32 --epochs 90
```

### Compress Pretrained VGG

```bash
python examples/compress_vgg.py --model vgg16 --compression-ratio 0.1
```

## Performance Benchmarks

Run benchmarks to compare MPO layers with standard layers:

```bash
python benchmarks/benchmark_layers.py
```

### Typical Results

| Layer                  | Original Params | MPO Params (rank=8) | Compression | Speedup |
| ---------------------- | --------------- | ------------------- | ----------- | ------- |
| Linear(4096, 4096)     | 16.8M           | 655K                | 25.6x       | 0.8x    |
| Conv2d(256, 512, 3)    | 1.2M            | 123K                | 9.7x        | 0.9x    |
| VGG-16 (full model)    | 138M            | 15M                 | 9.2x        | 0.85x   |
| ResNet-50 (full model) | 25.6M           | 8.2M                | 3.1x        | 0.95x   |

## Documentation

See the comprehensive tutorial in `docs/mpo_tutorial.md` covering:

- Mathematical foundations of TT decomposition
- How MPO compression works
- Implementation details
- Best practices and tips
- Advanced topics

## Limitations

### Current Limitations

- **Grouped/Depthwise Convolutions**: Not supported. Layers with `groups > 1` are automatically skipped during compression.
- **Spatial Decomposition**: The `decompose_spatial=True` option for TTConv2d is not yet implemented.
- **Padding='same'**: Only supported when it equals symmetric padding for `stride=1`. Other cases are skipped.

### Roadmap

Future releases will address:

- Support for grouped and depthwise convolutions
- Full spatial decomposition for convolutional layers
- Enhanced padding support for all stride configurations

## Key Concepts

### TT-Ranks

The `tt_ranks` parameter controls the trade-off between compression and accuracy:

- **Lower ranks** (4-8): High compression, some accuracy loss
- **Medium ranks** (8-16): Good balance
- **Higher ranks** (16-32): Less compression, minimal accuracy loss

### Automatic Factorization

The library automatically factorizes dimensions for optimal compression:

```python
# 1024 = 4 × 16 × 16 (automatic factorization)
layer = TTLinear(1024, 512, tt_ranks=8)
```

### Custom Factorization

You can also specify custom factorizations:

```python
layer = TTLinear(
    in_features=784,  # 28×28 MNIST
    out_features=256,
    inp_modes=[7, 4, 7, 4],  # 7×4×7×4 = 784
    out_modes=[4, 4, 4, 4],  # 4×4×4×4 = 256
    tt_ranks=[1, 8, 8, 8, 1]
)
```

### Initialization and Numerical Stability

Proper initialization is crucial for TT-decomposed layers to maintain stable gradients during training:

#### TTLinear Initialization

- Uses standard Xavier/Kaiming initialization for each core
- No additional scaling needed as the decomposition naturally regularizes

#### TTConv2d Initialization

- More complex due to spatial convolution followed by TT cores
- **Key insight**: Variance accumulates through both spatial conv and TT cores
- **Solution**: TT cores are scaled by `1/d^0.25` where `d` is the number of cores
- This empirically maintains output variance similar to standard Conv2d layers

Without proper initialization scaling, deep networks can experience:

- **Exploding activations**: Outputs growing exponentially through layers
- **Vanishing gradients**: Making training impossible
- **Poor convergence**: Model stuck at random performance

The library handles this automatically, but when implementing custom layers, careful attention to initialization is essential.

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite both the original paper and this implementation:

### Original Paper

```bibtex
@article{gao2020compressing,
  title={Compressing deep neural networks by matrix product operators},
  author={Gao, Ze-Feng and Song, Chao and Wang, Lei and others},
  journal={Physical Review Research},
  volume={2},
  number={2},
  pages={023300},
  year={2020}
}
```

### This Implementation

```bibtex
@software{torch-mpo2024,
  title={torch-mpo: PyTorch Matrix Product Operators},
  author={Woś, Krzysztof},
  year={2024},
  url={https://github.com/krzysztofwos/torch-mpo},
  version={0.2.1}
}
```

## License

MIT License
