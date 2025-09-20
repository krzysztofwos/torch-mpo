# Matrix Product Operators and Tensor-Train Decomposition: A Comprehensive Tutorial

## Table of Contents

1. [Introduction](#introduction)
2. [The Curse of Dimensionality](#the-curse-of-dimensionality)
3. [Tensor Decomposition Basics](#tensor-decomposition-basics)
4. [Tensor-Train (TT) Decomposition](#tensor-train-tt-decomposition)
5. [Matrix Product Operators (MPO)](#matrix-product-operators-mpo)
6. [Application to Neural Networks](#application-to-neural-networks)
7. [Implementation Details](#implementation-details)
8. [Practical Considerations](#practical-considerations)
9. [Code Examples](#code-examples)
10. [Further Reading](#further-reading)

## Introduction

Matrix Product Operators (MPO) and Tensor-Train (TT) decomposition are powerful mathematical techniques for representing high-dimensional tensors efficiently. In the context of deep learning, these methods can dramatically reduce the number of parameters in neural networks while maintaining model performance.

### Motivation

Consider a fully connected layer with 1024 input neurons and 1024 output neurons. This configuration requires a weight matrix containing 1,048,576 parameters. Through TT decomposition, the same transformation can be represented using approximately 40,000 parameters, achieving a compression ratio of 25:1.

## The Curse of Dimensionality

### The Problem

Deep neural networks suffer from the "curse of dimensionality" — the number of parameters grows exponentially with the size of the network. For a weight tensor of order $d$ with each dimension of size $n$, we need $n^d$ parameters. For example, a 4-dimensional tensor with each dimension of size 10 requires $10^4 = 10,000$ parameters.

### The Solution: Low-Rank Structure

Many real-world tensors exhibit low-rank structure, allowing accurate approximation through products of smaller tensors. This fundamental property underlies tensor decomposition methods.

## Tensor Decomposition Basics

### What is a Tensor?

A tensor is a multi-dimensional array:

- Order 0: Scalar (single number)
- Order 1: Vector (1D array)
- Order 2: Matrix (2D array)
- Order 3+: Higher-order tensor

### Types of Tensor Decomposition

1. **CP Decomposition**: Sum of rank-1 tensors
2. **Tucker Decomposition**: Core tensor with factor matrices
3. **Tensor-Train (TT)**: Chain of 3D tensors
4. **Matrix Product States (MPS)**: Quantum-inspired, similar to TT

## Tensor-Train (TT) Decomposition

### Mathematical Definition

A $d$-dimensional tensor $\mathcal{X}$ of shape $(n_1, n_2, \ldots, n_d)$ can be decomposed into a chain of 3-way tensors (cores):

$$
\mathcal{X}[i_1, i_2, \ldots, i_d] = \sum_{r_0, r_1, \ldots, r_d} \mathcal{G}_1[r_0, i_1, r_1] \times \mathcal{G}_2[r_1, i_2, r_2] \times \cdots \times \mathcal{G}_d[r_{d-1}, i_d, r_d]
$$

Where:

- $\mathcal{G}_k$ are the TT-cores
- $r_k$ are the TT-ranks ($r_0 = r_d = 1$)
- The summation is over all internal indices

### Visual Representation

```
Original tensor X:
[n₁ × n₂ × n₃ × n₄]

TT decomposition:
G₁ -- G₂ -- G₃ -- G₄
|     |     |     |
n₁    n₂    n₃    n₄
```

### Compression Ratio

Original parameters: $n_1 \times n_2 \times \cdots \times n_d$
TT parameters: $\sum_{i=1}^{d} (r_{i-1} \times n_i \times r_i)$

For a 4D tensor with $n=256$ and ranks $r=16$:

- Original: $256^4 = 4,294,967,296$ parameters
- TT format: approximately 65,000 parameters
- Compression ratio: 66,000:1

## Matrix Product Operators (MPO)

### MPO for Matrices

For neural networks, we often work with 2D weight matrices. To apply TT decomposition:

1. **Reshape**: Matrix $W \in \mathbb{R}^{m \times n} \rightarrow$ Tensor $W \in \mathbb{R}^{m_1 \times m_2 \times \cdots \times m_d \times n_1 \times n_2 \times \cdots \times n_d}$
2. **Decompose**: Apply TT decomposition to the reshaped tensor
3. **Contract**: Perform efficient multiplication using the TT structure

### Example: Factorizing Dimensions

Consider an original matrix of size $1024 \times 512$:

**Dimension factorization:**

- $1024 = 4 \times 16 \times 16$
- $512 = 8 \times 8 \times 8$

**TT-cores shapes with rank $r=8$:**

- $\mathcal{G}_1$: $[1 \times 4, 8 \times 8] = 256$ parameters
- $\mathcal{G}_2$: $[8 \times 16, 8 \times 8] = 8,192$ parameters
- $\mathcal{G}_3$: $[8 \times 16, 8 \times 1] = 1,024$ parameters

**Total:** 9,472 parameters vs 524,288 original (compression ratio ~55:1)

## Application to Neural Networks

### 1. Fully Connected Layers

Replace weight matrix $W$ with TT-cores:

```python
# Standard linear layer
y = Wx + b

# TT linear layer
y = TT_multiply(G_1, G_2, ..., G_d, x) + b
```

### 2. Convolutional Layers

For Conv2D with kernel $K \in \mathbb{R}^{h \times w \times c_{in} \times c_{out}}$:

- Keep spatial dimensions $(h, w)$ intact
- Decompose channel dimensions $(c_{in}, c_{out})$

### 3. Benefits

- **Memory Efficiency**: Reduction of parameters by factors of 10-100
- **Computational Efficiency**: Improved inference speed with optimized implementations
- **Regularization**: Inherent low-rank constraints provide regularization against overfitting

## Implementation Details

### TT-SVD Algorithm

The key algorithm for computing TT decomposition:

```python
def matrix_tt_svd(matrix, inp_modes, out_modes, ranks, epsilon=1e-10):
    """TT-SVD algorithm for matrix decomposition."""
    cores = []
    C = matrix

    # Reshape matrix into tensor
    tensor_shape = inp_modes + out_modes
    C = C.reshape(tensor_shape)

    # Move output modes to the right
    C = C.permute(list(range(len(inp_modes))) +
                  list(range(len(tensor_shape), len(tensor_shape) + len(out_modes))))

    for i in range(len(inp_modes) - 1):
        # Reshape for SVD
        C = C.reshape(ranks[i] * inp_modes[i], -1)

        # Compute SVD
        U, S, V = torch.svd(C)

        # Truncate to rank
        rank = min(ranks[i+1], U.shape[1])
        U = U[:, :rank]
        S = S[:rank]
        V = V[:, :rank]

        # Store core with output mode
        core_shape = (ranks[i], inp_modes[i], out_modes[i], ranks[i+1])
        cores.append(U.reshape(ranks[i], inp_modes[i], rank))

        # Update remainder
        C = torch.diag(S) @ V.T

    # Last core
    cores.append(C.reshape(ranks[-2], inp_modes[-1], out_modes[-1], ranks[-1]))

    return cores
```

### Forward Pass

Efficient computation through sequential contractions:

```python
def tt_forward(cores, x):
    """Forward propagation through TT-decomposed layer."""
    # Reshape input tensor to match mode dimensions
    x = x.reshape(batch_size, *input_modes)

    # Sequential contraction with TT cores
    for i, core in enumerate(cores):
        x = contract(x, core, mode=i)

    # Reshape to output dimensions
    return x.reshape(batch_size, output_dim)
```

## Practical Considerations

### 1. Rank Selection

- **Higher ranks**: Provide better approximation accuracy at the cost of increased parameters
- **Lower ranks**: Achieve greater compression with potential accuracy reduction
- **Practical guideline**: Initial rank values of 8-16 typically provide a reasonable starting point for optimization

### 2. Dimension Factorization

The selection of dimension factorization significantly impacts performance:

- **Balanced factorization**: 1024 = 32 × 32 (preferred)
- **Unbalanced factorization**: 1024 = 2 × 512 (suboptimal)
- **Number of modes**: Increasing modes improves compression but may complicate optimization

### 3. Initialization Strategies

- **Random initialization**: Standard methods such as Xavier or Kaiming initialization
- **Pretrained initialization**: Application of TT-SVD to existing weights through the `from_matrix()` method
- **Scale considerations**: Appropriate scaling is essential for training stability

#### Deep Dive: Initialization Challenges in TT Decomposition

One of the most critical aspects of implementing TT-decomposed layers is proper initialization. This is especially important for `TTConv2d` layers.

**The Problem:**
In `TTConv2d`, the computation flow is:

1. Spatial convolution: `input [B, C_in, H, W] → [B, rank_1, H', W']`
2. TT cores chain: `rank_1 → C_out` through multiple matrix multiplications

If each operation maintains unit variance, the final variance increases by a factor of approximately $d$ (where $d$ represents the number of TT cores), potentially causing activation explosion in deep networks.

**The Solution:**

- **TTLinear**: Standard initialization works well as cores directly process reshaped input
- **TTConv2d**: Scale TT cores by $1/d^{0.25}$ (fourth root of number of cores)
  - The choice of $d^{0.25}$ over $d^{0.5}$ is based on empirical observations demonstrating superior gradient flow characteristics
  - Spatial convolution uses standard Kaiming initialization
  - TT cores use scaled initialization to compensate

**Implementation Details:**

```python
# In TTConv2d initialization
d = len(self.cores)
scale_factor = 1.0 / d**0.25

# Apply to each core's initialization
std = math.sqrt(2.0 / fan_out) * scale_factor
```

**Consequences of Improper Initialization:**

- Without appropriate scaling: ResNet-50 outputs may reach magnitudes of $10^{18}$
- With proper scaling: Output magnitudes remain stable (order of $10^2$) throughout deep networks
- This consideration is particularly critical when training from scratch, while pretrained compression demonstrates greater robustness

### 4. Training Considerations

- **Learning rate adjustment**: Different parameterization may necessitate modified learning rates
- **Regularization effects**: The TT structure inherently provides regularization
- **Fine-tuning strategy**: Initialization from pretrained weights followed by compression

## Code Examples

Complete working examples are available in the `examples/` directory:

- `compress_vgg.py`: Compression of pretrained VGG models
- `cifar10_vgg16_mpo.py`: Training VGG-16 MPO on CIFAR-10 from scratch
- `mnist_lenet5_mpo.py`: MNIST classification using TT-decomposed layers

### Example 1: Basic TT Layer

```python
import torch
from torch_mpo import TTLinear

# Create TT linear layer
layer = TTLinear(
    in_features=1024,
    out_features=512,
    tt_ranks=8  # Compression parameter
)

# Use like standard linear layer
x = torch.randn(32, 1024)
y = layer(x)  # Output shape: [32, 512]

print(f"Compression ratio: {layer.compression_ratio():.2f}:1")
```

### Example 2: Compressing Existing Model

```python
from torch_mpo import compress_model
import torchvision.models as models

# Load pretrained model
model = models.resnet18(weights='IMAGENET1K_V1')  # or weights=None

# Compress with target 10x compression
compressed = compress_model(
    model=model,
    compression_ratio=0.1,  # 10x compression
    verbose=True
)

# Fine-tune compressed model
optimizer = torch.optim.Adam(compressed.parameters(), lr=1e-4)
# ... training loop ...
```

### Example 3: Custom Mode Selection

```python
# Optimal factorization for specific dimensions
layer = TTLinear(
    in_features=784,  # 28×28 MNIST
    out_features=256,  # 16×16
    inp_modes=[7, 4, 7, 4],  # 7×4×7×4 = 784
    out_modes=[4, 4, 4, 4],  # 4×4×4×4 = 256
    tt_ranks=[1, 8, 8, 8, 1]
)
```

### Example 4: Analyzing Compression

```python
def analyze_tt_compression(in_dim, out_dim, rank):
    """Analyze compression ratios for various TT configurations."""
    from torch_mpo import TTLinear

    original_params = in_dim * out_dim

    # Create a TTLinear layer to use auto-factorization
    layer = TTLinear(in_dim, out_dim, tt_ranks=rank)
    in_modes = layer.inp_modes
    out_modes = layer.out_modes

    # Calculate TT parameters
    d = len(in_modes)
    tt_params = 0
    for i in range(d):
        tt_params += layer.tt_ranks[i] * in_modes[i] * out_modes[i] * layer.tt_ranks[i+1]

    compression = original_params / tt_params

    print(f"Dimensions: {in_dim} × {out_dim}")
    print(f"Factorization: {in_modes} × {out_modes}")
    print(f"Original params: {original_params:,}")
    print(f"TT params (rank={rank}): {tt_params:,}")
    print(f"Compression ratio: {compression:.2f}:1")

# Example applications
analyze_tt_compression(1024, 1024, rank=8)
analyze_tt_compression(4096, 1000, rank=16)
```

## Performance Analysis

### Memory Footprint

| Layer Type                    | Parameters | Memory (FP32) | Memory (INT8) |
| ----------------------------- | ---------- | ------------- | ------------- |
| Linear(4096, 4096)            | 16.8M      | 64 MB         | 16 MB         |
| TTLinear(4096, 4096, rank=16) | 1.0M       | 4 MB          | 1 MB          |
| Compression                   | 16.8x      | 16x           | 16x           |

### Computational Complexity

- **Standard Linear**: $O(mn)$ for $m \times n$ matrix
- **TT Linear**: $O(dr^2 n^{1/d})$ where $d$ = modes, $r$ = rank
- **Speedup**: Depends on implementation and hardware

### Accuracy vs Compression Trade-off

**Empirical Results: VGG-16 on CIFAR-10**

| TT-Rank (Conv/FC) | Compression | Test Accuracy | Observations                                        |
| ----------------- | ----------- | ------------- | --------------------------------------------------- |
| 8/16              | ~319x       | ~10%          | Excessive compression prevents effective training   |
| 32/64             | ~60x        | ~58%          | Balanced trade-off between compression and accuracy |
| 64/128            | ~15x        | ~70%          | Higher accuracy with moderate compression           |
| Original          | 1x          | ~93%          | Baseline VGG-16 performance                         |

## Common Challenges and Mitigation Strategies

### 1. Gradient Instability

**Challenge**: Sequential matrix multiplications may lead to gradient vanishing or explosion

**Mitigation**: Proper initialization schemes, gradient clipping, and normalization techniques

### 2. Suboptimal Factorization

**Challenge**: Unbalanced mode dimensions result in inefficient compression

**Mitigation**: Implementation of balanced factorization schemes and exploration of alternative mode configurations

### 3. Rank Determination

**Challenge**: Insufficient rank values lead to accuracy degradation

**Mitigation**: Begin with higher rank values and systematically reduce while monitoring performance metrics

## Current Limitations

### TTConv2d Layer Limitations

The current implementation of TTConv2d has several limitations to be aware of:

1. **Grouped/Depthwise Convolutions**: Not supported. Layers with `groups > 1` will raise a `ValueError` or be skipped during automatic compression.

2. **Spatial Decomposition**: The `decompose_spatial=True` option is not yet implemented. The current implementation uses a spatial convolution followed by TT channel mixing.

3. **Padding='same'**: Only supported when it equals symmetric padding for `stride=1`. Other configurations are automatically skipped during compression.

4. **Approximation Quality**: The `from_conv_weight()` method for initializing from pretrained weights uses `matrix_tt_svd` which adds an additional layer of TT decomposition on top of SVD. This limits approximation quality even at high ranks. Future versions may improve this.

### Compression Trade-offs

When using TT decomposition, keep in mind:

- **Rank Selection**: The spatial projection rank (`tt_ranks[1]` in TTConv2d) is often the primary bottleneck for approximation quality
- **Auto-factorization**: The automatic factorization of dimensions may not always be optimal for your specific use case
- **Fine-tuning**: Pretrained weight initialization typically requires fine-tuning to recover accuracy

### Roadmap

Future releases are planned to address:

- Support for grouped and depthwise convolutions
- Full spatial decomposition for convolutional layers
- Enhanced padding support for all stride configurations
- Improved approximation quality for pretrained weight initialization

## Conclusion

Matrix Product Operators and Tensor-Train decomposition provide an effective framework for neural network compression. Through exploitation of low-rank tensor structure, substantial parameter reduction can be achieved while preserving model performance. Successful application requires understanding of the inherent trade-offs and systematic optimization of decomposition parameters for specific applications.

## Further Reading

1. **Original TT Paper**: "Tensor-Train Decomposition" by Oseledets (2011)
2. **MPO for DNNs**: "Compressing deep neural networks by matrix product operators" by Gao et al. (2020)
3. "Tensorizing Neural Networks" by Novikov et al. (2015)
4. "Ultimate tensorization: compressing convolutional and FC layers alike" by Garipov et al. (2016)
