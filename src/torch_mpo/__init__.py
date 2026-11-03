"""PyTorch Matrix Product Operators for Neural Network Compression."""

from importlib.metadata import PackageNotFoundError, version

from torch_mpo.layers import TTConv2d, TTLinear
from torch_mpo.utils import compress_model

try:
    __version__ = version("torch-mpo")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0+unknown"

__all__ = ["TTLinear", "TTConv2d", "compress_model"]
