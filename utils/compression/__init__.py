"""Weight-sharing quantization, entropy-constrained merging, and storage accounting."""

from .ecsq import DEFAULT_LAMBDA_GRID, build_merge_ladder, dp_merge, nearest_rung
from .huffman_encoding import HuffmanEncode
from .quantization import (
    compress,
    compressible_param_names,
    get_model_params,
    is_compressible_param,
    kmeans_binning,
    set_model_params,
    uniform_binning,
    uniform_bin_edges,
    uniform_quantization,
)
from .storage import compression_ratios, get_compression, size_of_model

__all__ = [
    'DEFAULT_LAMBDA_GRID',
    'build_merge_ladder',
    'dp_merge',
    'nearest_rung',
    'HuffmanEncode',
    'compress',
    'compressible_param_names',
    'get_model_params',
    'is_compressible_param',
    'kmeans_binning',
    'set_model_params',
    'uniform_binning',
    'uniform_bin_edges',
    'uniform_quantization',
    'compression_ratios',
    'get_compression',
    'size_of_model',
]
