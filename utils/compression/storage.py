"""Storage cost and compression ratio (CR) of a weight-shared model.

A quantized tensor is stored as its codebook (K floats of `centroid_bits` each)
plus its index stream. The index stream costs either ceil(log2 K) bits per weight
(fixed-width indices, Eq. 1) or its Huffman-coded length (Eq. 2). Each tensor's
index stream is Huffman-coded separately. Any tensor that was not quantized is
stored as 32-bit floats. The CR is the 32-bit size of all compressible weights
divided by the stored size.
"""

import math

import numpy as np

from .huffman_encoding import HuffmanEncode
from .quantization import compressible_param_names

ORIGINAL_WEIGHT_BITS = 32
DEFAULT_CENTROID_BITS = 32


def _index_stream(module, name):
    centers = getattr(module, f"{name}_centers", None)
    indices = getattr(module, f"{name}_indices", None)
    if centers is None or indices is None:
        return None, None
    return centers, indices.view(-1)


def _huffman_bits(idx, num_centers, weight_bits, fallback_bits):
    try:
        idx_numpy = idx.detach().cpu().numpy().astype(np.int64)
        if idx_numpy.max() >= num_centers or idx_numpy.min() < 0:
            idx_numpy = np.clip(idx_numpy, 0, num_centers - 1)
        bits, _ = HuffmanEncode.encode(idx_numpy, bits=int(weight_bits))
        return bits
    except Exception:
        return fallback_bits


def get_compression(module, name, huffman_encoding=True, centroid_bits=DEFAULT_CENTROID_BITS):
    """(original bits, stored bits) of the tensor ``module.<name>``."""
    param = getattr(module, name, None)
    if param is None:
        return 0, 0
    all_weights = param.numel()

    centers, idx = _index_stream(module, name)
    if centers is None:
        return all_weights * ORIGINAL_WEIGHT_BITS, all_weights * ORIGINAL_WEIGHT_BITS

    num_centers = centers.numel()
    weight_bits = np.ceil(math.log2(num_centers)) if num_centers > 0 else 0
    q_idx_uncompressed = idx.numel() * weight_bits
    if huffman_encoding and len(idx) > 0 and weight_bits > 0:
        q_idx = _huffman_bits(idx, num_centers, weight_bits, q_idx_uncompressed)
    else:
        q_idx = q_idx_uncompressed
    compressed_size = num_centers * centroid_bits + q_idx
    return all_weights * ORIGINAL_WEIGHT_BITS, compressed_size


def size_of_model(model, centroid_bits=DEFAULT_CENTROID_BITS):
    """Huffman-coded stored size of `model` in bits, and its CR (Eq. 2)."""
    total_original, total_compressed = 0, 0
    for _, module in model.named_modules():
        for param_name in compressible_param_names(module):
            original, compressed = get_compression(
                module, param_name, huffman_encoding=True, centroid_bits=centroid_bits)
            total_original += original
            total_compressed += compressed
    return total_compressed, total_original / total_compressed


def compression_ratios(model, centroid_bits=DEFAULT_CENTROID_BITS):
    """CR of `model` with fixed-width indices (``cr_raw``, Eq. 1) and Huffman-coded indices
    (``cr_huffman``, Eq. 2), plus the bit counts behind them."""
    original_bits = raw_bits = huffman_bits = 0.0
    for _, module in model.named_modules():
        for name in compressible_param_names(module):
            param = getattr(module, name, None)
            if param is None:
                continue
            all_weights = param.numel()
            original_bits += all_weights * ORIGINAL_WEIGHT_BITS

            centers, idx = _index_stream(module, name)
            if centers is None:
                raw_bits += all_weights * ORIGINAL_WEIGHT_BITS
                huffman_bits += all_weights * ORIGINAL_WEIGHT_BITS
                continue

            num_centers = centers.numel()
            weight_bits = np.ceil(math.log2(num_centers)) if num_centers > 0 else 0
            q_idx_raw = idx.numel() * weight_bits
            if len(idx) > 0 and weight_bits > 0:
                q_idx_huffman = _huffman_bits(idx, num_centers, weight_bits, q_idx_raw)
            else:
                q_idx_huffman = q_idx_raw
            centroids_bits = num_centers * centroid_bits
            raw_bits += centroids_bits + q_idx_raw
            huffman_bits += centroids_bits + q_idx_huffman

    return {
        "original_bits": original_bits,
        "raw_bits": raw_bits,
        "huffman_bits": huffman_bits,
        "cr_raw": original_bits / raw_bits if raw_bits > 0 else 0.0,
        "cr_huffman": original_bits / huffman_bits if huffman_bits > 0 else 0.0,
    }
