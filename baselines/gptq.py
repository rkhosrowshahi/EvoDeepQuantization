"""Frantar, Ashkboos, Hoefler and Alistarh, "GPTQ: Accurate Post-Training Quantization
for Generative Pre-trained Transformers" (ICLR 2023) [18].

Port of the official implementation (https://github.com/IST-DASLab/gptq: the
``GPTQ`` class of gptq.py and ``Quantizer`` of quant.py, default path:
min/max affine parameters, no static groups). For each layer:

  1. Accumulate H = 2 * mean(x x^T) over the layer's calibration inputs
     (im2col patches for Conv2d, rows for Linear).
  2. Quantize the weight columns one at a time to a fixed b-bit affine grid
     (round-to-nearest with a per-output-row scale and zero point), propagating
     each column's error into the remaining columns through the Cholesky factor
     of H^-1, in blocks ("lazy batch updates").

Layers are processed sequentially. Each layer's calibration pass runs on the
model with all earlier layers already quantized, so their error propagates into
later layers' statistics. Residual connections make the official per-block
caching awkward, so this costs one forward pass over the calibration batches per
layer.

Scope: every nn.Conv2d and nn.Linear (the official ``find_layers`` types). This
excludes the fused attention projection of torchvision's ViT-B/16, so ViT-B/16
is not covered. No pruning and no fine-tuning.

Storage: fixed-width b-bit codes plus a 16-bit scale and zero point per output
channel (GPTQ's own convention), without Huffman coding.
"""

import math

import torch
import torch.nn as nn
from tqdm import tqdm


def _weight_matrix(layer):
    """`layer.weight` as (out_features, in_features); Conv2d flattens to (out_c, in_c*kh*kw)."""
    W = layer.weight.data
    if isinstance(layer, nn.Conv2d):
        return W.flatten(1)
    return W


def _find_affine_params(W_slice, bits, sym=False):
    """Per-output-row min/max affine parameters (quant.py ``Quantizer.find_params``)."""
    maxq = 2 ** bits - 1
    zeros = torch.zeros(W_slice.shape[0], device=W_slice.device, dtype=W_slice.dtype)
    xmin = torch.minimum(W_slice.min(dim=1).values, zeros)
    xmax = torch.maximum(W_slice.max(dim=1).values, zeros)
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        xmin = torch.where(xmin < 0, -xmax, xmin)
    degenerate = (xmin == 0) & (xmax == 0)
    xmin = torch.where(degenerate, -torch.ones_like(xmin), xmin)
    xmax = torch.where(degenerate, torch.ones_like(xmax), xmax)
    scale = (xmax - xmin) / maxq
    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = torch.round(-xmin / scale)
    return scale.unsqueeze(1), zero.unsqueeze(1), maxq


def _affine_quantize(w, scale, zero, maxq):
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


class _GPTQLayer:
    """Hessian accumulator and column-wise quantizer of one Conv2d/Linear layer."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = _weight_matrix(layer).shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp, image_chunk_size=1):
        if isinstance(self.layer, nn.Linear):
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            self._accumulate(inp.t())
        elif isinstance(self.layer, nn.Conv2d):
            # Unfold a few images at a time to bound memory on wide early layers;
            # the running-average update of H is the same either way.
            unfold = nn.Unfold(
                self.layer.kernel_size, dilation=self.layer.dilation,
                padding=self.layer.padding, stride=self.layer.stride,
            )
            for start in range(0, inp.shape[0], image_chunk_size):
                chunk = unfold(inp[start:start + image_chunk_size])   # (chunk, columns, n_patches)
                chunk = chunk.permute(1, 0, 2).flatten(1)             # (columns, chunk*n_patches)
                self._accumulate(chunk)

    def _accumulate(self, inp):
        tmp = inp.shape[1]
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    @torch.no_grad()
    def quantize(self, bits, blocksize=128, percdamp=0.01, group_size=-1, actorder=False, sym=False):
        """Quantize the layer's weight in place; returns the number of (scale, zero) groups stored."""
        W = _weight_matrix(self.layer).clone().float()
        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        Q = torch.zeros_like(W)
        if group_size == -1:
            scale, zero, maxq = _find_affine_params(W, bits, sym=sym)
        else:
            scale = zero = maxq = None

        n_groups_per_row = 1 if group_size == -1 else math.ceil(self.columns / group_size)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1 and (i1 + i) % group_size == 0:
                    g_end = min(i1 + i + group_size, self.columns)
                    scale, zero, maxq = _find_affine_params(W[:, (i1 + i):g_end], bits, sym=sym)

                q = _affine_quantize(w.unsqueeze(1), scale, zero, maxq).flatten()
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)
        return self.rows * n_groups_per_row


def _target_layers(model):
    """(name, layer) for every nn.Conv2d / nn.Linear, in named_modules() order."""
    return [(name, layer) for name, layer in model.named_modules()
            if isinstance(layer, (nn.Conv2d, nn.Linear))]


def gptq_quantize_network(model, calib_batches, bits, group_size=-1, percdamp=0.01,
                          actorder=False, sym=False, blocksize=128, verbose=True):
    """Quantize every Conv2d/Linear layer of `model` in place, one layer at a time.

    calib_batches: (images, labels) tensors on the model's device.
    Returns [(layer_name, rows, columns, n_groups)] for `gptq_compressed_bits`.
    """
    targets = _target_layers(model)
    stats = []
    param_dtype = next(model.parameters()).dtype
    iterator = tqdm(targets, desc="[gptq] quantizing layers") if verbose else targets
    for name, layer in iterator:
        acc = _GPTQLayer(layer)

        def _hook(_module, inp, _out):
            acc.add_batch(inp[0].detach())

        handle = layer.register_forward_hook(_hook)
        with torch.no_grad():
            for images, _ in calib_batches:
                model(images.to(dtype=param_dtype))
        handle.remove()

        n_groups = acc.quantize(bits, blocksize=blocksize, percdamp=percdamp,
                                group_size=group_size, actorder=actorder, sym=sym)
        stats.append((name, acc.rows, acc.columns, n_groups))
    return stats


def gptq_compressed_bits(stats, bits, model=None, sym=False, scale_bits=16, original_bits=32):
    """Stored size of a `gptq_quantize_network` result, without entropy coding.

    Per quantized layer: rows * columns codes of `bits` bits, plus one scale (and,
    if asymmetric, one zero point) of `scale_bits` bits per group. With `model`,
    every other compressible weight (BatchNorm scales) is counted at 32 bits on
    both sides of the ratio, so the CR covers the same parameters as every other
    method.

    Returns (original_bits, compressed_bits, bits_per_weight, cr).
    """
    targeted_weights = sum(rows * columns for _, rows, columns, _ in stats)
    index_bits = targeted_weights * bits
    n_groups_total = sum(n_groups for _, _, _, n_groups in stats)
    overhead_bits = n_groups_total * scale_bits * (2 if not sym else 1)
    compressed_bits = index_bits + overhead_bits
    original_total_bits = targeted_weights * original_bits

    if model is not None:
        from utils.compression import get_model_params
        all_weights = len(get_model_params(model))
        untouched_weights = all_weights - targeted_weights
        compressed_bits += untouched_weights * original_bits
        original_total_bits += untouched_weights * original_bits

    bits_per_weight = compressed_bits / (targeted_weights if model is None else all_weights)
    cr = original_total_bits / compressed_bits
    return original_total_bits, compressed_bits, bits_per_weight, cr
