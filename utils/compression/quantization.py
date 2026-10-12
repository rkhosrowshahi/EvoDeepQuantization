"""Weight sharing with one codebook for the whole network.

All non-bias parameters are flattened into a single vector, in
``model.named_modules()`` order, with no use of layer boundaries or shapes
(`get_model_params`). That vector is mapped onto a shared codebook, either by
Uniform Binning Quantization (UBQ, Section III-A) or, for the Deep Compression
baseline, by K-means. The codebook is then written back into every tensor
(`set_model_params`). Each tensor keeps a reference to the codebook
(``<name>_centers``) and its own index stream (``<name>_indices``); a forward
pre-hook rebuilds the weights from them. ``utils/compression/storage.py`` turns
these into a storage cost.

Biases are left in full precision and are not counted in the storage either.
They are a tiny fraction of the weights but individually sensitive.
"""

import copy
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

# scikit-learn's KMeans leaks memory on Windows unless OpenMP runs single-threaded.
os.environ['OMP_NUM_THREADS'] = "1"

from sklearn.cluster import KMeans  # noqa: E402


def is_compressible_param(param_name):
    """Whether a parameter is shared through the codebook (everything but biases)."""
    return not param_name.endswith("bias")


def _strip_compression_suffixes(name):
    """Base parameter name of a quantized tensor's ``_centers`` / ``_indices`` entry."""
    for suffix in ("_centers", "_indices"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def compressible_param_names(module):
    """Base names of `module`'s own compressible parameters, quantized or not."""
    names, seen = [], set()

    def add(base):
        if not base or base in seen or not is_compressible_param(base):
            return
        if getattr(module, base, None) is None:
            return
        seen.add(base)
        names.append(base)

    for name, _ in module.named_parameters(recurse=False):
        add(_strip_compression_suffixes(name))
    for name, _ in module.named_buffers(recurse=False):
        if name.endswith("_indices"):
            add(_strip_compression_suffixes(name))
    return names


class SharedCodebookLookup:
    """Forward pre-hook that rebuilds ``module.<name>`` from its codebook and indices."""

    def __init__(self, name, shape):
        self._tensor_name = name
        self._shape = shape

    def __call__(self, module, inputs):
        setattr(module, self._tensor_name, self.lookup_weights(module))

    def lookup_weights(self, module):
        indices = getattr(module, self._tensor_name + '_indices')
        centers = getattr(module, self._tensor_name + '_centers')
        weights = F.embedding(indices, centers).squeeze()
        return weights.to(torch.float32).view(self._shape)


def uniform_quantization(module, name, centers, indices):
    """Replace ``module.<name>`` by codebook `centers` (K x 1) indexed by `indices`."""
    param = getattr(module, name).detach()
    device = param.device
    method = SharedCodebookLookup(name, param.shape)

    centers = torch.nn.Parameter(torch.from_numpy(centers).float().to(device))
    indices = torch.from_numpy(indices).to(device)
    if name in module._parameters:
        del module._parameters[name]
    module.register_parameter(name + "_centers", centers)
    module.register_buffer(name + "_indices", indices)
    setattr(module, name, method.lookup_weights(module))
    module.register_forward_pre_hook(method)
    return module


def get_model_params(model):
    """All compressible parameters of `model`, flattened into one vector."""
    param_list = []
    for _, layer in model.named_modules():
        for param_name, param in layer.named_parameters(recurse=False):
            if param is None or not is_compressible_param(param_name):
                continue
            param_list.append(param.flatten().detach().cpu().numpy())
    if len(param_list) == 0:
        return np.array([])
    return np.concatenate(param_list)


def set_model_params(model, centers, bin_indices):
    """Write a shared codebook into `model` in place.

    `bin_indices` follows the flattening order of `get_model_params`; every tensor
    receives its own slice of it and the whole codebook `centers`.
    """
    start = 0
    for _, layer in model.named_modules():
        for param_name, param in list(layer.named_parameters(recurse=False)):
            if param is None or not is_compressible_param(param_name):
                continue
            n = param.shape.numel()
            uniform_quantization(layer, param_name, centers=centers.reshape(-1, 1),
                                 indices=bin_indices[start: start + n])
            start += n
    return model


def uniform_bin_edges(min_weight, max_weight, num_bins):
    """Edges of `num_bins` equal-width bins over [min_weight, max_weight], anchored at zero.

    The bin width is (max - min) / (num_bins - 1) (Eq. 3), and the grid is shifted so
    that 0 sits at a bin centre. The bin holding the bulk of the weights is then
    symmetric about zero and its mean stays near zero for every num_bins.
    """
    if num_bins < 2:
        return np.linspace(min_weight, max_weight, max(num_bins, 1))

    width = (max_weight - min_weight) / (num_bins - 1)
    if not np.isfinite(width) or width <= 0:
        return np.linspace(min_weight, max_weight, num_bins)

    # Centres on the lattice j * width; edges at (j + 0.5) * width (Eq. 4).
    first = math.floor(min_weight / width - 0.5)
    return (np.arange(num_bins) + first + 0.5) * width


def _binning_from_edges(params, bin_edges):
    """Assign each weight to its bin (Eq. 5), drop empty bins, and use each bin's mean (Eq. 6).

    Returns (centers, bin_indices) with indices compacted to the non-empty bins.
    """
    params = np.asarray(params)
    if len(params) == 0:
        return np.array([]), np.array([])

    bin_edges = np.asarray(bin_edges)
    num_bins = len(bin_edges)
    raw_bin_indices = np.digitize(params, bin_edges) - 1
    raw_bin_indices = np.clip(raw_bin_indices, 0, num_bins - 1)

    bin_counts = np.bincount(raw_bin_indices, minlength=num_bins)
    nonempty_bins = np.flatnonzero(bin_counts)
    if len(nonempty_bins) == 0:
        return np.array([]), np.array([])

    old_to_new = np.full(num_bins, -1, dtype=np.int64)
    old_to_new[nonempty_bins] = np.arange(len(nonempty_bins), dtype=np.int64)
    bin_indices = old_to_new[raw_bin_indices]

    weight_sums = np.bincount(raw_bin_indices, weights=params, minlength=num_bins)
    centers = weight_sums[nonempty_bins] / bin_counts[nonempty_bins]
    return centers, bin_indices


def uniform_binning(params, num_bins):
    """UBQ: `num_bins` zero-anchored equal-width bins over the weight range, in O(N)."""
    params = np.asarray(params)
    if len(params) == 0:
        return np.array([]), np.array([])
    bin_edges = uniform_bin_edges(params.min(), params.max(), num_bins)
    return _binning_from_edges(params, bin_edges)


def kmeans_binning(params, num_bins, n_init=10, max_iter=300, random_state=None):
    """Global K-means codebook (k-means++ init, `n_init` restarts); empty clusters dropped."""
    kmeans = KMeans(
        n_clusters=num_bins,
        init='k-means++',
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    kmeans.fit(params.reshape(-1, 1))

    centers = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_
    new_centers = []
    new_labels = labels.copy()
    for center_idx, label in enumerate(np.unique(labels)):
        new_centers.append(centers[label])
        new_labels[np.where(labels == label)[0]] = center_idx
    return np.array(new_centers), new_labels


def compress(num_bins, model, method='uniform', n_init=10, max_iter=300, random_state=None,
             build_model=True):
    """Quantize every compressible weight of `model` with one shared codebook.

    Args:
        num_bins: requested codebook size (K0 for UBQ, K for K-means)
        model: the pretrained model (not modified)
        method: 'uniform' (UBQ) or 'kmeans'
        n_init, max_iter, random_state: K-means settings
        build_model: when False, skip copying `model` and only return the codebook
            and indices (the compressed model comes back as None)

    Returns:
        (compressed_model, centers, bin_indices)
    """
    num_bins = int(num_bins)
    model.eval()
    compressed_model = None
    if build_model:
        compressed_model = copy.deepcopy(model)
        compressed_model.eval()
    params = get_model_params(model)

    if method == 'uniform':
        centers, bin_indices = uniform_binning(params, num_bins)
    elif method == 'kmeans':
        centers, bin_indices = kmeans_binning(
            params, num_bins, n_init=n_init, max_iter=max_iter, random_state=random_state)
    else:
        raise ValueError(f"Unknown compression method: {method}. Must be 'uniform' or 'kmeans'")

    if compressed_model is not None:
        set_model_params(compressed_model, centers, bin_indices)
    return compressed_model, centers, bin_indices
