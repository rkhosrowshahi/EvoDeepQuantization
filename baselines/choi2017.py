"""Choi, El-Khamy and Lee, "Towards the Limit of Network Quantization" (ICLR 2017) [6].

Reimplementation of two of the paper's schemes, following its equations and the
Algorithm 1 pseudocode of its Appendix A.3 (no reference code is available):

    * Hessian-weighted K-means (Eq. 7)          -> `hessian_weighted_kmeans`
    * Iterative ECSQ (Algorithm 1, App. A.3)     -> `iterative_ecsq`

The other two schemes reported in that paper are run by ``run_kmeans.py``
(K-means) and ``run_uniform.py`` (Uniform Quantization).

Hessian: the paper's diagonal Hessian is computed with a second-order
backpropagation recursion [2], [38] that has no published implementation.
`fisher_diagonal` uses the empirical Fisher diagonal E[(dL/dw)^2] over
per-sample gradients on calibration images instead. That is the standard
diagonal-Hessian stand-in for a cross-entropy loss and targets the same quantity
as the paper's own Section 3.5 alternative (Adam's second-moment estimate).

As for every method in this repository: no pruning and no fine-tuning.
"""

import contextlib
import copy
import time

import numpy as np
import torch

# utils.compression sets OMP_NUM_THREADS, so it is imported before scikit-learn.
from utils.compression import get_model_params, is_compressible_param, set_model_params
from sklearn.cluster import KMeans


def _compressible_param_list(model):
    """(layer, param_name, param) for every compressible parameter, in `get_model_params` order."""
    out = []
    for _, layer in model.named_modules():
        for param_name, param in layer.named_parameters(recurse=False):
            if param is None or not is_compressible_param(param_name):
                continue
            out.append((layer, param_name, param))
    return out


@contextlib.contextmanager
def _deterministic_kernels():
    """Use deterministic GPU kernels inside the block, then restore the previous settings.

    cuDNN's default backward kernels accumulate with atomics, so without this the
    Fisher diagonal, and every clustering weighted by it, differs slightly between runs.
    """
    cudnn_deterministic = torch.backends.cudnn.deterministic
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)


def fisher_diagonal(model, batches, device, max_samples=500, seed=0):
    """Empirical Fisher diagonal, aligned with `get_model_params(model)`.

    batches: (images, labels) tensors on `device` (e.g. a CachedEvalSet's .batches).
    Averages per-sample squared gradients (not the square of the batch-mean
    gradient, which is biased by cross terms), one backward pass per image, over
    `max_samples` randomly chosen images. Computed with deterministic GPU kernels,
    so it is identical across runs.
    """
    with _deterministic_kernels():
        return _fisher_diagonal(model, batches, device, max_samples, seed)


def _fisher_diagonal(model, batches, device, max_samples, seed):
    model.eval()
    params = _compressible_param_list(model)
    sq_sum = [torch.zeros_like(p, dtype=torch.float64) for _, _, p in params]

    rng = np.random.default_rng(seed)
    all_images = torch.cat([b[0] for b in batches], dim=0)
    all_labels = torch.cat([b[1] for b in batches], dim=0)
    n_total = all_images.shape[0]
    idx = rng.choice(n_total, size=min(max_samples, n_total), replace=False)

    param_dtype = next(model.parameters()).dtype
    n_done = 0
    for i in idx:
        model.zero_grad(set_to_none=True)
        x = all_images[i:i + 1].to(device=device, dtype=param_dtype)
        y = all_labels[i:i + 1].to(device=device)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        for (_, _, p), acc in zip(params, sq_sum):
            if p.grad is not None:
                acc += p.grad.detach().to(torch.float64) ** 2
        n_done += 1
    model.zero_grad(set_to_none=True)

    flat = [(acc / max(n_done, 1)).flatten().cpu().numpy() for acc in sq_sum]
    return np.concatenate(flat) if flat else np.array([])


def hessian_weighted_kmeans(w, h, k, init="uniform", n_init=1, max_iter=300, random_state=0):
    """Hessian-weighted K-means (Eq. 7): minimize sum_j sum_{w_i in C_j} h_i |w_i - c_j|^2.

    h_i > 0 does not depend on the cluster, so it drops out of the assignment
    (nearest center) and only reweights the center update to sum(h_i w_i) / sum(h_i).
    That is exactly scikit-learn's KMeans with `sample_weight`. init="uniform"
    places the initial centers on a linear grid (the paper's Section 4.3);
    "random" uses k-means++. Empty clusters are dropped.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1, 1)
    h = np.asarray(h, dtype=np.float64)
    k = min(k, len(w))

    if init == "uniform":
        lo, hi = w.min(), w.max()
        edges = np.linspace(lo, hi, k + 1)
        init_centers = ((edges[:-1] + edges[1:]) / 2.0).reshape(-1, 1)
        n_init = 1
    else:
        init_centers = "k-means++"

    km = KMeans(n_clusters=k, init=init_centers, n_init=n_init, max_iter=max_iter, random_state=random_state)
    km.fit(w, sample_weight=h)

    labels = km.labels_
    centers_full = km.cluster_centers_.flatten()
    counts = np.bincount(labels, minlength=k)
    nonempty = np.flatnonzero(counts)
    remap = -np.ones(k, dtype=np.int64)
    remap[nonempty] = np.arange(len(nonempty))
    return centers_full[nonempty].astype(np.float32), remap[labels].astype(np.int64)


def iterative_ecsq(w, h, k, lam, init_centers=None, max_iter=100, tol=1e-6, min_iter=5,
                   device="cpu", chunk_size=200_000):
    """Iterative entropy-constrained quantization (Algorithm 1 of Appendix A.3).

    Assignment: w_i -> argmin_j { h_i (w_i - c_j)^2 - lam log2(p_j) }, with the
                previous iteration's p_j.
    Update:     c_j <- sum(h_i w_i) / sum(h_i) over C_j,   p_j <- |C_j| / N.

    init_centers: starting centers. Real weights are heavy-tailed, so a linear grid
        over [min(w), max(w)] (the fallback when None) leaves most centers in
        near-empty tails; the sweeps warm-start from `hessian_weighted_kmeans` at k.
    tol: relative change of the objective J that stops the iteration (after
        `min_iter` iterations); an absolute tolerance would depend on the scale of h.

    lam = 0 reduces to Hessian-weighted K-means; h = ones(N) gives the
    non-Hessian-weighted variant.

    Implementation notes:
    * The assignment uses the equivalent cost (w_i - c_j)^2 - (lam / h_i) log2(p_j).
      Some weights have h_i = 0 (no gradient on the calibration images); for them
      the literal cost is 0 for every cluster, so they are sent to the cluster that
      is cheapest to encode instead of all to cluster 0.
    * Empty clusters are relocated to the points with the largest weighted error,
      as scikit-learn's KMeans does; the pseudocode has no rule for them.

    Returns (centers, bin_indices, n_iter, converged, final_J).
    """
    w_t = torch.as_tensor(np.asarray(w, dtype=np.float64), device=device)
    h_t = torch.as_tensor(np.asarray(h, dtype=np.float64), device=device)
    N = w_t.numel()
    if N == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64), 0, True, 0.0

    k = min(k, N)
    lo, hi = float(w_t.min()), float(w_t.max())
    if k <= 1 or hi <= lo:
        c = float(w_t.mean())
        return np.array([c], dtype=np.float32), np.zeros(N, dtype=np.int64), 0, True, 0.0

    if init_centers is not None:
        centers = torch.as_tensor(np.asarray(init_centers, dtype=np.float64), device=device)
        if centers.numel() != k:
            k = centers.numel()  # the warm start may have dropped empty clusters
    else:
        edges = torch.linspace(lo, hi, k + 1, dtype=torch.float64, device=device)
        centers = (edges[:-1] + edges[1:]) / 2.0
    p = torch.full((k,), 1.0 / k, dtype=torch.float64, device=device)

    assign = torch.zeros(N, dtype=torch.long, device=device)
    prev_J, converged, n_iter = None, False, 0
    for n_iter in range(1, max_iter + 1):
        log_p = torch.log2(torch.clamp(p, min=1e-12))
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            w_chunk = w_t[start:end].unsqueeze(1)
            h_row = h_t[start:end]
            dist_sq = (w_chunk - centers.unsqueeze(0)) ** 2  # (chunk, k)
            if lam == 0:
                cost = dist_sq
            else:
                h_safe = torch.clamp(h_row, min=1e-30).unsqueeze(1)
                cost = dist_sq - (lam / h_safe) * log_p.unsqueeze(0)
                zero_h = h_row == 0
                if zero_h.any():
                    cost[zero_h] = -log_p.unsqueeze(0).expand(int(zero_h.sum()), -1)
            assign[start:end] = cost.argmin(dim=1)

        counts = torch.bincount(assign, minlength=k).to(torch.float64)
        hw_sum = torch.zeros(k, dtype=torch.float64, device=device).scatter_add_(0, assign, h_t * w_t)
        h_sum = torch.zeros(k, dtype=torch.float64, device=device).scatter_add_(0, assign, h_t)
        nonempty = h_sum > 0
        new_centers = centers.clone()
        new_centers[nonempty] = hw_sum[nonempty] / h_sum[nonempty]
        new_p = counts / N

        if not bool(nonempty.all()):
            n_empty = int((~nonempty).sum().item())
            err = h_t * (w_t - new_centers[assign]) ** 2
            _, top_idx = torch.topk(err, k=min(n_empty, N))
            empty_slots = torch.nonzero(~nonempty, as_tuple=True)[0]
            new_centers[empty_slots[: len(top_idx)]] = w_t[top_idx]

        D = (h_t * (w_t - new_centers[assign]) ** 2).sum().item() / N
        valid_p = new_p[new_p > 0]
        H = -(valid_p * torch.log2(valid_p)).sum().item()
        J = D + lam * H

        centers, p = new_centers, new_p
        if prev_J is not None and n_iter >= min_iter and abs(prev_J - J) < tol * abs(prev_J):
            converged = True
            prev_J = J
            break
        prev_J = J

    counts = torch.bincount(assign, minlength=k)
    keep = counts > 0
    if not bool(keep.all()):
        remap = torch.full((k,), -1, dtype=torch.long, device=device)
        remap[keep] = torch.arange(int(keep.sum().item()), device=device)
        assign = remap[assign]
        centers = centers[keep]

    return (
        centers.detach().cpu().numpy().astype(np.float32),
        assign.detach().cpu().numpy().astype(np.int64),
        n_iter, converged, prev_J,
    )


def bracket_lambda(w, h, warm_centers, k_init, device,
                   probe_exponents=(-16, 4), n_probe=21, probe_max_iter=12, n_final=11):
    """Lambda grid for `iterative_ecsq`, bracketed on this model's own weight/Hessian scale.

    Lambda is in units of Hessian-weighted squared error per bit, so a useful range
    depends on the checkpoint. A cheap probe (few iterations, no accuracy evaluation)
    over a wide log-spaced range finds where the realized number of clusters starts
    to fall below the warm start (lam_lo) and where it collapses to <= 3 (lam_hi).
    Returns [0] + n_final values log-spaced in [lam_lo, lam_hi].
    """
    k_start = len(warm_centers)
    probe_lams = np.logspace(probe_exponents[0], probe_exponents[1], n_probe)
    realized = []
    t0 = time.time()
    for lam in probe_lams:
        centers, _, _, _, _ = iterative_ecsq(
            w, h, k_init, lam, init_centers=warm_centers,
            max_iter=probe_max_iter, tol=1e-5, min_iter=3, device=device,
        )
        realized.append(len(centers))
    print(f"  [bracket_lambda] probe: {list(zip([f'{l:.1e}' for l in probe_lams], realized))}  "
          f"[{time.time() - t0:.1f}s]")

    realized = np.array(realized)
    still_full = np.flatnonzero(realized >= k_start)
    lam_lo = probe_lams[still_full[-1]] if len(still_full) else probe_lams[0]
    collapsed = np.flatnonzero(realized <= 3)
    lam_hi = probe_lams[collapsed[0]] if len(collapsed) else probe_lams[-1]

    if lam_hi <= lam_lo:
        lam_lo, lam_hi = probe_lams[0], probe_lams[-1]
        print(f"  [bracket_lambda] WARNING: no clean transition in probe range "
              f"({probe_exponents}); falling back to full probed range.")

    lam_lo = max(lam_lo, probe_lams[0])
    final = [0.0] + list(np.geomspace(lam_lo, lam_hi, n_final))
    print(f"  [bracket_lambda] final grid: lam in [{lam_lo:.2e}, {lam_hi:.2e}], {len(final)} points incl. 0.0")
    return final


def quantize(model, method, k, h_diag, lam=0.0, device="cpu", **kwargs):
    """Quantize a copy of `model` with one shared codebook.

    method: "hessian_kmeans" or "iterative_ecsq".
    h_diag: `fisher_diagonal(...)` output (np.ones(N) for the non-Hessian variant).

    Returns (compressed_model, centers, bin_indices, extra); extra holds
    n_iter / converged / final_J for iterative_ecsq.
    """
    compressed_model = copy.deepcopy(model)
    compressed_model.eval()

    w = get_model_params(compressed_model)
    if len(h_diag) != len(w):
        raise ValueError(f"h_diag length {len(h_diag)} != weight count {len(w)}")

    extra = {}
    if method == "hessian_kmeans":
        centers, bin_indices = hessian_weighted_kmeans(w, h_diag, k, **kwargs)
    elif method == "iterative_ecsq":
        centers, bin_indices, n_iter, converged, J = iterative_ecsq(w, h_diag, k, lam, device=device, **kwargs)
        extra = {"n_iter": n_iter, "converged": converged, "final_J": J}
    else:
        raise ValueError(f"Unknown method: {method!r}")

    set_model_params(compressed_model, centers, bin_indices)
    return compressed_model, centers, bin_indices, extra
