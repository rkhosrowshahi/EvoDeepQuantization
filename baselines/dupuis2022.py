"""Dupuis, Novo, O'Connor and Bosio, "A Heuristic Exploration of Retraining-Free
Weight-Sharing for CNN Compression" (ASP-DAC 2022) [17].

Bi-level, layer-wise weight sharing, following the official code
(https://github.com/e-dupuis/retraining-free-weight-sharing, the
``two_step_optimization`` path of its main.py):

    Level 1 (`layer_exploration`): for each layer alone, sweep its number of
        clusters k (other layers untouched), measure the network's accuracy, and
        keep the most accurate k at each index bit-width as that layer's candidates.
    Level 2 (`combination_search`): NSGA-II over one gene per layer (an index into
        its candidate list) with objectives (accuracy loss, -CR). Every candidate
        is clustered and evaluated; no accuracy predictor is used.

Differences from the reference implementation:
  * Per-layer 1-D K-means uses scikit-learn with the reference's linear
    (linspace) initialization instead of fast_pytorch_kmeans.
  * Integer genes are encoded as continuous SBX/PM genes rounded at evaluation
    (the reference's int_sbx/int_pm operators are not in current pymoo).
  * Every non-bias weight is clustered, including BatchNorm scales, matching the
    parameter set of every other method here (the reference skips BatchNorm).
  * The CR is computed with the same accounting as every other method; NSGA-II
    optimizes the Huffman-coded CR. ``cr_raw`` (fixed-width indices, the
    reference's own accounting) is reported as well.
  * No pruning and no fine-tuning.

Not applicable to ViT-B/16: torchvision's ViT keeps parameters on modules
without a ``weight`` (e.g. the fused attention projection); the reference is
CNN-only as well.
"""

import copy

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from tqdm import tqdm

# utils.compression sets OMP_NUM_THREADS, so it is imported before scikit-learn.
from utils.compression import compression_ratios, is_compressible_param, uniform_quantization
from sklearn.cluster import KMeans


def _named_compressible_layers(model):
    """(module_name, layer) for every module with a compressible (non-bias) parameter."""
    out = []
    for name, layer in model.named_modules():
        for param_name, param in layer.named_parameters(recurse=False):
            if param is None or not is_compressible_param(param_name):
                continue
            out.append((name, layer))
    return out


def layer_1d_kmeans(w, k):
    """1-D K-means of one layer's weights with a linear (linspace) init, as in the reference.

    Empty clusters are dropped. Returns (centers, labels, inertia).
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1, 1)
    k = min(int(k), len(w))
    if k <= 1:
        center = float(w.mean())
        labels = np.zeros(len(w), dtype=np.int64)
        inertia = float(np.sum((w.flatten() - center) ** 2))
        return np.array([center], dtype=np.float32), labels, inertia

    lo, hi = float(w.min()), float(w.max())
    init_centers = np.linspace(lo, hi, k).reshape(-1, 1)
    km = KMeans(n_clusters=k, init=init_centers, n_init=1, max_iter=300, random_state=0)
    km.fit(w)

    labels = km.labels_
    centers_full = km.cluster_centers_.flatten()
    counts = np.bincount(labels, minlength=k)
    nonempty = np.flatnonzero(counts)
    remap = -np.ones(k, dtype=np.int64)
    remap[nonempty] = np.arange(len(nonempty))
    centers = centers_full[nonempty].astype(np.float32)
    new_labels = remap[labels].astype(np.int64)
    return centers, new_labels, float(km.inertia_)


def cluster_layers(model, k_vector):
    """Cluster a fresh copy of `model` with one k per layer (None leaves a layer untouched).

    Always starts from the pristine model, so every candidate is measured against
    the original weights. Returns (compressed_model, inertias).
    """
    compressed = copy.deepcopy(model)
    layers = [layer for _, layer in _named_compressible_layers(compressed)]
    if len(layers) != len(k_vector):
        raise ValueError(f"k_vector length {len(k_vector)} != {len(layers)} compressible layers")

    inertias = []
    for layer, k in zip(layers, k_vector):
        if k is None:
            inertias.append(0.0)
            continue
        w = layer.weight.detach().cpu().numpy().astype(np.float64).flatten()
        centers, labels, inertia = layer_1d_kmeans(w, k)
        uniform_quantization(layer, "weight", centers=centers.reshape(-1, 1), indices=labels)
        inertias.append(inertia)
    return compressed, inertias


def layer_exploration(model, val_cache, baseline_acc, min_acc_fraction=0.98,
                      max_bit_range=8, min_k=2, relax_factor=0.9, max_relax_iters=4, verbose=True):
    """Level 1: sweep k per layer, then keep the most accurate k per index bit-width.

    The k grid is log-spaced over [2, 2^max_bit_range]. A candidate must keep
    accuracy > min_acc_fraction * baseline; if some layer has no candidate, the
    threshold is relaxed by `relax_factor` for all layers (up to `max_relax_iters`
    times), as in the reference.

    Returns (k_candidates, layer_df): {layer_idx: sorted candidate k values}, and
    every (layer, k) probe with its accuracy.
    """
    named_layers = _named_compressible_layers(model)
    n_layers = len(named_layers)
    layer_names = [name for name, _ in named_layers]

    rows = []
    for li, (name, layer) in enumerate(named_layers):
        n_weights = layer.weight.numel()
        raw = np.logspace(1, max_bit_range, max_bit_range ** 2, base=2)
        k_range = list(dict.fromkeys(int(k) for k in raw))
        k_range = [k for k in k_range if min_k <= k <= n_weights]

        iterator = tqdm(k_range, desc=f"[dupuis2022] layer_exploration {name}") if verbose else k_range
        for k in iterator:
            k_vector = [None] * n_layers
            k_vector[li] = k
            compressed, inertias = cluster_layers(model, k_vector)
            acc = val_cache.accuracy(compressed)
            rows.append({
                "layer_idx": li, "layer": name, "k": k,
                "inertia": inertias[li], "accuracy": acc,
                "accuracy_loss": baseline_acc - acc,
            })
            del compressed

    df = pd.DataFrame(rows)
    df["nb_bit"] = np.ceil(np.log2(df["k"])).astype(int)

    threshold = min_acc_fraction
    k_candidates = None
    for _ in range(max_relax_iters + 1):
        filtered = df[df.accuracy > threshold * baseline_acc]
        missing = [name for name in layer_names if (filtered.layer == name).sum() == 0]
        if not missing:
            grouped = filtered.groupby(["layer", "nb_bit"])
            best = filtered.loc[grouped["accuracy_loss"].idxmin().values]
            k_candidates = {
                li: sorted(best.loc[best.layer == name, "k"].unique().tolist())
                for li, name in enumerate(layer_names)
            }
            break
        if verbose:
            print(f"  [dupuis2022] {len(missing)}/{n_layers} layer(s) with no candidate "
                  f"at accuracy >= {threshold:.2%} of baseline; relaxing to {threshold * relax_factor:.2%}")
        threshold *= relax_factor
    if k_candidates is None:
        raise RuntimeError(
            f"layer_exploration: no candidate found for every layer even after "
            f"{max_relax_iters} relaxations (final threshold {threshold:.2%})")

    if verbose:
        sizes = [len(v) for v in k_candidates.values()]
        combos = 1
        for s in sizes:
            combos *= s
        print(f"  [dupuis2022] candidates/layer: min={min(sizes)} max={max(sizes)} "
              f"mean={np.mean(sizes):.1f}  combinations={combos:.2e}")

    return k_candidates, df


def combination_search(model, k_candidates, val_cache, baseline_acc, pop_size=100, n_gen=50,
                       seed=0, verbose=True):
    """Level 2: NSGA-II over one candidate index per layer, objectives (accuracy loss, -CR).

    Repeated genotypes are evaluated once. Returns (pareto_df, eval_cache):
    pareto_df has one row per non-dominated combination (k_vector, accuracy,
    cr_raw, cr_huffman).
    """
    named_layers = _named_compressible_layers(model)
    n_layers = len(named_layers)
    candidate_lists = [k_candidates[li] for li in range(n_layers)]
    xu = np.array([len(c) - 1 for c in candidate_lists], dtype=float)
    xl = np.zeros(n_layers)

    eval_cache = {}

    def _decode(x):
        idx = np.clip(np.round(x).astype(int), 0, xu.astype(int))
        return tuple(idx.tolist())

    class WeightSharingProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n_layers, n_obj=2, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            n = len(X)
            f1 = np.zeros(n)
            f2 = np.zeros(n)
            for i in range(n):
                idx = _decode(X[i])
                if idx in eval_cache:
                    acc, cr_raw, cr_huffman = eval_cache[idx]
                else:
                    k_vector = [candidate_lists[j][idx[j]] for j in range(n_layers)]
                    compressed, _ = cluster_layers(model, k_vector)
                    acc = val_cache.accuracy(compressed)
                    ratios = compression_ratios(compressed)
                    cr_raw, cr_huffman = ratios["cr_raw"], ratios["cr_huffman"]
                    del compressed
                    eval_cache[idx] = (acc, cr_raw, cr_huffman)
                f1[i] = baseline_acc - acc
                f2[i] = -cr_huffman
            out["F"] = np.column_stack([f1, f2])

    problem = WeightSharingProblem()
    algorithm = NSGA2(pop_size=pop_size, crossover=SBX(prob=0.9, eta=15),
                      mutation=PM(eta=20), eliminate_duplicates=True)

    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=verbose)

    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    rows = []
    for x, f in zip(X, F):
        idx = _decode(x)
        acc, cr_raw, cr_huffman = eval_cache[idx]
        k_vector = [candidate_lists[j][idx[j]] for j in range(n_layers)]
        rows.append({
            "k_vector": k_vector,
            "val_eval_score": acc,
            "accuracy_loss": f[0],
            "cr_raw": cr_raw,
            "cr_huffman": cr_huffman,
        })
    pareto_df = pd.DataFrame(rows).drop_duplicates(subset=["cr_huffman", "val_eval_score"])
    pareto_df = pareto_df.sort_values("cr_huffman").reset_index(drop=True)

    if verbose:
        print(f"  [dupuis2022] combination_search: {len(eval_cache)} distinct combinations "
              f"evaluated, {len(pareto_df)} on the returned Pareto front")

    return pareto_df, eval_cache


def run_dupuis(model, val_cache, baseline_acc, min_acc_fraction=0.98, max_bit_range=8,
               min_k=2, pop_size=100, n_gen=50, seed=0, verbose=True):
    """Level 1 then Level 2. Returns (pareto_df, k_candidates, layer_df)."""
    k_candidates, layer_df = layer_exploration(
        model, val_cache, baseline_acc, min_acc_fraction=min_acc_fraction,
        max_bit_range=max_bit_range, min_k=min_k, verbose=verbose)
    pareto_df, _ = combination_search(
        model, k_candidates, val_cache, baseline_acc, pop_size=pop_size, n_gen=n_gen,
        seed=seed, verbose=verbose)
    return pareto_df, k_candidates, layer_df
