"""Entropy-constrained scalar quantization (ECSQ): optimal merging of neighboring bins.

Given the non-empty bins of a uniform grid, choose a partition into contiguous groups
that minimizes the Lagrangian (Eq. 7)

    J = D + lambda * R

    D = added distortion = sum_g m_g * (c_g - c_merged)^2
        (between-group squared error; the within-bin error does not change with the
        merge, so it is constant across partitions and drops out of the argmin)
    R = rate             = sum_g -m_g * log2(m_g / N)
        (occupancy entropy in bits, the quantity Huffman coding of the index stream pays)

Both terms depend only on each bin's (population, centroid): no weights, calibration
data, forward passes or gradients are needed. Bins are contiguous on the weight axis,
so the optimal partition is found exactly by dynamic programming in O(K^2)
(Algorithm 2). Sweeping lambda gives the merge ladder (Algorithm 3).
"""

import numpy as np

# lambda = 0 keeps every bin (merging can only add distortion); a large lambda merges
# everything into one bin. This span covers both ends for the weight magnitudes seen in practice.
DEFAULT_LAMBDA_GRID = np.geomspace(1e-10, 1e-3, 80)


def dp_merge(counts, centers, lam):
    """Optimal contiguous grouping of K bins under J = D + lam * R (Algorithm 2).

    Args:
        counts: population of each bin, in ascending centroid order
        centers: bin centroids, ascending
        lam: rate/distortion trade-off; 0 keeps all bins

    Returns:
        (group_of_bin, merged_centers): group_of_bin[b] is the merged group that bin b
        falls into; merged_centers holds the population-weighted mean of each group.
    """
    m = np.asarray(counts, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    N = m.sum()
    K = len(m)
    if K == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)

    # Prefix sums M, S, T of Eq. (8).
    M = np.concatenate([[0.0], np.cumsum(m)])
    MC = np.concatenate([[0.0], np.cumsum(m * c)])
    MC2 = np.concatenate([[0.0], np.cumsum(m * c * c)])

    dp = np.full(K + 1, np.inf)
    dp[0] = 0.0
    parent = np.zeros(K + 1, dtype=np.int64)

    for j in range(1, K + 1):
        i = np.arange(j)
        mm = M[j] - M[i]
        ss = MC[j] - MC[i]
        dD = np.maximum((MC2[j] - MC2[i]) - ss * ss / mm, 0.0)  # clip floating-point noise
        dR = -mm * np.log2(mm / N)
        tot = dp[i] + dD + lam * dR
        k = int(np.argmin(tot))
        dp[j], parent[j] = tot[k], k

    groups, j = [], K
    while j > 0:
        i = parent[j]
        groups.append((i, j))
        j = i
    groups.reverse()

    group_of = np.empty(K, dtype=np.int64)
    merged = np.empty(len(groups), dtype=np.float64)
    for g, (a, b) in enumerate(groups):
        group_of[a:b] = g
        merged[g] = (MC[b] - MC[a]) / (M[b] - M[a])
    return group_of, merged


def build_merge_ladder(counts, centers, lambda_grid=None):
    """Solve the whole lambda sweep once, keeping one partition per attained size (Algorithm 3).

    The Lagrangian only reaches partitions on the lower convex hull of the
    rate/distortion curve, so some codebook sizes are not attainable. Callers request
    a target size and take the nearest rung (see `nearest_rung`).

    Returns:
        dict {n_merged_bins: (group_of_bin, merged_centers)}, ascending keys; each
        size keeps the partition of the smallest lambda that attains it.
    """
    grid = DEFAULT_LAMBDA_GRID if lambda_grid is None else np.asarray(lambda_grid, dtype=float)
    ladder = {}
    for lam in grid:
        group_of, merged = dp_merge(counts, centers, lam)
        ladder.setdefault(len(merged), (group_of, merged))
    return dict(sorted(ladder.items()))


def nearest_rung(ladder, k_target):
    """Rung of the ladder closest to k_target (ties go to the larger, i.e. less merged, size)."""
    sizes = np.fromiter(ladder.keys(), dtype=np.int64)
    best = int(sizes[np.argmin(np.abs(sizes - int(k_target)) - 1e-9 * sizes)])
    return best, ladder[best]
