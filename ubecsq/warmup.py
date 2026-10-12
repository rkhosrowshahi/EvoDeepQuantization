"""Automatic search-range selection for K0 (Algorithm 4, Section III-D)."""

import numpy as np

from utils.compression import compress


def find_k_bounds(model, accuracy_fn, pretrained_val_acc, args):
    """Bisect log2(K0) for the two edges of the search range, using UBQ alone.

    Err(K0) is the relative accuracy loss of UBQ with K0 bins (Eq. 13), assumed
    non-increasing in K0. The two edges sit on opposite sides of their thresholds
    on purpose:

        K0_lb = largest K0 whose error is still ABOVE warmup_min_error (50%). The
                aggressive edge keeps the "still bad" side of the accuracy cliff in
                range, since a point just past the cliff can still be Pareto-optimal.
        K0_ub = smallest K0 whose error is AT OR BELOW warmup_max_error (0.5%). Past
                this point extra bins buy no new trade-off.

    The bisection is continuous in log2(K0) and stops within
    warmup_log_k_resolution (0.1, about 7% in K0). Both edges are then snapped
    outward to powers of two: K0_lb down and K0_ub up. By monotonicity this can
    only move each edge further past its threshold, never back across it.

    accuracy_fn(model) -> float is the calibration accuracy of a model.
    Returns (K0_lb, K0_ub).
    """
    if pretrained_val_acc <= 0:
        raise ValueError(
            f"find_k_bounds: pretrained accuracy is {pretrained_val_acc}, so the relative error "
            "is undefined. Check the model, dataset and checkpoint.")
    min_error = args.warmup_min_error
    max_error = args.warmup_max_error
    lo, hi = args.warmup_log_k_range
    resolution = args.warmup_log_k_resolution

    cache = {}  # rounded log2(K0) -> (error, K0 actually evaluated)

    def error_at(log_k):
        key = round(log_k, 6)
        if key not in cache:
            k = max(1, round(2 ** log_k))
            compressed_model, *_ = compress(k, model)
            acc = accuracy_fn(compressed_model)
            del compressed_model
            error = (pretrained_val_acc - acc) / pretrained_val_acc
            cache[key] = (error, k)
            print(f"    log2(K)={log_k:.4f} -> K={k}: error={error:.2%}")
        return cache[key][0]

    def bisect_boundary(threshold, a, b):
        """Narrow [a, b] to within `resolution`, keeping error(a) > threshold >= error(b)."""
        while b - a > resolution:
            mid = (a + b) / 2
            if error_at(mid) <= threshold:
                b = mid
            else:
                a = mid
        return a, b

    print(f"K-bounds warmup: bisecting log2(K) in [{lo}, {hi}] "
          f"(thresholds: min_bins<={min_error:.2%}, max_bins<={max_error:.2%})")
    err_lo, err_hi = error_at(lo), error_at(hi)

    if err_lo <= min_error:
        # Even the range floor is under the threshold: the floor is the answer.
        k_min_log = lo
    elif err_hi > min_error:
        print(f"  WARNING: error(K={cache[round(hi, 6)][1]})={err_hi:.1%} never drops to "
              f"{min_error:.2%}; using that K as min_bins. Consider raising warmup_log_k_range.")
        k_min_log = hi
    else:
        k_min_log, _ = bisect_boundary(min_error, lo, hi)

    if err_hi > max_error:
        print(f"  WARNING: error(K={cache[round(hi, 6)][1]})={err_hi:.1%} never drops to "
              f"{max_error:.2%}; using that K as max_bins. Consider raising warmup_log_k_range.")
        k_max_log = hi
    else:
        _, k_max_log = bisect_boundary(max_error, k_min_log, hi)
    if k_max_log <= k_min_log and k_min_log < hi:
        # Both thresholds met on a near-vertical cliff: give the upper edge some headroom.
        k_max_log = min(k_min_log + resolution, hi)
        error_at(k_max_log)

    # Snap outward to powers of two; evaluate them so the log reports the real error.
    k_min_pow2_log = int(np.floor(k_min_log))
    k_max_pow2_log = int(np.ceil(k_max_log))
    if k_max_pow2_log <= k_min_pow2_log:
        k_max_pow2_log = k_min_pow2_log + 1
    err_min_pow2 = error_at(k_min_pow2_log)
    err_max_pow2 = error_at(k_max_pow2_log)

    k_min, k_max = 2 ** k_min_pow2_log, 2 ** k_max_pow2_log
    print(f"K-bounds warmup result: min_bins={k_min} (error~{err_min_pow2:.2%}), "
          f"max_bins={k_max} (error~{err_max_pow2:.2%}), {len(cache)} evals")
    return k_min, k_max
