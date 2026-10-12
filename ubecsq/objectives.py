"""The two objectives (Eqs. 14-15) and the hypervolume of a front."""

import numpy as np
from pymoo.indicators.hv import HV


def objectives(accuracy, compression_ratio):
    """(f1, f2) = (1 - top-1 accuracy, 1 / CR), both minimized."""
    f1 = 1 - accuracy
    f2 = 1 / compression_ratio if compression_ratio > 0 else float('inf')
    return f1, f2


def hv_bounded(F, ref_point=None):
    """Hypervolume of objective vectors `F` against a fixed reference point.

    Both objectives live in [0, 1] (1 - accuracy, and 1/CR since compression never
    expands the model), so the reference point (1, 1) bounds the hypervolume to
    [0, 1] for every run. Points at or beyond the reference contribute zero volume.
    """
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return 0.0
    ref = np.ones(F.shape[1]) if ref_point is None else np.asarray(ref_point, dtype=float)
    return float(HV(ref_point=ref)(np.minimum(F, ref)))
