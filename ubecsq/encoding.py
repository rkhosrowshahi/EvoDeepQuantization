"""Genotype of the bin count K0.

The gene t lives in [log2(min_bins), log2(max_bins)] and decodes to
K0 = round(2**t). The second objective (1/CR) is close to linear in log2(K0),
so a gene that is uniform in log2 spreads solutions evenly across index widths.

Genes are canonical: every gene is snapped to the exact representative of the
K0 it decodes to, so pymoo's duplicate elimination sees two individuals with
the same K0 as identical.
"""

import numpy as np
from pymoo.core.repair import Repair


def gene_bounds(args):
    """(lower, upper) bound of the K0 gene."""
    if args.min_bins < 1:
        raise ValueError(f"min_bins must be >= 1, got {args.min_bins}")
    return float(np.log2(args.min_bins)), float(np.log2(args.max_bins))


def decode_n_bins(gene, args):
    """Decode gene value(s) to integer bin count(s), clipped to [min_bins, max_bins]."""
    gene_array = np.asarray(gene, dtype=float)
    n_bins = np.rint(np.power(2.0, gene_array))
    n_bins = np.clip(n_bins, args.min_bins, args.max_bins).astype(int)
    return n_bins if gene_array.ndim else int(n_bins)


def encode_n_bins(n_bins):
    """Canonical gene value(s) for the given bin count(s)."""
    n_bins_array = np.asarray(n_bins, dtype=float)
    gene = np.log2(n_bins_array)
    return gene if n_bins_array.ndim else float(gene)


def canonicalize_genes(X, args):
    """Snap column 0 of X onto the canonical gene of the K0 it decodes to."""
    X = np.asarray(X, dtype=float).copy()
    X[:, 0] = encode_n_bins(decode_n_bins(X[:, 0], args))
    return X


class NBinsRepair(Repair):
    """Canonicalize the K0 gene inside pymoo's mating loop.

    pymoo applies repair before duplicate elimination, so snapping here is what
    makes duplicate elimination work on a gene that is continuous in the
    genotype but discrete in the phenotype.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

    def _do(self, problem, X, **kwargs):
        return canonicalize_genes(X, self.args)
