#!/usr/bin/env python3
"""Uniform Quantization baseline (Choi et al. [6]): UBQ alone.

Equal-width bins with mean centroids and empty bins dropped, at K0 = 2, 4, ..., 2048;
no merging and no search. The gap between this curve and the UB-ECSQ front is the
combined gain of entropy-constrained merging and the (K0, rho) search.

    python baselines/run_uniform.py --dataset cifar10
    python baselines/run_uniform.py --dataset imagenet --arch resnet50
"""

import argparse

from common import add_common_args, setup
from utils.compression import compress

K_LIST = [2 ** b for b in range(1, 12)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--k_list", type=int, nargs="+", default=K_LIST, help="bin counts K0")
    cli = parser.parse_args()

    s = setup(cli)
    rows = []
    for k in cli.k_list:
        compressed, centers, _ = compress(k, s.model)
        rows.append(s.score(compressed, n_bins=k, n_bins_realized=len(centers)))
        del compressed
    s.save(rows, f"uniform_{s.stem}.csv")


if __name__ == "__main__":
    main()
