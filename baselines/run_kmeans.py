#!/usr/bin/env python3
"""K-means baseline in the style of Deep Compression (Han et al. [23]).

One global K-means codebook shared across all layers (k-means++ initialization,
10 restarts, 50 Lloyd iterations), at K = 4, 8, ..., 1024. No pruning and no
retraining of the centroids.

    python baselines/run_kmeans.py --dataset cifar10
    python baselines/run_kmeans.py --dataset imagenet --arch vgg16_bn

K-means on the full weight vector is slow on large models (VGG16-BN has 138M weights).
"""

import argparse

from common import add_common_args, setup
from utils.compression import compress

K_LIST = [2 ** b for b in range(2, 11)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--k_list", type=int, nargs="+", default=K_LIST, help="codebook sizes K")
    parser.add_argument("--n_init", type=int, default=10, help="K-means restarts")
    parser.add_argument("--max_iter", type=int, default=50, help="Lloyd iterations per restart")
    cli = parser.parse_args()

    s = setup(cli)
    rows = []
    for k in cli.k_list:
        compressed, centers, _ = compress(k, s.model, method="kmeans", n_init=cli.n_init, max_iter=cli.max_iter)
        rows.append(s.score(compressed, n_bins=k, n_bins_realized=len(centers)))
        del compressed
    s.save(rows, f"kmeans_{s.stem}.csv")


if __name__ == "__main__":
    main()
