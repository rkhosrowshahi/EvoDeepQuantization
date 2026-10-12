#!/usr/bin/env python3
"""Layer-wise weight-sharing baseline of Dupuis et al. [17]; see dupuis2022.py.

Settings used in the paper: per-layer k grid 2^1..2^8, candidates must keep 98%
of the uncompressed accuracy, NSGA-II with a population of 100 for 50 generations
on CIFAR-10/100 and 10 generations on ImageNet. CNNs only (no ViT-B/16).

    python baselines/run_dupuis2022.py --dataset cifar10
    python baselines/run_dupuis2022.py --dataset imagenet --arch resnet34

This is the most expensive baseline: every candidate is clustered layer by layer
and evaluated, and Level 1 alone probes every layer at every k (about 32 hours
for CIFAR-10 / ResNet-18 at the paper's settings).
"""

import argparse
import time

from common import add_common_args, setup
from dupuis2022 import cluster_layers, run_dupuis


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--max_bit_range", type=int, default=8, help="per-layer k grid spans 2^1..2^max_bit_range")
    parser.add_argument("--min_acc_fraction", type=float, default=0.98,
                        help="Level-1 candidates keep at least this fraction of the uncompressed accuracy")
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--n_gen", type=int, default=None, help="default: 50 on CIFAR, 10 on ImageNet")
    parser.add_argument("--seed", type=int, default=0)
    cli = parser.parse_args()
    if cli.arch.startswith("vit"):
        parser.error("the layer-wise method of Dupuis et al. is not applicable to ViT models")
    n_gen = cli.n_gen or (10 if cli.dataset == "imagenet" else 50)

    s = setup(cli)
    t0 = time.time()
    pareto_df, _, layer_df = run_dupuis(
        s.model, s.calibration, s.baseline_val_acc,
        min_acc_fraction=cli.min_acc_fraction, max_bit_range=cli.max_bit_range,
        pop_size=cli.pop_size, n_gen=n_gen, seed=cli.seed)
    print(f"[dupuis2022] total time: {time.time() - t0:.1f}s")

    s.save(layer_df, f"dupuis2022_layer_exploration_{s.stem}.csv")
    if cli.eval_test:
        scores = []
        for k_vector in pareto_df["k_vector"]:
            compressed, _ = cluster_layers(s.model, k_vector)
            scores.append(s.test_accuracy(compressed))
            del compressed
        pareto_df["test_eval_score"] = scores
    s.save(pareto_df, f"dupuis2022_{s.stem}.csv")
    print(pareto_df[["val_eval_score", "cr_raw", "cr_huffman"]].to_string(index=False))


if __name__ == "__main__":
    main()
