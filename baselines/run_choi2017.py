#!/usr/bin/env python3
"""Hessian K-means and Iterative ECSQ baselines (Choi et al. [6]); see choi2017.py.

* Hessian K-means: one global codebook at K = 4, 8, ..., 1024.
* Iterative ECSQ: Algorithm 1 of Choi et al., warm-started from the Hessian
  K-means solution at K = 1024 and swept over the Lagrange multiplier lambda
  (lambda = 0 is Hessian K-means at K = 1024). The CIFAR-10 grid was bracketed by
  hand; elsewhere it is bracketed automatically on the model's own scale.

The Hessian diagonal is the empirical Fisher diagonal from 500 calibration images
on CIFAR-10 and 200 elsewhere.

    python baselines/run_choi2017.py --dataset cifar10
    python baselines/run_choi2017.py --dataset imagenet --arch resnet18
"""

import argparse
import time

from common import add_common_args, setup
from choi2017 import bracket_lambda, fisher_diagonal, hessian_weighted_kmeans, quantize
from utils.compression import get_model_params

K_LIST = [2 ** b for b in range(2, 11)]
CIFAR10_LAMBDA_LIST = [0.0, 1e-7, 3e-7, 9e-7, 2.7e-6, 8.1e-6, 2.43e-5, 7.29e-5, 2.19e-4, 6.56e-4, 1.97e-3, 5.9e-3]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--k_list", type=int, nargs="+", default=K_LIST,
                        help="Hessian K-means codebook sizes; the largest is the Iterative ECSQ warm start")
    parser.add_argument("--lambda_list", type=float, nargs="+", default=None,
                        help="Iterative ECSQ lambdas (default: the CIFAR-10 grid, or bracketed automatically)")
    parser.add_argument("--fisher_samples", type=int, default=None,
                        help="calibration images for the Fisher diagonal (default: 500 on CIFAR-10, else 200)")
    parser.add_argument("--skip_hessian_kmeans", action="store_true")
    parser.add_argument("--skip_iterative_ecsq", action="store_true")
    cli = parser.parse_args()
    fisher_samples = cli.fisher_samples or (500 if cli.dataset == "cifar10" else 200)

    s = setup(cli)
    t0 = time.time()
    h_diag = fisher_diagonal(s.model, s.calibration.batches, device=s.device, max_samples=fisher_samples)
    w = get_model_params(s.model)
    print(f"Fisher diagonal: {len(h_diag)} weights, {fisher_samples} calibration images [{time.time() - t0:.1f}s]  "
          f"mean={h_diag.mean():.3e}  max={h_diag.max():.3e}  zero-fraction={(h_diag == 0).mean():.3f}")

    if not cli.skip_hessian_kmeans:
        rows = []
        for k in cli.k_list:
            compressed, centers, _, _ = quantize(s.model, "hessian_kmeans", k=k, h_diag=h_diag)
            rows.append(s.score(compressed, n_bins=k, n_bins_realized=len(centers)))
            del compressed
        s.save(rows, f"choi2017_hessian_kmeans_{s.stem}.csv")

    if not cli.skip_iterative_ecsq:
        k_init = max(cli.k_list)
        print(f"  [iterative_ecsq] warm-starting from hessian_weighted_kmeans(k={k_init})...")
        warm_centers, _ = hessian_weighted_kmeans(w, h_diag, k_init, init="uniform")

        lambda_list = cli.lambda_list
        if lambda_list is None:
            if cli.dataset == "cifar10":
                lambda_list = CIFAR10_LAMBDA_LIST
            else:
                lambda_list = bracket_lambda(w, h_diag, warm_centers, k_init, s.device)

        rows = []
        for lam in lambda_list:
            compressed, centers, _, extra = quantize(
                s.model, "iterative_ecsq", k=k_init, h_diag=h_diag, lam=lam, device=s.device,
                init_centers=warm_centers, max_iter=100, tol=1e-6)
            rows.append(s.score(compressed, lam=lam, k_init=k_init, n_bins_realized=len(centers),
                                n_iter=extra["n_iter"], converged=extra["converged"]))
            del compressed
        s.save(rows, f"choi2017_iterative_ecsq_{s.stem}.csv")


if __name__ == "__main__":
    main()
