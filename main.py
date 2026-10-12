#!/usr/bin/env python3
"""Run UB-ECSQ-NSGA2 on one (dataset, architecture, seed).

    python main.py --config configs/cifar10/resnet18/seed1.yaml
    python main.py --config configs/imagenet/resnet50/seed3.yaml --data_dir /path/to/imagenet --gpu 1

Writes <save_dir>/final_pareto_front.csv, the front after every generation
(<save_dir>/steps/), the hypervolume history, figures, and run.log.
"""

import os

import matplotlib

matplotlib.use("Agg")

from ubecsq import run_ecsq  # noqa: E402
from ubecsq.logging_utils import tee_stdio_to_file  # noqa: E402
from utils.config import parse_args  # noqa: E402


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "run.log")
    with tee_stdio_to_file(log_path):
        print(f"Logging output to: {log_path}")
        run_ecsq(args)


if __name__ == "__main__":
    main()
