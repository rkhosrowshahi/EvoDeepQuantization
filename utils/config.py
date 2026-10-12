"""Command-line arguments and YAML configs.

A YAML config supplies defaults; any flag given on the command line overrides it:

    python main.py --config configs/cifar10/resnet18/seed1.yaml --gpu 1
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(config):
    """Find `config` as given, or relative to the project root."""
    path = Path(config)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Config file not found: {config}")


def _add_data_args(parser):
    group = parser.add_argument_group("data and model")
    group.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet"])
    group.add_argument("--arch", type=str, default="resnet18", help="torchvision model name")
    group.add_argument("--data_dir", type=str, default="./data", help="dataset root")
    group.add_argument("--model_path", type=str, default=None,
                       help="CIFAR checkpoint directory (downloaded from the Hugging Face Hub if empty)")
    group.add_argument("--weights", type=str, default="IMAGENET1K_V1", help="ImageNet: torchvision weights")
    group.add_argument("--calibration_size", type=int, default=5000,
                       help="ImageNet: validation images used as the calibration split")
    group.add_argument("--test_size", type=int, default=40000,
                       help="ImageNet: validation images held out as the test split")
    group.add_argument("--data_seed", type=int, default=2,
                       help="seed of the calibration/test split (CIFAR: must match the checkpoint, 2)")


def _add_system_args(parser):
    group = parser.add_argument_group("system")
    group.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    group.add_argument("--gpu", type=int, default=0, help="GPU id (-1: default CUDA device)")
    group.add_argument("--batch_size", type=int, default=256, help="evaluation batch size")
    group.add_argument("--workers", type=int, default=8, help="data loader workers")


def build_parser():
    parser = argparse.ArgumentParser(
        description="UB-ECSQ-NSGA2: multi-objective evolutionary weight-sharing compression.")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config; command-line flags override its values")
    _add_data_args(parser)
    _add_system_args(parser)

    search = parser.add_argument_group("search (Algorithm 1)")
    search.add_argument("--quantizer_seed", type=int, default=1, help="seed of the evolutionary search")
    search.add_argument("--steps", type=int, default=10, help="NSGA-II generations G")
    search.add_argument("--np", type=int, default=100, help="population size NP")
    search.add_argument("--crossover_prob", type=float, default=0.9, help="SBX probability")
    search.add_argument("--crossover_prob_var", type=float, default=1.0, help="SBX per-variable probability")
    search.add_argument("--crossover_eta", type=int, default=15, help="SBX distribution index")
    search.add_argument("--mutation_prob", type=float, default=0.9, help="polynomial mutation probability")
    search.add_argument("--mutation_prob_var", type=float, default=1.0,
                        help="polynomial mutation per-variable probability")
    search.add_argument("--mutation_eta", type=int, default=10, help="polynomial mutation distribution index")
    search.add_argument("--ecsq_lambda_grid", type=float, nargs="+", default=None,
                        help="lambda values of the merge ladder (default: 80 values in [1e-10, 1e-3])")
    search.add_argument("--hv_ref_point", type=float, nargs="+", default=[1.0, 1.0],
                        help="hypervolume reference point for (1 - accuracy, 1/CR)")

    bounds = parser.add_argument_group("bin-count range (Algorithm 4)")
    bounds.add_argument("--auto_k_bounds", action=argparse.BooleanOptionalAction, default=True,
                        help="set [K0_lb, K0_ub] by the bisection warm-up; with --no-auto_k_bounds, "
                             "use --min_bins/--max_bins")
    bounds.add_argument("--min_bins", type=int, default=2, help="K0 lower bound without the warm-up")
    bounds.add_argument("--max_bins", type=int, default=1024, help="K0 upper bound without the warm-up")
    bounds.add_argument("--warmup_min_error", type=float, default=0.50,
                        help="relative accuracy loss defining K0_lb (epsilon_min)")
    bounds.add_argument("--warmup_max_error", type=float, default=0.005,
                        help="relative accuracy loss defining K0_ub (epsilon_max)")
    bounds.add_argument("--warmup_log_k_range", type=float, nargs=2, default=[1, 13],
                        help="log2(K0) interval the warm-up bisects")
    bounds.add_argument("--warmup_log_k_resolution", type=float, default=0.1,
                        help="bisection resolution in log2(K0)")

    output = parser.add_argument_group("output")
    output.add_argument("--save_dir", type=str, default=None,
                        help="output directory (default: ./results/<dataset>/<arch>/seed<quantizer_seed>)")
    output.add_argument("--save_interval", type=int, default=1, help="save the front every N generations")
    output.add_argument("--eval_test", action="store_true",
                        help="after the search, evaluate every Pareto solution on the test split")
    output.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    output.add_argument("--wandb_project", type=str, default="EvoDeepQuantization")
    output.add_argument("--wandb_entity", type=str, default=None)
    output.add_argument("--wandb_name", type=str, default=None)
    return parser


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = build_parser()
    if pre_args.config:
        config_path = resolve_config_path(pre_args.config)
        config = load_yaml_config(config_path)
        known = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - known)
        if unknown:
            raise ValueError(f"Unknown keys in {config_path}: {unknown}")
        parser.set_defaults(**{k: v for k, v in config.items() if k != "config"})
        print(f"Loaded config: {config_path}")

    args = parser.parse_args(argv)
    if args.save_dir is None:
        args.save_dir = f"./results/{args.dataset}/{args.arch}/seed{args.quantizer_seed}"
    return args


def parse_train_args():
    parser = argparse.ArgumentParser(description="Train a ResNet on CIFAR-10/100 from scratch.")
    _add_data_args(parser)
    _add_system_args(parser)
    parser.add_argument("--save_dir", type=str, required=True, help="where checkpoints are written")
    parser.add_argument("--resume", action="store_true", help="resume from --checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file to resume from")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--epochs", default=182, type=int, help="number of epochs")
    parser.add_argument("--decreasing_lr", default="91,136", help="learning-rate milestones (x0.1)")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    return parser.parse_args()
