"""Shared setup of the baseline sweeps.

Every baseline uses the checkpoint, preprocessing and calibration split of
UB-ECSQ (read from configs/<dataset>/<arch>/seed1.yaml), is scored on the
calibration split, and uses the same storage accounting.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# utils.compression sets OMP_NUM_THREADS before scikit-learn is first imported.
from utils.compression import compression_ratios  # noqa: E402
from utils import CachedEvalSet, create_model, evaluate, get_dataset, resolve_device, setup_seed  # noqa: E402
from utils.config import parse_args  # noqa: E402
from utils.models import get_weights_preprocessing  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "baselines"


def add_common_args(parser):
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument("--arch", default="resnet18",
                        help="resnet18 for CIFAR; resnet18|resnet34|resnet50|resnet101|vgg16_bn|vit_b_16 for ImageNet")
    parser.add_argument("--data_dir", default=None, help="override the config's data_dir")
    parser.add_argument("--model_path", default=None, help="override the config's model_path (CIFAR)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id")
    parser.add_argument("--eval_test", action="store_true", help="also report accuracy on the test split")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="where the CSVs are written")


@dataclass
class Setup:
    args: object
    cli: object
    device: str
    model: torch.nn.Module
    calibration: CachedEvalSet
    test_loader: object
    baseline_val_acc: float
    baseline_test_acc: float

    @property
    def stem(self):
        """File-name stem of this setting: cifar10, cifar100 or imagenet_<arch>."""
        return self.args.dataset if self.args.dataset != "imagenet" else f"imagenet_{self.args.arch}"

    def test_accuracy(self, model):
        return evaluate(model, self.test_loader, device=self.device)

    def score(self, model, **columns):
        """One CSV row for a codebook-quantized model: accuracy and both compression ratios."""
        row = dict(columns)
        row["val_eval_score"] = self.calibration.accuracy(model)
        if self.cli.eval_test:
            row["test_eval_score"] = self.test_accuracy(model)
        ratios = compression_ratios(model)
        row["cr_raw"] = ratios["cr_raw"]
        row["cr_huffman"] = ratios["cr_huffman"]
        row["bits_per_weight"] = 32.0 / ratios["cr_huffman"]
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        return row

    def save(self, rows, filename):
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        df["baseline_val_acc"] = self.baseline_val_acc
        if self.cli.eval_test:
            df["baseline_test_acc"] = self.baseline_test_acc
        out_dir = Path(self.cli.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        df.to_csv(path, index=False)
        print(f"wrote {path}")
        return path


def setup(cli):
    """Load the model and calibration split of (cli.dataset, cli.arch) like main.py does."""
    config = PROJECT_ROOT / "configs" / cli.dataset / cli.arch / "seed1.yaml"
    if not config.exists():
        raise FileNotFoundError(f"No config for {cli.dataset}/{cli.arch}: {config}")
    argv = ["--config", str(config)]
    for flag in ("data_dir", "model_path", "gpu"):
        value = getattr(cli, flag)
        if value is not None:
            argv += [f"--{flag}", str(value)]
    args = parse_args(argv)
    device = resolve_device(args)
    setup_seed(args.data_seed)

    resize_size = 256
    if args.dataset == "imagenet":
        meta = get_weights_preprocessing(args.arch, args.weights)
        if meta is not None:
            resize_size = meta[0]
    _, val_dataset, test_dataset, num_classes, _ = get_dataset(
        args.dataset, args.data_dir, calibration_size=args.calibration_size,
        test_size=args.test_size, resize_size=resize_size)
    model = create_model(args.arch, args.dataset, num_classes, model_path=args.model_path,
                         weights=args.weights).to(device)
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    calibration = CachedEvalSet(val_loader, device=device)
    baseline_val_acc = calibration.accuracy(model)

    test_loader, baseline_test_acc = None, float("nan")
    if cli.eval_test:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers)
        baseline_test_acc = evaluate(model, test_loader, device=device)
    print(f"[{args.dataset}/{args.arch}] uncompressed calibration accuracy: {baseline_val_acc:.4f} "
          f"({len(calibration)} images)"
          + (f", test accuracy: {baseline_test_acc:.4f}" if cli.eval_test else ""))
    return Setup(args, cli, device, model, calibration, test_loader, baseline_val_acc, baseline_test_acc)
