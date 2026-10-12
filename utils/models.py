"""Pretrained models used in the paper.

* ImageNet-1K: torchvision checkpoints, pinned to IMAGENET1K_V1 so every
  architecture shares one training recipe and one evaluation resize (256).
  Any torchvision classification model name can be passed as ``arch``.
* CIFAR-10/100: torchvision ResNets with a 3x3 stride-1 stem and no max-pool,
  trained from scratch (``train_cifar.py``). The checkpoints are downloaded
  from the Hugging Face Hub on first use (see ``utils/hf_checkpoints.py``).
"""

import torchvision
from torch import nn

from utils.hf_checkpoints import ensure_local_checkpoint
from utils.utils import load_checkpoint

DEFAULT_IMAGENET_WEIGHTS = "IMAGENET1K_V1"


def create_model(arch, dataset, num_classes, model_path=None, weights=DEFAULT_IMAGENET_WEIGHTS):
    """Build `arch` for `dataset`, with pretrained weights.

    Args:
        arch: torchvision model name, e.g. 'resnet18', 'vgg16_bn', 'vit_b_16'
        dataset: 'cifar10', 'cifar100' or 'imagenet'
        num_classes: number of output classes (CIFAR only)
        model_path: CIFAR only. Directory holding '0model_SA_best.pth.tar';
            downloaded from the Hugging Face Hub if missing.
        weights: ImageNet only. torchvision weights name.
    """
    if dataset in ('cifar10', 'cifar100'):
        if not arch.startswith('resnet'):
            raise ValueError(f"CIFAR models are ResNets with a 3x3 stem, got '{arch}'")
        model = torchvision.models.get_model(arch, weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if model_path is not None:
            ensure_local_checkpoint(dataset, arch, model_path)
            print(f"Loading checkpoint from {model_path}")
            checkpoint = load_checkpoint(model_path, filename="model_SA_best.pth.tar")
            if checkpoint is None:
                raise FileNotFoundError(f"No checkpoint for {dataset}/{arch} in {model_path}")
            model.load_state_dict(checkpoint['state_dict'])
        return model

    init_weights = resolve_weights(arch, weights)
    print(f"Loading pretrained weights for {arch}: weights={init_weights}")
    return torchvision.models.get_model(arch, weights=init_weights)


def resolve_weights(arch, weights=DEFAULT_IMAGENET_WEIGHTS):
    """Resolve a torchvision weights name for `arch`, falling back to DEFAULT when absent.

    torchvision's "DEFAULT" picks the best available checkpoint per architecture,
    which is IMAGENET1K_V2 (a different recipe and eval resize) for resnet50/101.
    """
    if weights in (None, "DEFAULT"):
        return "DEFAULT"
    try:
        enum = torchvision.models.get_model_weights(arch)
    except Exception:
        return weights
    if getattr(enum, weights, None) is not None:
        return weights
    fallback = enum.DEFAULT.name
    print(f"WARNING: {arch} has no {weights}; falling back to DEFAULT ({fallback})")
    return fallback


def get_weights_preprocessing(arch, weights=DEFAULT_IMAGENET_WEIGHTS):
    """(resize_size, crop_size, reported top-1) of a torchvision checkpoint, or None.

    Reads the weights' metadata only; nothing is downloaded.
    """
    try:
        enum = torchvision.models.get_model_weights(arch)
    except Exception:
        return None
    resolved = resolve_weights(arch, weights)
    entry = enum.DEFAULT if resolved in (None, "DEFAULT") else getattr(enum, resolved, None)
    if entry is None:
        return None
    transform = entry.transforms()
    acc1 = entry.meta.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1")
    return int(transform.resize_size[0]), int(transform.crop_size[0]), acc1
