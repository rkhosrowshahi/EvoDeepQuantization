"""Seeding, device selection, evaluation, and checkpoint helpers."""

import os
import random
import shutil

import numpy as np
import torch


def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(args):
    """'cpu', 'cuda' or 'cuda:<gpu>' from ``args.device`` / ``args.gpu``."""
    if args.device == 'cpu':
        return "cpu"
    if args.device != 'cuda':
        raise ValueError(f"Invalid device: {args.device}. Choose 'cpu' or 'cuda'")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available; running on CPU.")
        return "cpu"
    if args.gpu == -1:
        return "cuda"
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU device id {args.gpu} is not available. "
                         f"Available devices: {torch.cuda.device_count()}")
    return f"cuda:{args.gpu}"


def evaluate(model, data_loader, device="cuda"):
    """Top-1 accuracy of `model` over a data loader."""
    if isinstance(data_loader, CachedEvalSet):
        return data_loader.accuracy(model)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


class CachedEvalSet:
    """A dataset decoded once and kept resident on the device.

    Every candidate is scored on the same calibration images, so they are
    decoded and moved to the device once; each evaluation is then a pure
    forward pass. Images are stored in ``dtype`` (fp16 by default, halving
    memory) and cast to the model's parameter dtype at forward time.
    """

    def __init__(self, data_loader, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.batches = [
            (images.to(device=device, dtype=dtype, non_blocking=True),
             labels.to(device=device, non_blocking=True))
            for images, labels in data_loader
        ]
        if not self.batches:
            raise ValueError("Cannot cache an empty data loader")
        self.num_samples = sum(int(labels.numel()) for _, labels in self.batches)

    def __len__(self):
        return self.num_samples

    @property
    def nbytes(self):
        return sum(int(images.numel()) * images.element_size() for images, _ in self.batches)

    def accuracy(self, model):
        """Top-1 accuracy over the whole cached set. Deterministic."""
        model.eval()
        correct = 0
        with torch.no_grad():
            param_dtype = next(model.parameters()).dtype
            for images, labels in self.batches:
                if images.dtype != param_dtype:
                    images = images.to(param_dtype)
                predicted = model(images).argmax(dim=1)
                correct += (predicted == labels).sum().item()
        return correct / self.num_samples


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Precision@k (in %) for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, save_path, filename="checkpoint.pth.tar"):
    """Save `state` as 0<filename>, and copy it to 0model_SA_best.pth.tar when `is_best`."""
    filepath = os.path.join(save_path, "0" + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, "0model_SA_best.pth.tar"))


def load_checkpoint(save_path, filename="checkpoint.pth.tar"):
    """Load 0<filename> from `save_path` onto the CPU, or None if it does not exist."""
    filepath = os.path.join(save_path, "0" + filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, map_location="cpu")
    print("Checkpoint not found! path:{}".format(filepath))
    return None
