"""CIFAR-10/100 and ImageNet-1K with the calibration/test splits used in the paper.

The splits are drawn from NumPy's global RNG, so call ``setup_seed(data_seed)``
first; every entry point does. For CIFAR, ``data_seed`` must match the seed the
checkpoint was trained with (the default, 2). Otherwise the calibration split
overlaps the training images and accuracy is inflated.

* CIFAR-10/100: a class-balanced 10% of the training set (5,000 images) is held
  out as the calibration split; the 10,000 test images are kept for testing.
* ImageNet-1K: the 50,000 validation images are split class-balanced into a
  calibration split (``calibration_size``, 5,000 in the paper) and a disjoint
  test split (``test_size``, 40,000 in the paper).
"""

import numpy as np
import torchvision
from torch.utils.data import Subset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_CIFAR = {
    'cifar10': (torchvision.datasets.CIFAR10, 10, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'cifar100': (torchvision.datasets.CIFAR100, 100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def _balanced_indices(targets, n_per_class, num_classes):
    """`n_per_class` random indices of each class, in class order."""
    indices = []
    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        cls_idx = np.random.permutation(cls_idx)
        n = min(n_per_class, len(cls_idx))
        indices.append(cls_idx[:n])
    return np.concatenate(indices)


def _cifar(dataset_name, data_dir):
    dataset_cls, num_classes, mean, std = _CIFAR[dataset_name]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train_dataset = dataset_cls(root=data_dir, train=True, download=True, transform=None)
    targets = np.array(full_train_dataset.targets)

    # Class-balanced 10% of the training set is the calibration split.
    val_per_class = int((len(full_train_dataset) * 0.1) // num_classes)
    val_indices = _balanced_indices(targets, val_per_class, num_classes)
    train_indices = np.setdiff1d(np.arange(len(full_train_dataset)), val_indices)

    train_dataset = Subset(
        dataset_cls(root=data_dir, train=True, download=False, transform=transform_train), train_indices)
    val_dataset = Subset(
        dataset_cls(root=data_dir, train=True, download=False, transform=transform_eval), val_indices)
    test_dataset = dataset_cls(root=data_dir, train=False, download=True, transform=transform_eval)
    return train_dataset, val_dataset, test_dataset, num_classes, 32


def _imagenet(data_dir, calibration_size, test_size, resize_size):
    num_classes = 1000
    if calibration_size + test_size > 50000:
        raise ValueError(
            f"calibration_size + test_size = {calibration_size + test_size} "
            f"exceeds the 50000 available ImageNet validation images")

    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    print(f"ImageNet eval transform: Resize({resize_size}) -> CenterCrop(224)")
    full_val_dataset = torchvision.datasets.ImageNet(root=data_dir, split='val', transform=transform)

    if hasattr(full_val_dataset, "targets"):
        targets = np.array(full_val_dataset.targets)
    else:
        targets = np.array([t[1] for t in full_val_dataset.samples])

    val_per_class = calibration_size // num_classes
    test_per_class = test_size // num_classes
    print(f"ImageNet split: calibration {val_per_class * num_classes} images ({val_per_class}/class), "
          f"test {test_per_class * num_classes} images ({test_per_class}/class)")

    val_indices = _balanced_indices(targets, val_per_class, num_classes)
    test_indices = []
    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        cls_idx = np.random.permutation(cls_idx)
        val_cls_idx = val_indices[np.where(targets[val_indices] == cls)[0]]
        remaining_cls_idx = [idx for idx in cls_idx if idx not in val_cls_idx]
        n = min(test_per_class, len(remaining_cls_idx))
        test_indices.extend(remaining_cls_idx[:n])
    test_indices = np.array(test_indices)

    val_dataset = Subset(full_val_dataset, val_indices)
    test_dataset = Subset(full_val_dataset, test_indices)
    # There is no training set for ImageNet: the models are torchvision checkpoints.
    return val_dataset, val_dataset, test_dataset, num_classes, 224


def get_dataset(dataset_name, data_dir='./data', calibration_size=5000, test_size=40000, resize_size=256):
    """Returns (train_dataset, calibration_dataset, test_dataset, num_classes, input_size).

    calibration_size, test_size, resize_size: ImageNet only. resize_size is the
    shorter-side resize before the 224 centre crop and belongs to the checkpoint
    (256 for every IMAGENET1K_V1 model used in the paper).
    """
    name = dataset_name.lower()
    if name in _CIFAR:
        return _cifar(name, data_dir)
    if name == 'imagenet':
        return _imagenet(data_dir, calibration_size, test_size, resize_size)
    raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'cifar10', 'cifar100' or 'imagenet'")
