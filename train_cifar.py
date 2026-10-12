#!/usr/bin/env python3
"""Train the CIFAR-10/100 ResNet-18 checkpoints from scratch.

Not needed to reproduce the paper: the trained checkpoints are downloaded from
the Hugging Face Hub automatically. The checkpoint with the best calibration
accuracy is saved as <save_dir>/0model_SA_best.pth.tar, the file main.py loads
from ``model_path``.

    python train_cifar.py --dataset cifar10 --arch resnet18 --save_dir ./checkpoints/cifar10/resnet18
"""

import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from utils.config import parse_train_args  # noqa: E402
from utils.dataset import get_dataset  # noqa: E402
from utils.models import create_model  # noqa: E402
from utils.utils import AverageMeter, accuracy, resolve_device, save_checkpoint, setup_seed  # noqa: E402


def train(train_loader, model, criterion, optimizer, epoch, args, device):
    """Train the model for one epoch."""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))

        if batch_idx % args.print_freq == 0:
            print(f"Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})")
    return top1.avg / 100.0


def validate(val_loader, model, criterion, args, device):
    """Top-1 accuracy on a held-out split."""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))

            if batch_idx % args.print_freq == 0:
                print(f"Test: [{batch_idx}/{len(val_loader)}]\t"
                      f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                      f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})")

    print(f" * Prec@1 {top1.avg:.3f}")
    return top1.avg / 100.0


def main():
    args = parse_train_args()
    if args.dataset not in ("cifar10", "cifar100"):
        raise ValueError("train_cifar.py trains CIFAR-10/100 models; ImageNet models come from torchvision")

    # The calibration split is drawn from this seed: keep it equal to the data_seed used later.
    setup_seed(args.data_seed)
    device = resolve_device(args)
    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset, val_dataset, test_dataset, num_classes, _ = get_dataset(args.dataset, args.data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = create_model(args.arch, args.dataset, num_classes).to(device)
    print(f"train / calibration / test images: "
          f"{len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    milestones = list(map(int, args.decreasing_lr.split(",")))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_sa = 0
    start_epoch = 0
    all_result = {"train_ta": [], "val_ta": [], "test_ta": []}
    if args.resume:
        print(f"resume from checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        best_sa = checkpoint["best_sa"]
        start_epoch = checkpoint["epoch"]
        all_result = checkpoint["result"]
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("loading from epoch:", start_epoch, "best_sa=", best_sa)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(f"Epoch #{epoch}, Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        acc = train(train_loader, model, criterion, optimizer, epoch, args, device)
        tacc = validate(val_loader, model, criterion, args, device)
        scheduler.step()

        all_result["train_ta"].append(acc)
        all_result["val_ta"].append(tacc)
        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)

        save_checkpoint(
            {
                "result": all_result,
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_sa": best_sa,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=is_best_sa,
            save_path=args.save_dir,
        )
        print(f"one epoch duration:{time.time() - start_time}")

    plt.plot(all_result["train_ta"], label="train_acc")
    plt.plot(all_result["val_ta"], label="val_acc")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "0net_train.pdf"))
    plt.close()

    print("Performance on the test set")
    test_tacc = validate(test_loader, model, criterion, args, device)
    all_result["test_ta"].append(test_tacc)
    if len(all_result["val_ta"]) != 0:
        best_epoch = int(np.argmax(np.array(all_result["val_ta"])))
        print(f"* best SA = {all_result['val_ta'][best_epoch]}, Epoch = {best_epoch + 1}")
    print(f"* Test accuracy = {test_tacc:.4f}")


if __name__ == "__main__":
    main()
