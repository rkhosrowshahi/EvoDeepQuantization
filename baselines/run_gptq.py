#!/usr/bin/env python3
"""GPTQ baseline (Frantar et al. [18]); see gptq.py.

Every Conv2d/Linear layer is quantized with asymmetric per-output-channel scales and
no grouping, calibrated on 4 batches of the calibration split (2,048 images on
CIFAR, 1,024 on ImageNet). The CR counts the fixed-width codes plus a 16-bit scale
and zero point per output channel, without Huffman coding. CNNs only (no ViT-B/16).

    python baselines/run_gptq.py --dataset cifar10 --bits 4      # the paper ran 4 bits on CIFAR-10
    python baselines/run_gptq.py --dataset cifar100              # 2, 3, 4, 6 and 8 bits
    python baselines/run_gptq.py --dataset imagenet --arch resnet101
"""

import argparse
import copy
import time

import torch

from common import add_common_args, setup
from gptq import gptq_compressed_bits, gptq_quantize_network

BITS_LIST = [2, 3, 4, 6, 8]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--bits", type=int, nargs="+", default=BITS_LIST, help="bit widths")
    parser.add_argument("--group_size", type=int, default=-1, help="-1: one scale/zero per output channel")
    parser.add_argument("--percdamp", type=float, default=0.01, help="Hessian dampening")
    parser.add_argument("--act_order", action="store_true", help="quantize columns in decreasing Hessian order")
    parser.add_argument("--sym", action="store_true", help="symmetric quantization (default: asymmetric)")
    parser.add_argument("--calib_batches", type=int, default=4, help="calibration batches")
    cli = parser.parse_args()
    if cli.arch.startswith("vit"):
        parser.error("this GPTQ port covers Conv2d/Linear layers only, not ViT's fused attention projection")

    s = setup(cli)
    calib_batches = s.calibration.batches[:cli.calib_batches]
    print(f"[gptq] calibrating on {sum(images.shape[0] for images, _ in calib_batches)} images")

    rows = []
    for bits in cli.bits:
        t0 = time.time()
        model = copy.deepcopy(s.model)  # GPTQ works in place: always start from the pretrained weights
        stats = gptq_quantize_network(model, calib_batches, bits, group_size=cli.group_size,
                                      percdamp=cli.percdamp, actorder=cli.act_order, sym=cli.sym)
        row = {"bits": bits, "group_size": cli.group_size, "val_eval_score": s.calibration.accuracy(model)}
        if cli.eval_test:
            row["test_eval_score"] = s.test_accuracy(model)
        _, _, bits_per_weight, cr = gptq_compressed_bits(stats, bits, model=model, sym=cli.sym)
        row.update(bits_per_weight=bits_per_weight, cr_raw=cr)
        print(f"  [gptq] bits={bits}  val={row['val_eval_score']:.4f}  bits_per_weight={bits_per_weight:.3f}  "
              f"cr_raw={cr:.2f}  [{time.time() - t0:.1f}s]")
        rows.append(row)
        del model
        torch.cuda.empty_cache()
    s.save(rows, f"gptq_{s.stem}.csv")


if __name__ == "__main__":
    main()
