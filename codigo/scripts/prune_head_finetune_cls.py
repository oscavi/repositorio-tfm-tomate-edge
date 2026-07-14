import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune the YOLO classification head channels and run a short recovery fine-tune."
    )
    parser.add_argument("--model", required=True, help="Input .pt model")
    parser.add_argument("--data", required=True, help="Dataset root with train/val class folders")
    parser.add_argument("--outdir", required=True, help="Output directory for pruned artifacts")
    parser.add_argument("--ratio", type=float, default=0.20, help="Fraction of head channels to remove")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="domainmix_pd5_umair_hard_headprune20_recovery_e5")
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def round_keep_channels(total_channels, ratio, multiple=8):
    keep = int(round(total_channels * (1.0 - ratio)))
    keep = max(multiple, (keep // multiple) * multiple)
    keep = min(total_channels, keep)
    return keep


def prune_classification_head(yolo, ratio):
    model = yolo.model
    head = model.model[-1]
    conv_block = head.conv
    conv = conv_block.conv
    bn = conv_block.bn
    linear = head.linear

    if conv.out_channels != linear.in_features:
        raise ValueError(
            f"Expected head conv out_channels ({conv.out_channels}) to match "
            f"linear in_features ({linear.in_features})."
        )

    old_channels = conv.out_channels
    keep_channels = round_keep_channels(old_channels, ratio)
    remove_channels = old_channels - keep_channels

    with torch.no_grad():
        # L1 norm of each output filter: simple, deterministic channel saliency.
        scores = conv.weight.detach().abs().sum(dim=(1, 2, 3))
        keep_idx = torch.topk(scores, keep_channels, largest=True).indices.sort().values

        new_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=keep_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
        )
        new_conv.weight.copy_(conv.weight[keep_idx].detach())
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias[keep_idx].detach())

        new_bn = nn.BatchNorm2d(
            keep_channels,
            eps=bn.eps,
            momentum=bn.momentum,
            affine=bn.affine,
            track_running_stats=bn.track_running_stats,
        )
        if bn.affine:
            new_bn.weight.copy_(bn.weight[keep_idx].detach())
            new_bn.bias.copy_(bn.bias[keep_idx].detach())
        if bn.track_running_stats:
            new_bn.running_mean.copy_(bn.running_mean[keep_idx].detach())
            new_bn.running_var.copy_(bn.running_var[keep_idx].detach())
            new_bn.num_batches_tracked.copy_(bn.num_batches_tracked.detach())

        new_linear = nn.Linear(keep_channels, linear.out_features, bias=linear.bias is not None)
        new_linear.weight.copy_(linear.weight[:, keep_idx].detach())
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias.detach())

        conv_block.conv = new_conv
        conv_block.bn = new_bn
        head.linear = new_linear

    return {
        "old_head_channels": old_channels,
        "kept_head_channels": keep_channels,
        "removed_head_channels": remove_channels,
        "requested_ratio": ratio,
        "actual_ratio": remove_channels / old_channels,
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.model, task="classify")
    before_params = count_params(yolo.model)
    pruning = prune_classification_head(yolo, args.ratio)
    after_params = count_params(yolo.model)

    init_path = outdir / "head_pruned_init.pt"
    yolo.save(str(init_path))

    metadata = {
        "source_model": str(Path(args.model).resolve()),
        "data": str(Path(args.data).resolve()),
        "params_before": before_params,
        "params_after_head_prune": after_params,
        "param_reduction": before_params - after_params,
        "param_reduction_ratio": (before_params - after_params) / before_params,
        "pruning": pruning,
        "init_model": str(init_path.resolve()),
    }
    (outdir / "pruning_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    train_kwargs = dict(
        data=str(Path(args.data)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(outdir / "runs"),
        name=args.name,
        pretrained=False,
        patience=max(args.epochs, 5),
        plots=True,
        amp=True,
    )
    if args.augment:
        train_kwargs.update(
            hsv_h=0.035,
            hsv_s=0.65,
            hsv_v=0.55,
            degrees=20,
            translate=0.12,
            scale=0.35,
            shear=6,
            perspective=0.0008,
            fliplr=0.5,
            erasing=0.25,
        )

    yolo.train(**train_kwargs)


if __name__ == "__main__":
    main()
