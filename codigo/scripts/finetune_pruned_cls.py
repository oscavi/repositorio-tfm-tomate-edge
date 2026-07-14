import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an already-pruned YOLO classification model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="Dataset root with train/val folders.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(data_root, imgsz, batch, workers, augment):
    data_root = Path(data_root)
    train_tfms = [
        transforms.Resize((imgsz, imgsz), interpolation=transforms.InterpolationMode.BILINEAR),
    ]
    if augment:
        train_tfms = [
            transforms.RandomResizedCrop(imgsz, scale=(0.72, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.30, contrast=0.25, saturation=0.25, hue=0.035),
            transforms.RandomRotation(degrees=15, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
    train_tfms.extend([transforms.ToTensor()])
    val_tfms = transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )

    train_ds = datasets.ImageFolder(data_root / "train", transform=transforms.Compose(train_tfms), allow_empty=True)
    val_ds = datasets.ImageFolder(data_root / "val", transform=val_tfms, allow_empty=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return train_ds, val_ds, train_loader, val_loader


def model_logits(model, images):
    out = model(images)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        probs_or_logits = model_logits(model, images)
        loss = criterion(probs_or_logits, labels)
        pred = probs_or_logits.argmax(1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        loss_sum += float(loss.item()) * labels.numel()
    return {
        "loss": loss_sum / total if total else None,
        "top1_accuracy": correct / total if total else None,
        "correct": correct,
        "total": total,
    }


def freeze_backbone(model):
    for idx, module in enumerate(model.model):
        if idx < len(model.model) - 1:
            for param in module.parameters():
                param.requires_grad = False


def main():
    args = parse_args()
    seed_everything(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    yolo = YOLO(args.model, task="classify")
    model = yolo.model.to(device)
    if args.freeze_backbone:
        freeze_backbone(model)

    train_ds, val_ds, train_loader, val_loader = make_loaders(
        args.data, args.imgsz, args.batch, args.workers, args.augment
    )
    if yolo.model.names:
        expected = [yolo.model.names[i] for i in sorted(yolo.model.names)]
        if list(train_ds.classes) != expected:
            raise ValueError(f"Dataset class order differs from model names: {train_ds.classes} != {expected}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    history = []
    best_acc = -1.0
    best_path = outdir / "best.pt"
    last_path = outdir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model_logits(model, images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pred = logits.detach().argmax(1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            loss_sum += float(loss.item()) * labels.numel()
        scheduler.step()

        train_metrics = {
            "loss": loss_sum / total if total else None,
            "top1_accuracy": correct / total if total else None,
            "correct": correct,
            "total": total,
        }
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        yolo.save(str(last_path))
        if val_metrics["top1_accuracy"] is not None and val_metrics["top1_accuracy"] > best_acc:
            best_acc = val_metrics["top1_accuracy"]
            shutil.copy2(last_path, best_path)

    metadata = {
        "source_model": str(Path(args.model).resolve()),
        "data": str(Path(args.data).resolve()),
        "classes": train_ds.classes,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "freeze_backbone": args.freeze_backbone,
        "augment": args.augment,
        "best_val_top1": best_acc,
        "best_model": str(best_path.resolve()),
        "last_model": str(last_path.resolve()),
        "history": history,
    }
    (outdir / "finetune_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
