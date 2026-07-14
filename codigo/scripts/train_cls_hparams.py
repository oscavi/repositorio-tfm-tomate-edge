import argparse
from pathlib import Path

from ultralytics import YOLO


AUG_PROFILES = {
    "none": {},
    "low": {
        "hsv_h": 0.015,
        "hsv_s": 0.30,
        "hsv_v": 0.25,
        "degrees": 8,
        "translate": 0.05,
        "scale": 0.15,
        "shear": 2,
        "perspective": 0.0002,
        "fliplr": 0.5,
        "erasing": 0.10,
    },
    "medium": {
        "hsv_h": 0.035,
        "hsv_s": 0.65,
        "hsv_v": 0.55,
        "degrees": 20,
        "translate": 0.12,
        "scale": 0.35,
        "shear": 6,
        "perspective": 0.0008,
        "fliplr": 0.5,
        "erasing": 0.25,
    },
    "high": {
        "hsv_h": 0.050,
        "hsv_s": 0.80,
        "hsv_v": 0.70,
        "degrees": 30,
        "translate": 0.16,
        "scale": 0.45,
        "shear": 8,
        "perspective": 0.0012,
        "fliplr": 0.5,
        "erasing": 0.35,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Ultralytics classifier with explicit hyperparameters.")
    parser.add_argument("--data", required=True, help="Dataset root with train/val class folders.")
    parser.add_argument("--model", required=True, help="Initial model or checkpoint.")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/pareto")
    parser.add_argument("--name", required=True)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--aug-profile", choices=sorted(AUG_PROFILES), default="medium")
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    data = Path(args.data)
    if not (data / "train").exists() or not (data / "val").exists():
        raise FileNotFoundError(f"Expected train/val folders under {data}")

    train_kwargs = {
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "pretrained": True,
        "patience": args.patience,
        "plots": True,
        "amp": True,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "cos_lr": args.cos_lr,
        "seed": args.seed,
        "deterministic": True,
        "exist_ok": False,
    }
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze
    train_kwargs.update(AUG_PROFILES[args.aug_profile])
    train_kwargs["mixup"] = args.mixup
    train_kwargs["cutmix"] = args.cutmix

    model = YOLO(args.model, task="classify")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
