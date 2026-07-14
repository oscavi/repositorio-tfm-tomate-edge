import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Ultralytics classification model.")
    parser.add_argument("--data", required=True, help="Dataset root with train/val class folders")
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="tomato_cls_baseline")
    parser.add_argument("--device", default="0")
    parser.add_argument("--augment", action="store_true", help="Enable stronger field-like augmentation knobs")
    return parser.parse_args()


def main():
    args = parse_args()
    data = Path(args.data)
    if not (data / "train").exists():
        raise FileNotFoundError(f"Expected train folder under {data}")

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project="runs",
        name=args.name,
        pretrained=True,
        patience=10,
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

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
