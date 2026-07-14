import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for tomato leaf disease detection.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model, e.g. yolov8n.pt or yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="tomato_yolo")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project="runs",
        name=args.name,
        pretrained=True,
        patience=20,
        cache=False,
        amp=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
