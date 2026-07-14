import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO weights for deployment.")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "torchscript"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Use FP16 where supported")
    parser.add_argument("--int8", action="store_true", help="Use INT8 where supported")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    model.export(format=args.format, imgsz=args.imgsz, half=args.half, int8=args.int8)


if __name__ == "__main__":
    main()
