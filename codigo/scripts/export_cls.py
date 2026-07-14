import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export a trained classification model.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "torchscript"])
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)
    YOLO(str(weights)).export(
        format=args.format,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        half=args.half,
        int8=args.int8,
    )


if __name__ == "__main__":
    main()
