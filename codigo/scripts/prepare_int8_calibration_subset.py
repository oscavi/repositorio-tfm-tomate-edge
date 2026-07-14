import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images_in(class_dir):
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def main():
    parser = argparse.ArgumentParser(description="Create a balanced image subset for TensorRT INT8 calibration.")
    parser.add_argument("--src", required=True, help="Classification train directory with one subdirectory per class.")
    parser.add_argument("--out", required=True, help="Output calibration directory.")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        raise FileNotFoundError(src)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rng = random.Random(args.seed)
    rows = ["class,available,selected"]
    total = 0
    for class_dir in sorted(path for path in src.iterdir() if path.is_dir()):
        imgs = images_in(class_dir)
        selected = imgs if len(imgs) <= args.per_class else rng.sample(imgs, args.per_class)
        dst_class = out / class_dir.name
        dst_class.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(selected):
            dst = dst_class / f"{idx:04d}_{img.name}"
            shutil.copy2(img, dst)
        total += len(selected)
        rows.append(f"{class_dir.name},{len(imgs)},{len(selected)}")

    (out / "calibration_manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {total} calibration images to {out}")


if __name__ == "__main__":
    main()
