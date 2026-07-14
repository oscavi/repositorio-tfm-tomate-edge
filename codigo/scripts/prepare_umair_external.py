import argparse
import csv
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAP = {
    "Bacterial_spot": "Tomato___Bacterial_spot",
    "Early_blight": "Tomato___Early_blight",
    "healthy": "Tomato___healthy",
    "Late_blight": "Tomato___Late_blight",
    "Leaf_Mold": "Tomato___Leaf_Mold",
    "Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Target_Spot": "Tomato___Target_Spot",
    "Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare UmairPirzada tomato validation set for pd5 external evaluation.")
    parser.add_argument("--src", required=True, help="Source folder with class subfolders.")
    parser.add_argument("--out", required=True, help="Output folder with mapped class subfolders.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        mapped = CLASS_MAP.get(cls_dir.name)
        if not mapped:
            rows.append({"source_class": cls_dir.name, "mapped_class": "", "count": 0, "status": "excluded"})
            continue
        count = 0
        for count, img in enumerate(
            (p for p in sorted(cls_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
            start=1,
        ):
            dst = out / mapped / f"umair_{cls_dir.name}_{count:05d}{img.suffix.lower()}"
            link_or_copy(img, dst)
        rows.append({"source_class": cls_dir.name, "mapped_class": mapped, "count": count, "status": "included"})

    with (out / "mapping_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_class", "mapped_class", "count", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(out)


if __name__ == "__main__":
    main()
