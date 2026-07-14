import argparse
import csv
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images(root):
    for cls_dir in sorted(Path(root).iterdir()):
        if not cls_dir.is_dir():
            continue
        for path in sorted(cls_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                yield cls_dir.name, path


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def clear_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Create a weighted Zenodo+PlantDoc classification dataset.")
    parser.add_argument("--zenodo", required=True, help="Root with train/val folders.")
    parser.add_argument("--plantdoc", required=True, help="Root with train/val folders.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--plantdoc-repeat", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    zenodo = Path(args.zenodo)
    plantdoc = Path(args.plantdoc)
    out = Path(args.out)
    if args.force:
        clear_dir(out)
    else:
        out.mkdir(parents=True, exist_ok=True)

    class_names = sorted(
        {p.name for p in (zenodo / "train").iterdir() if p.is_dir()}
        | {p.name for p in (plantdoc / "train").iterdir() if p.is_dir()}
        | {p.name for p in (plantdoc / "val").iterdir() if p.is_dir()}
    )
    for split in ["train", "val"]:
        for cls in class_names:
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {}

    for cls, src in images(zenodo / "train"):
        dst = out / "train" / cls / f"zenodo_{src.name}"
        link_or_copy(src, dst)
        counts[("train", cls)] = counts.get(("train", cls), 0) + 1

    for rep in range(args.plantdoc_repeat):
        for cls, src in images(plantdoc / "train"):
            dst = out / "train" / cls / f"plantdoc_r{rep + 1:02d}_{src.name}"
            link_or_copy(src, dst)
            counts[("train", cls)] = counts.get(("train", cls), 0) + 1

    # Use PlantDoc val as the validation split so model selection tracks field-domain behavior.
    for cls, src in images(plantdoc / "val"):
        dst = out / "val" / cls / f"plantdoc_{src.name}"
        link_or_copy(src, dst)
        counts[("val", cls)] = counts.get(("val", cls), 0) + 1

    summary = out / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "count"])
        writer.writeheader()
        for split in ["train", "val"]:
            for cls in class_names:
                writer.writerow({"split": split, "class": cls, "count": counts.get((split, cls), 0)})

    print(summary)


if __name__ == "__main__":
    main()
