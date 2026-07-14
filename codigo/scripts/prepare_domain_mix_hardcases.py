import argparse
import csv
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root):
    for cls_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
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
    parser = argparse.ArgumentParser(description="Extend a domain-mix dataset with external hard cases.")
    parser.add_argument("--base", required=True, help="Existing classification dataset with train/val folders.")
    parser.add_argument("--hard", required=True, help="Hard-case folder with class subfolders.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--hard-repeat", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    hard = Path(args.hard)
    out = Path(args.out)
    if args.force:
        clear_dir(out)
    else:
        out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ["train", "val"]:
        if not (base / split).exists():
            continue
        for cls, src in iter_images(base / split):
            dst = out / split / cls / src.name
            link_or_copy(src, dst)
            counts[(split, cls, "base")] = counts.get((split, cls, "base"), 0) + 1

    for rep in range(args.hard_repeat):
        for cls, src in iter_images(hard):
            dst = out / "train" / cls / f"hard_r{rep + 1:02d}_{src.name}"
            link_or_copy(src, dst)
            counts[("train", cls, "hard")] = counts.get(("train", cls, "hard"), 0) + 1

    summary_rows = []
    all_classes = sorted({key[1] for key in counts})
    for split in ["train", "val"]:
        for cls in all_classes:
            base_count = counts.get((split, cls, "base"), 0)
            hard_count = counts.get((split, cls, "hard"), 0)
            total = base_count + hard_count
            if total:
                summary_rows.append(
                    {
                        "split": split,
                        "class": cls,
                        "base_count": base_count,
                        "hard_count": hard_count,
                        "total": total,
                    }
                )

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "base_count", "hard_count", "total"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(out)


if __name__ == "__main__":
    main()
