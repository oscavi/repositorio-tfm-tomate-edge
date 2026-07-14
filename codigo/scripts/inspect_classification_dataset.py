import argparse
import csv
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_split(root: Path, split: str):
    split_dir = root / split
    rows = []
    if not split_dir.exists():
        return rows
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        count = sum(1 for f in class_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        rows.append((split, class_dir.name, count))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="dataset_summary.csv")
    args = parser.parse_args()

    root = Path(args.root)
    all_rows = []
    for split in ("train", "val", "test"):
        all_rows.extend(count_split(root, split))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "count"])
        writer.writerows(all_rows)

    totals = {}
    for split, cls, count in all_rows:
        totals[split] = totals.get(split, 0) + count
        print(f"{split:5s} {cls:55s} {count:5d}")
    print("totals", totals)
    print(f"summary_csv={out}")


if __name__ == "__main__":
    main()
