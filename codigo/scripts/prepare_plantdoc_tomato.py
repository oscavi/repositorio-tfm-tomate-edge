import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image


CLASS_MAP = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sanitize(name):
    name = re.sub(r'[<>:"/\\\\|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180] or "image"


def git_paths(repo):
    proc = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "-z", "--name-only", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [p.decode("utf-8", errors="replace") for p in proc.stdout.split(b"\0") if p]


def git_show(repo, path):
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{path}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract PlantDoc tomato classes into a safe classification dataset.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo)
    out = Path(args.out)
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {}
    for path in git_paths(repo):
        parts = Path(path).parts
        if len(parts) < 3:
            continue
        split, plantdoc_class = parts[0], parts[1]
        if split not in {"train", "test"} or plantdoc_class not in CLASS_MAP:
            continue
        suffix = Path(path).suffix.lower()
        if suffix not in IMAGE_EXTS:
            continue

        target_split = "val" if split == "test" else "train"
        target_class = CLASS_MAP[plantdoc_class]
        dest_dir = out / target_split / target_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        index = counts.get((target_split, target_class), 0) + 1
        counts[(target_split, target_class)] = index
        dest_name = f"{index:04d}_{sanitize(Path(path).name)}"
        dest = dest_dir / dest_name

        try:
            dest.write_bytes(git_show(repo, path))
        except subprocess.CalledProcessError:
            continue
        if not valid_image(dest):
            dest.unlink(missing_ok=True)
            continue
        rows.append(
            {
                "source_split": split,
                "target_split": target_split,
                "plantdoc_class": plantdoc_class,
                "target_class": target_class,
                "source_path": path,
                "target_path": str(dest),
            }
        )

    summary_path = out / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "count"])
        writer.writeheader()
        for split in ["train", "val"]:
            for cls in sorted(set(CLASS_MAP.values())):
                n = counts.get((split, cls), 0)
                if n:
                    writer.writerow({"split": split, "class": cls, "count": n})

    manifest_path = out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["source_split", "target_split", "plantdoc_class", "target_class", "source_path", "target_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} images")
    print(summary_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
