import argparse
import csv
import json
import gc
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(data_root):
    rows = []
    for cls_dir in sorted(Path(data_root).iterdir()):
        if not cls_dir.is_dir():
            continue
        for path in sorted(cls_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                rows.append((path, cls_dir.name))
    return rows


def predict_in_chunks(model, paths, imgsz, batch, chunk_size):
    all_results = []
    for start in range(0, len(paths), chunk_size):
        chunk = paths[start : start + chunk_size]
        all_results.extend(model.predict(chunk, imgsz=imgsz, batch=batch, verbose=False))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_results


def safe_font(size=20, bold=False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def short_label(label):
    return label.replace("Tomato___", "").replace("_", " ")


def make_confusion(labels, matrix, out):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([short_label(x) for x in labels], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([short_label(x) for x in labels], fontsize=8)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta PlantDoc mapeada")
    ax.set_title("PlantDoc tomate - matriz de confusion externa")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def make_examples(rows, out, limit=16):
    errors = [r for r in rows if not r["correct"]]
    samples = errors[:limit] if errors else rows[:limit]
    if not samples:
        return
    cell_w, cell_h = 360, 300
    cols = 4
    rows_n = int(np.ceil(len(samples) / cols))
    canvas = Image.new("RGB", (cols * cell_w, rows_n * cell_h + 70), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), "PlantDoc: ejemplos de prediccion externa", fill=(36, 49, 43), font=safe_font(30, True))
    for idx, row in enumerate(samples):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h + 70
        try:
            img = Image.open(row["path"]).convert("RGB")
            img.thumbnail((300, 185), Image.LANCZOS)
            canvas.paste(img, (x + (cell_w - img.width) // 2, y + 8))
        except Exception:
            pass
        color = (46, 125, 91) if row["correct"] else (170, 70, 45)
        draw.text((x + 18, y + 205), f"GT: {short_label(row['true'])}", fill=(36, 49, 43), font=safe_font(17, True))
        draw.text((x + 18, y + 230), f"Pred: {short_label(row['pred'])}", fill=color, font=safe_font(17, True))
        draw.text((x + 18, y + 255), f"Conf: {row['confidence']:.3f}", fill=(80, 90, 86), font=safe_font(16))
    canvas.save(out, quality=95)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an Ultralytics classification model on an external folder.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="Folder with class subfolders.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model, task="classify")
    model_names = model.names
    idx_to_name = {int(k): v for k, v in model_names.items()} if isinstance(model_names, dict) else dict(enumerate(model_names))
    known_labels = list(idx_to_name.values())

    samples = collect_images(args.data)
    paths = [str(path) for path, _ in samples]
    results = predict_in_chunks(model, paths, args.imgsz, args.batch, args.chunk_size)

    rows = []
    by_class = defaultdict(lambda: {"total": 0, "correct": 0, "conf": []})
    confusion_labels = sorted(set([label for _, label in samples]) | set(known_labels))
    label_idx = {label: i for i, label in enumerate(confusion_labels)}
    matrix = np.zeros((len(confusion_labels), len(confusion_labels)), dtype=int)

    for (path, true_label), result in zip(samples, results):
        top1 = int(result.probs.top1)
        conf = float(result.probs.top1conf)
        pred_label = idx_to_name[top1]
        correct = pred_label == true_label
        rows.append(
            {
                "path": str(path),
                "true": true_label,
                "pred": pred_label,
                "confidence": conf,
                "correct": correct,
            }
        )
        by_class[true_label]["total"] += 1
        by_class[true_label]["correct"] += int(correct)
        by_class[true_label]["conf"].append(conf)
        matrix[label_idx[true_label], label_idx[pred_label]] += 1

    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    summary = {
        "total": total,
        "correct": correct,
        "top1_accuracy": correct / total if total else 0,
        "mean_confidence": float(np.mean([r["confidence"] for r in rows])) if rows else 0,
        "pred_distribution": Counter(r["pred"] for r in rows),
        "class_metrics": {},
    }
    for cls, vals in sorted(by_class.items()):
        summary["class_metrics"][cls] = {
            "total": vals["total"],
            "correct": vals["correct"],
            "accuracy": vals["correct"] / vals["total"] if vals["total"] else 0,
            "mean_confidence": float(np.mean(vals["conf"])) if vals["conf"] else 0,
        }

    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true", "pred", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=dict)

    make_confusion(confusion_labels, matrix, out / "confusion_matrix.png")
    make_examples(rows, out / "prediction_examples.jpg")

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
