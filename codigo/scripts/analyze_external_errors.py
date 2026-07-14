import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def safe_font(size=18, bold=False):
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


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["confidence"] = float(row["confidence"])
        row["correct"] = row["correct"].lower() == "true"
    return rows


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_gallery(rows, out, title, limit=24):
    samples = rows[:limit]
    if not samples:
        return
    cell_w, cell_h = 350, 290
    cols = 4
    rows_n = (len(samples) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows_n * cell_h + 80), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 22), title, fill=(34, 64, 76), font=safe_font(30, True))
    for idx, row in enumerate(samples):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h + 80
        try:
            img = Image.open(row["path"]).convert("RGB")
            img.thumbnail((280, 175), Image.LANCZOS)
            canvas.paste(img, (x + (cell_w - img.width) // 2, y + 4))
        except Exception:
            pass
        color = (42, 116, 79) if row["correct"] else (176, 76, 45)
        draw.text((x + 16, y + 190), f"GT: {short_label(row['true'])}", fill=(33, 40, 38), font=safe_font(16, True))
        draw.text((x + 16, y + 214), f"Pred: {short_label(row['pred'])}", fill=color, font=safe_font(16, True))
        draw.text((x + 16, y + 238), f"Conf: {row['confidence']:.3f}", fill=(86, 91, 88), font=safe_font(15))
    canvas.save(out, quality=94)


def main():
    parser = argparse.ArgumentParser(description="Analyze external classification errors and create hard-case subset.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hard-out")
    parser.add_argument("--title", default="Analisis de errores externos")
    parser.add_argument("--low-conf-threshold", type=float, default=0.65)
    parser.add_argument("--max-low-conf-per-class", type=int, default=40)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    hard_out = Path(args.hard_out) if args.hard_out else None
    if hard_out:
        if hard_out.exists():
            shutil.rmtree(hard_out)
        hard_out.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.predictions)
    errors = [r for r in rows if not r["correct"]]
    correct_low = [
        r for r in rows if r["correct"] and r["confidence"] <= args.low_conf_threshold
    ]
    correct_low_by_class = defaultdict(list)
    for row in sorted(correct_low, key=lambda r: r["confidence"]):
        correct_low_by_class[row["true"]].append(row)
    selected_low = []
    for cls, cls_rows in correct_low_by_class.items():
        selected_low.extend(cls_rows[: args.max_low_conf_per_class])
    hard_rows = errors + selected_low

    confusion = Counter((r["true"], r["pred"]) for r in errors)
    confusion_rows = [
        {
            "true": true,
            "pred": pred,
            "count": count,
            "share_of_class_errors": count / max(1, sum(1 for r in errors if r["true"] == true)),
        }
        for (true, pred), count in confusion.items()
    ]
    confusion_rows.sort(key=lambda r: (-r["count"], r["true"], r["pred"]))
    write_csv(out / "confusions_ranked.csv", confusion_rows, ["true", "pred", "count", "share_of_class_errors"])

    class_summary = []
    for cls in sorted({r["true"] for r in rows}):
        cls_rows = [r for r in rows if r["true"] == cls]
        cls_errors = [r for r in cls_rows if not r["correct"]]
        top_confusions = [
            f"{short_label(pred)}: {count}"
            for (true, pred), count in confusion.most_common()
            if true == cls
        ][:3]
        class_summary.append(
            {
                "class": cls,
                "total": len(cls_rows),
                "errors": len(cls_errors),
                "accuracy": (len(cls_rows) - len(cls_errors)) / len(cls_rows),
                "mean_error_confidence": sum(r["confidence"] for r in cls_errors) / len(cls_errors) if cls_errors else 0,
                "top_confusions": "; ".join(top_confusions),
            }
        )
    write_csv(
        out / "class_error_summary.csv",
        class_summary,
        ["class", "total", "errors", "accuracy", "mean_error_confidence", "top_confusions"],
    )

    # Global galleries and per-class galleries for the weakest classes.
    make_gallery(sorted(errors, key=lambda r: -r["confidence"]), out / "high_confidence_errors.jpg", "Errores con alta confianza")
    make_gallery(sorted(errors, key=lambda r: r["confidence"]), out / "low_confidence_errors.jpg", "Errores con baja confianza")
    weakest = sorted(class_summary, key=lambda r: r["accuracy"])[:4]
    for item in weakest:
        cls = item["class"]
        cls_errors = [r for r in errors if r["true"] == cls]
        make_gallery(
            sorted(cls_errors, key=lambda r: -r["confidence"]),
            out / f"errors_{cls}.jpg",
            f"Errores: {short_label(cls)}",
        )

    if hard_out:
        manifest_rows = []
        for idx, row in enumerate(hard_rows, start=1):
            src = Path(row["path"])
            reason = "error" if not row["correct"] else "low_conf_correct"
            dst = hard_out / row["true"] / f"hard_{idx:05d}_{reason}{src.suffix.lower()}"
            link_or_copy(src, dst)
            manifest_rows.append(
                {
                    "dst": str(dst),
                    "src": str(src),
                    "true": row["true"],
                    "pred": row["pred"],
                    "confidence": row["confidence"],
                    "reason": reason,
                }
            )
        write_csv(hard_out / "manifest.csv", manifest_rows, ["dst", "src", "true", "pred", "confidence", "reason"])

    summary = {
        "total_predictions": len(rows),
        "errors": len(errors),
        "low_confidence_correct_selected": len(selected_low),
        "hard_cases_total": len(hard_rows),
        "hard_cases_path": str(hard_out) if hard_out else None,
        "weakest_classes": weakest,
        "top_confusions": confusion_rows[:15],
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    readme = out / "README.md"
    hard_line = f"Subconjunto hard cases: `{hard_out}`\n\n" if hard_out else ""
    readme.write_text(
        f"# {args.title}\n\n"
        f"Total de predicciones: {len(rows)}\n\n"
        f"Errores: {len(errors)}\n\n"
        f"Casos correctos de baja confianza seleccionados: {len(selected_low)}\n\n"
        f"{hard_line}"
        "Artefactos principales:\n\n"
        "- `class_error_summary.csv`\n"
        "- `confusions_ranked.csv`\n"
        "- `high_confidence_errors.jpg`\n"
        "- `low_confidence_errors.jpg`\n"
        "- `summary.json`\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
