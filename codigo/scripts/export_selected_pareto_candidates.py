import csv
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(r"F:\TFM experimento\training")
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
EXPORT = ROOT / "scripts" / "export_cls.py"
SELECTED = ROOT / "runs" / "pareto_sweep_2026_05_21" / "analysis" / "selected_edge_candidates.csv"
OUT = ROOT / "exports" / "pareto_candidates"
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "names.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    rows = list(csv.DictReader(SELECTED.open("r", encoding="utf-8")))
    manifest = []
    for row in rows:
        name = row["name"]
        imgsz = int(row["imgsz"])
        candidate_dir = OUT / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        weights = Path(row["weights"])
        copied_pt = candidate_dir / f"{name}.pt"
        if not copied_pt.exists():
            shutil.copy2(weights, copied_pt)

        expected_onnx = copied_pt.with_suffix(".onnx")
        if not expected_onnx.exists():
            cmd = [
                str(PYTHON),
                str(EXPORT),
                "--weights",
                str(copied_pt),
                "--format",
                "onnx",
                "--imgsz",
                str(imgsz),
                "--opset",
                "12",
                "--simplify",
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)

        manifest.append(
            {
                "name": name,
                "role": row["selected_for_edge"],
                "imgsz": imgsz,
                "pt": str(copied_pt),
                "onnx": str(expected_onnx),
                "zenodo_top1": row["zenodo_top1"],
                "plantdoc_top1": row["plantdoc_top1"],
                "umair_top1": row["umair_top1"],
            }
        )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
