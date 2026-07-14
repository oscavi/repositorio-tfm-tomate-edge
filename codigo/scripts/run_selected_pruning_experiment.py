import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(r"F:\TFM experimento\training")
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PRUNE = ROOT / "scripts" / "prune_head_finetune_cls.py"
EVAL = ROOT / "scripts" / "eval_cls_external.py"
DATA = ROOT / "dataset" / "domain_mix_zenodo_plantdoc_pd5_umair_hard"
ZENODO_VAL = ROOT / "dataset" / "zenodo_tomato_leaf" / "tomato-dataset" / "val"
PLANTDOC_VAL = ROOT / "dataset" / "plantdoc_tomato_cls" / "val"
UMAIR_VAL = ROOT / "dataset" / "external_prepared" / "umair_valid_pd5_mapped"
OUT = ROOT / "runs" / "pruning" / "pareto_selected_headprune20"


CASES = [
    {
        "base": "p03",
        "role": "best_plantdoc",
        "model": ROOT / "runs" / "classify" / "runs" / "pareto" / "p03_img416_lr5e4_wd5e4_augmed_e25" / "weights" / "best.pt",
        "imgsz": 416,
        "batch": 16,
    },
    {
        "base": "p07",
        "role": "best_umair",
        "model": ROOT / "runs" / "classify" / "runs" / "pareto" / "p07_img320_lr5e4_wd1e3_aughigh_e25" / "weights" / "best.pt",
        "imgsz": 320,
        "batch": 32,
    },
]


def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\n\n=== CMD ===\n")
        f.write(" ".join(str(x) for x in cmd) + "\n")
        f.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def recovery_best(outdir, name):
    direct = outdir / "runs" / name / "weights" / "best.pt"
    if direct.exists():
        return direct
    candidates = list((outdir / "runs").rglob("weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No recovery best.pt under {outdir}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def eval_model(model, data, out, imgsz, batch):
    if (out / "summary.json").exists():
        return
    run(
        [
            str(PYTHON),
            str(EVAL),
            "--model",
            str(model),
            "--data",
            str(data),
            "--out",
            str(out),
            "--imgsz",
            str(imgsz),
            "--batch",
            str(min(batch, 16)),
            "--chunk-size",
            "256",
        ],
        OUT / "logs" / f"{out.name}.log",
    )


def read_top1(path):
    return json.loads(path.read_text(encoding="utf-8"))["top1_accuracy"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        name = f"{case['base']}_headprune20_recovery_e5"
        outdir = OUT / name
        init_model = outdir / "head_pruned_init.pt"
        if not init_model.exists():
            run(
                [
                    str(PYTHON),
                    str(PRUNE),
                    "--model",
                    str(case["model"]),
                    "--data",
                    str(DATA),
                    "--outdir",
                    str(outdir),
                    "--ratio",
                    "0.20",
                    "--epochs",
                    "5",
                    "--imgsz",
                    str(case["imgsz"]),
                    "--batch",
                    str(case["batch"]),
                    "--workers",
                    "4",
                    "--device",
                    "0",
                    "--name",
                    name,
                    "--augment",
                ],
                OUT / "logs" / f"{name}_train.log",
            )

        best = recovery_best(outdir, name)
        for variant, model in [("pruned_init", init_model), ("recovery", best)]:
            for ds_name, ds_path in [("zenodo", ZENODO_VAL), ("plantdoc", PLANTDOC_VAL), ("umair", UMAIR_VAL)]:
                eval_model(
                    model=model,
                    data=ds_path,
                    out=OUT / "external_eval" / f"{case['base']}_{variant}_{ds_name}",
                    imgsz=case["imgsz"],
                    batch=case["batch"],
                )

        metadata = json.loads((outdir / "pruning_metadata.json").read_text(encoding="utf-8"))
        for variant in ["pruned_init", "recovery"]:
            rows.append(
                {
                    "base": case["base"],
                    "role": case["role"],
                    "variant": variant,
                    "imgsz": case["imgsz"],
                    "ratio": "0.20",
                    "params_before": metadata["params_before"],
                    "params_after": metadata["params_after_head_prune"],
                    "param_reduction_ratio": metadata["param_reduction_ratio"],
                    "model": str(init_model if variant == "pruned_init" else best),
                    "zenodo_top1": read_top1(OUT / "external_eval" / f"{case['base']}_{variant}_zenodo" / "summary.json"),
                    "plantdoc_top1": read_top1(OUT / "external_eval" / f"{case['base']}_{variant}_plantdoc" / "summary.json"),
                    "umair_top1": read_top1(OUT / "external_eval" / f"{case['base']}_{variant}_umair" / "summary.json"),
                }
            )

    with (OUT / "pruning_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT / "pruning_results.csv")


if __name__ == "__main__":
    main()
