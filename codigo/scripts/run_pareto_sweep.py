import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN = ROOT / "scripts" / "train_cls_hparams.py"
EVAL = ROOT / "scripts" / "eval_cls_external.py"

DATASET = ROOT / "dataset" / "domain_mix_zenodo_plantdoc_pd5_umair_hard"
BASE_MODEL = ROOT / "runs" / "classify" / "runs" / "domainmix_pd5_from_fieldaug_e25_e10" / "weights" / "best.pt"
ZENODO_VAL = ROOT / "dataset" / "zenodo_tomato_leaf" / "tomato-dataset" / "val"
PLANTDOC_VAL = ROOT / "dataset" / "plantdoc_tomato_cls" / "val"
UMAIR_VAL = ROOT / "dataset" / "external_prepared" / "umair_valid_pd5_mapped"


CONFIGS = [
    {
        "name": "p01_img224_lr5e4_wd5e4_augmed_e25",
        "epochs": 25,
        "imgsz": 224,
        "batch": 64,
        "optimizer": "AdamW",
        "lr0": 5e-4,
        "weight_decay": 5e-4,
        "dropout": 0.0,
        "freeze": 0,
        "cos_lr": False,
        "aug_profile": "medium",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
    {
        "name": "p02_img320_lr5e4_wd5e4_augmed_e25",
        "epochs": 25,
        "imgsz": 320,
        "batch": 32,
        "optimizer": "AdamW",
        "lr0": 5e-4,
        "weight_decay": 5e-4,
        "dropout": 0.0,
        "freeze": 0,
        "cos_lr": False,
        "aug_profile": "medium",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
    {
        "name": "p03_img416_lr5e4_wd5e4_augmed_e25",
        "epochs": 25,
        "imgsz": 416,
        "batch": 16,
        "optimizer": "AdamW",
        "lr0": 5e-4,
        "weight_decay": 5e-4,
        "dropout": 0.0,
        "freeze": 0,
        "cos_lr": False,
        "aug_profile": "medium",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
    {
        "name": "p04_img320_lr1e3_wd5e4_augmed_e25",
        "epochs": 25,
        "imgsz": 320,
        "batch": 32,
        "optimizer": "AdamW",
        "lr0": 1e-3,
        "weight_decay": 5e-4,
        "dropout": 0.0,
        "freeze": 0,
        "cos_lr": False,
        "aug_profile": "medium",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
    {
        "name": "p05_img320_lr1e4_wd5e4_augmed_cos_e25",
        "epochs": 25,
        "imgsz": 320,
        "batch": 32,
        "optimizer": "AdamW",
        "lr0": 1e-4,
        "weight_decay": 5e-4,
        "dropout": 0.0,
        "freeze": 0,
        "cos_lr": True,
        "aug_profile": "medium",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
    {
        "name": "p06_img320_lr5e4_wd1e4_auglow_e25",
        "epochs": 25,
        "imgsz": 320,
        "batch": 32,
        "optimizer": "AdamW",
        "lr0": 5e-4,
        "weight_decay": 1e-4,
        "dropout": 0.0,
        "freeze": 0,
        "cos_lr": False,
        "aug_profile": "low",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
    {
        "name": "p07_img320_lr5e4_wd1e3_aughigh_e25",
        "epochs": 25,
        "imgsz": 320,
        "batch": 32,
        "optimizer": "AdamW",
        "lr0": 5e-4,
        "weight_decay": 1e-3,
        "dropout": 0.0,
        "freeze": 0,
        "cos_lr": False,
        "aug_profile": "high",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
    {
        "name": "p08_img320_lr5e4_wd5e4_dropout01_e25",
        "epochs": 25,
        "imgsz": 320,
        "batch": 32,
        "optimizer": "AdamW",
        "lr0": 5e-4,
        "weight_decay": 5e-4,
        "dropout": 0.1,
        "freeze": 0,
        "cos_lr": False,
        "aug_profile": "medium",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
    {
        "name": "p09_img320_lr5e4_wd5e4_freeze9_e25",
        "epochs": 25,
        "imgsz": 320,
        "batch": 32,
        "optimizer": "AdamW",
        "lr0": 5e-4,
        "weight_decay": 5e-4,
        "dropout": 0.0,
        "freeze": 9,
        "cos_lr": False,
        "aug_profile": "medium",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
    {
        "name": "p10_img320_lr5e4_wd5e4_mixcut_e25",
        "epochs": 25,
        "imgsz": 320,
        "batch": 32,
        "optimizer": "AdamW",
        "lr0": 5e-4,
        "weight_decay": 5e-4,
        "dropout": 0.0,
        "freeze": 0,
        "cos_lr": False,
        "aug_profile": "medium",
        "mixup": 0.05,
        "cutmix": 0.20,
    },
    {
        "name": "p11_img224_lr5e4_wd5e4_aughigh_e10",
        "epochs": 10,
        "imgsz": 224,
        "batch": 64,
        "optimizer": "AdamW",
        "lr0": 5e-4,
        "weight_decay": 5e-4,
        "dropout": 0.0,
        "freeze": 0,
        "cos_lr": False,
        "aug_profile": "high",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
    {
        "name": "p12_img320_sgd_lr1e2_wd5e4_augmed_e25",
        "epochs": 25,
        "imgsz": 320,
        "batch": 32,
        "optimizer": "SGD",
        "lr0": 1e-2,
        "weight_decay": 5e-4,
        "dropout": 0.0,
        "freeze": 0,
        "cos_lr": False,
        "aug_profile": "medium",
        "mixup": 0.0,
        "cutmix": 0.0,
    },
]


def run_command(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n=== {} ===\n".format(datetime.now().isoformat(timespec="seconds")))
        log.write(" ".join(str(c) for c in cmd) + "\n")
        log.flush()
        result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    return result.returncode


def best_weights(config):
    return ROOT / "runs" / "classify" / "runs" / "pareto" / config["name"] / "weights" / "best.pt"


def run_train(config, run_root):
    if best_weights(config).exists():
        return "skipped"
    cmd = [
        str(PYTHON),
        str(TRAIN),
        "--data",
        str(DATASET),
        "--model",
        str(BASE_MODEL),
        "--epochs",
        str(config["epochs"]),
        "--imgsz",
        str(config["imgsz"]),
        "--batch",
        str(config["batch"]),
        "--workers",
        "4",
        "--device",
        "0",
        "--project",
        "runs/pareto",
        "--name",
        config["name"],
        "--optimizer",
        config["optimizer"],
        "--lr0",
        str(config["lr0"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--dropout",
        str(config["dropout"]),
        "--freeze",
        str(config["freeze"]),
        "--aug-profile",
        config["aug_profile"],
        "--mixup",
        str(config["mixup"]),
        "--cutmix",
        str(config["cutmix"]),
    ]
    if config["cos_lr"]:
        cmd.append("--cos-lr")
    code = run_command(cmd, run_root / "logs" / f"{config['name']}_train.log")
    if code != 0:
        raise RuntimeError(f"Training failed for {config['name']}; see log.")
    return "trained"


def eval_one(config, dataset_name, dataset_path, run_root):
    out = run_root / "external_eval" / f"{config['name']}_{dataset_name}"
    summary = out / "summary.json"
    if summary.exists():
        return
    cmd = [
        str(PYTHON),
        str(EVAL),
        "--model",
        str(best_weights(config)),
        "--data",
        str(dataset_path),
        "--out",
        str(out),
        "--imgsz",
        str(config["imgsz"]),
        "--batch",
        str(min(config["batch"], 32)),
    ]
    code = run_command(cmd, run_root / "logs" / f"{config['name']}_{dataset_name}_eval.log")
    if code != 0:
        raise RuntimeError(f"Evaluation failed for {config['name']} on {dataset_name}; see log.")


def load_summary(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_row(config, run_root):
    row = dict(config)
    weights = best_weights(config)
    row["weights"] = str(weights)
    row["model_size_mb"] = round(weights.stat().st_size / (1024 * 1024), 3) if weights.exists() else None
    for dataset_name in ["zenodo", "plantdoc", "umair"]:
        summary = load_summary(run_root / "external_eval" / f"{config['name']}_{dataset_name}" / "summary.json")
        row[f"{dataset_name}_top1"] = summary["top1_accuracy"]
        row[f"{dataset_name}_total"] = summary["total"]
        row[f"{dataset_name}_correct"] = summary["correct"]
    return row


def dominates(a, b):
    maximize = ["zenodo_top1", "plantdoc_top1", "umair_top1"]
    better_or_equal = all(float(a[m]) >= float(b[m]) for m in maximize)
    strictly_better = any(float(a[m]) > float(b[m]) for m in maximize)
    smaller_or_equal = float(a["model_size_mb"]) <= float(b["model_size_mb"])
    smaller_strict = float(a["model_size_mb"]) < float(b["model_size_mb"])
    return better_or_equal and smaller_or_equal and (strictly_better or smaller_strict)


def mark_pareto(rows):
    for row in rows:
        row["pareto_training_front"] = "yes"
    for i, row in enumerate(rows):
        for j, other in enumerate(rows):
            if i != j and dominates(other, row):
                row["pareto_training_front"] = "no"
                break


def write_plan(run_root):
    plan_path = run_root / "pareto_experiment_plan.csv"
    with plan_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CONFIGS[0].keys()))
        writer.writeheader()
        writer.writerows(CONFIGS)
    with (run_root / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Pareto hyperparameter sweep\n\n")
        f.write("Compact multiobjective sweep for tomato leaf disease classification.\n\n")
        f.write("Objectives: maximize Zenodo, PlantDoc and Umair Top-1; minimize model size first, then later combine with Jetson latency, FPS and power.\n\n")
        f.write("Training base checkpoint: `{}`\n\n".format(BASE_MODEL))
        f.write("Training dataset: `{}`\n\n".format(DATASET))
        f.write("External validation datasets: Zenodo val, PlantDoc val and Umair field validation.\n")


def main():
    parser = argparse.ArgumentParser(description="Run compact Pareto hyperparameter sweep.")
    parser.add_argument("--run-root", default=str(ROOT / "runs" / "pareto_sweep_2026_05_21"))
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all configs.")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    write_plan(run_root)
    configs = CONFIGS[: args.max_runs] if args.max_runs else CONFIGS

    status_path = run_root / "status.json"
    status = {"started_at": datetime.now().isoformat(timespec="seconds"), "runs": {}}
    if args.plan_only:
        print(run_root)
        return

    for config in configs:
        name = config["name"]
        t0 = time.time()
        status["runs"][name] = {"status": "running", "updated_at": datetime.now().isoformat(timespec="seconds")}
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        train_status = run_train(config, run_root)
        for dataset_name, dataset_path in [
            ("zenodo", ZENODO_VAL),
            ("plantdoc", PLANTDOC_VAL),
            ("umair", UMAIR_VAL),
        ]:
            eval_one(config, dataset_name, dataset_path, run_root)
        status["runs"][name] = {
            "status": "done",
            "train": train_status,
            "seconds": round(time.time() - t0, 1),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    rows = [collect_row(config, run_root) for config in configs]
    mark_pareto(rows)
    out_csv = run_root / "pareto_training_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    status["finished_at"] = datetime.now().isoformat(timespec="seconds")
    status["results_csv"] = str(out_csv)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(out_csv)


if __name__ == "__main__":
    if not PYTHON.exists():
        print(f"Python not found: {PYTHON}", file=sys.stderr)
        sys.exit(1)
    main()
