import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


RUN_ROOT = Path(r"F:\TFM experimento\training\runs\pareto_sweep_2026_05_21")
RESULTS = RUN_ROOT / "pareto_training_results.csv"
OUT = RUN_ROOT / "analysis"


def load_rows():
    with RESULTS.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["zenodo_top1", "plantdoc_top1", "umair_top1", "model_size_mb"]:
            row[key] = float(row[key])
        for key in ["epochs", "imgsz", "batch", "freeze"]:
            row[key] = int(row[key])
    return rows


def short_name(name):
    return name.split("_")[0]


def plot_scatter(rows, x, y, path, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for row in rows:
        is_front = row["pareto_training_front"].lower() == "yes"
        ax.scatter(
            row[x],
            row[y],
            s=90 if is_front else 55,
            marker="o" if is_front else "x",
            color="#1f77b4" if is_front else "#8a8f98",
        )
        ax.annotate(short_name(row["name"]), (row[x], row[y]), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{ylabel} frente a {xlabel}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grouped(rows, path):
    labels = [short_name(r["name"]) for r in rows]
    xs = range(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar([x - width for x in xs], [r["zenodo_top1"] for r in rows], width, label="Zenodo", color="#4c78a8")
    ax.bar(xs, [r["plantdoc_top1"] for r in rows], width, label="PlantDoc", color="#f58518")
    ax.bar([x + width for x in xs], [r["umair_top1"] for r in rows], width, label="Umair", color="#54a24b")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("Top-1")
    ax.set_title("Exactitud Top-1 por configuración y dominio")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    selected_names = {
        "p02_img320_lr5e4_wd5e4_augmed_e25": "equilibrio_general",
        "p03_img416_lr5e4_wd5e4_augmed_e25": "mejor_plantdoc",
        "p05_img320_lr1e4_wd5e4_augmed_cos_e25": "mejor_zenodo_control",
        "p07_img320_lr5e4_wd1e3_aughigh_e25": "mejor_umair",
        "p08_img320_lr5e4_wd5e4_dropout01_e25": "equilibrio_umair_dropout",
    }
    for row in rows:
        row["selected_for_edge"] = selected_names.get(row["name"], "")

    fieldnames = list(rows[0].keys())
    with (OUT / "pareto_training_results_enriched.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected = [r for r in rows if r["selected_for_edge"]]
    with (OUT / "selected_edge_candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    plot_grouped(rows, OUT / "top1_by_domain.png")
    plot_scatter(rows, "plantdoc_top1", "umair_top1", OUT / "pareto_plantdoc_vs_umair.png", "PlantDoc Top-1", "Umair Top-1")
    plot_scatter(rows, "zenodo_top1", "plantdoc_top1", OUT / "zenodo_vs_plantdoc.png", "Zenodo Top-1", "PlantDoc Top-1")

    summary = {
        "best_zenodo": max(rows, key=lambda r: r["zenodo_top1"])["name"],
        "best_plantdoc": max(rows, key=lambda r: r["plantdoc_top1"])["name"],
        "best_umair": max(rows, key=lambda r: r["umair_top1"])["name"],
        "selected_edge_candidates": selected_names,
        "outputs": {
            "enriched_csv": str(OUT / "pareto_training_results_enriched.csv"),
            "selected_csv": str(OUT / "selected_edge_candidates.csv"),
            "top1_plot": str(OUT / "top1_by_domain.png"),
            "plantdoc_umair_plot": str(OUT / "pareto_plantdoc_vs_umair.png"),
        },
    }
    (OUT / "pareto_analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
