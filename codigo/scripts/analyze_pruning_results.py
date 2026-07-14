import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASELINES = {
    "p03": {"zenodo_top1": 0.988, "plantdoc_top1": 0.7391304347826086, "umair_top1": 0.9874047582024569},
    "p07": {"zenodo_top1": 0.989, "plantdoc_top1": 0.6811594202898551, "umair_top1": 0.9917586689472866},
}


DOMAINS = [
    ("zenodo_top1", "Zenodo"),
    ("plantdoc_top1", "PlantDoc"),
    ("umair_top1", "Umair"),
]


def main():
    parser = argparse.ArgumentParser(description="Create pruning comparison tables and figures.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.csv)

    base_rows = []
    for base, metrics in BASELINES.items():
        row = {
            "base": base,
            "role": rows.loc[rows["base"] == base, "role"].iloc[0],
            "variant": "base",
            "imgsz": rows.loc[rows["base"] == base, "imgsz"].iloc[0],
            "ratio": 0.0,
            "params_before": rows.loc[rows["base"] == base, "params_before"].iloc[0],
            "params_after": rows.loc[rows["base"] == base, "params_before"].iloc[0],
            "param_reduction_ratio": 0.0,
            "model": "",
        }
        row.update(metrics)
        base_rows.append(row)

    enriched = pd.concat([pd.DataFrame(base_rows), rows], ignore_index=True)
    for metric, _ in DOMAINS:
        enriched[f"{metric}_delta_vs_base"] = enriched.apply(
            lambda row: row[metric] - BASELINES[row["base"]][metric], axis=1
        )
    enriched.to_csv(outdir / "pruning_results_enriched.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {"base": "#4c78a8", "pruned_init": "#59a14f", "recovery": "#e15759"}
    for ax, base in zip(axes, ["p03", "p07"]):
        sub = enriched[enriched["base"] == base]
        x = range(len(DOMAINS))
        width = 0.25
        for offset, variant in zip([-width, 0, width], ["base", "pruned_init", "recovery"]):
            vals = [float(sub[sub["variant"] == variant][metric].iloc[0]) for metric, _ in DOMAINS]
            ax.bar([i + offset for i in x], vals, width=width, label=variant, color=colors[variant])
        ax.set_title(f"{base}: comparacion de poda")
        ax.set_xticks(list(x))
        ax.set_xticklabels([label for _, label in DOMAINS])
        ax.set_ylim(0.30, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Top-1 accuracy")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "pruning_top1_comparison.png", dpi=180)

    summary = enriched[["base", "variant", "param_reduction_ratio", "zenodo_top1", "plantdoc_top1", "umair_top1"]]
    summary.to_csv(outdir / "pruning_summary_compact.csv", index=False)
    print(outdir)


if __name__ == "__main__":
    main()
