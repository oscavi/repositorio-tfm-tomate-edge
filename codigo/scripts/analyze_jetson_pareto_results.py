import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FLOAT = r"([0-9]+(?:\.[0-9]+)?)"
LOADED_SIZE_RE = re.compile(r"Loaded engine size:\s+" + FLOAT + r"\s+MB")
ENGINE_LOAD_RE = re.compile(r"Engine loaded in\s+" + FLOAT + r"\s+sec")
THROUGHPUT_RE = re.compile(r"Throughput:\s+" + FLOAT + r"\s+qps")
LATENCY_RE = re.compile(
    r"Latency: min = " + FLOAT + r" ms, max = " + FLOAT + r" ms, mean = "
    + FLOAT + r" ms, median = " + FLOAT + r" ms, percentile\(99%\) = "
    + FLOAT + r" ms"
)
GPU_RE = re.compile(
    r"GPU Compute Time: min = " + FLOAT + r" ms, max = " + FLOAT + r" ms, mean = "
    + FLOAT + r" ms, median = " + FLOAT + r" ms, percentile\(99%\) = "
    + FLOAT + r" ms"
)
RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
CPU_RE = re.compile(r"CPU\s+\[([^\]]+)\]")
GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
POWER_RE = re.compile(r"(POM_5V_IN|POM_5V_GPU|POM_5V_CPU|VDD_IN|VDD_CPU|VDD_GPU)\s+(\d+)/(\d+)")


def mean(values):
    return sum(values) / len(values) if values else None


def parse_name(path):
    stem = Path(path).name.replace("_trtexec_load.log", "")
    order, rest = stem.split("_", 1)
    mode = next(m for m in ["fp32", "fp16", "int8"] if rest.endswith("_" + m))
    case = rest[: -(len(mode) + 1)]
    return order, case, mode


def parse_trtexec(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    row = {}
    if m := LOADED_SIZE_RE.search(text):
        row["engine_size_mb"] = float(m.group(1))
    if m := ENGINE_LOAD_RE.search(text):
        row["engine_load_s"] = float(m.group(1))
    if m := THROUGHPUT_RE.search(text):
        row["throughput_qps"] = float(m.group(1))
    if m := LATENCY_RE.search(text):
        keys = ["latency_min_ms", "latency_max_ms", "latency_mean_ms", "latency_median_ms", "latency_p99_ms"]
        row.update({k: float(v) for k, v in zip(keys, m.groups())})
    if m := GPU_RE.search(text):
        keys = ["gpu_min_ms", "gpu_max_ms", "gpu_mean_ms", "gpu_median_ms", "gpu_p99_ms"]
        row.update({k: float(v) for k, v in zip(keys, m.groups())})
    return row


def parse_cpu(cpu_text):
    values = []
    for part in cpu_text.split(","):
        m = re.search(r"(\d+)%@", part)
        if m:
            values.append(float(m.group(1)))
    return values


def parse_tegrastats(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        row = {}
        if m := RAM_RE.search(line):
            row["tegrastats_ram_used_mb"] = float(m.group(1))
        if m := CPU_RE.search(line):
            cpu = parse_cpu(m.group(1))
            row["tegrastats_cpu_avg_pct"] = mean(cpu)
            row["tegrastats_cpu_max_pct"] = max(cpu) if cpu else None
        if m := GR3D_RE.search(line):
            row["tegrastats_gr3d_pct"] = float(m.group(1))
        for rail, instant, avg in POWER_RE.findall(line):
            key = "tegrastats_" + rail.lower() + "_mw"
            row[key] = float(instant)
        if row:
            rows.append(row)
    summary = {}
    for key in sorted({k for r in rows for k in r}):
        values = [r[key] for r in rows if key in r and r[key] is not None]
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_max"] = max(values) if values else None
    summary["tegrastats_samples"] = len(rows)
    return summary


def parse_power(path):
    df = pd.read_csv(path)
    out = {"power_samples": len(df)}
    for col in df.columns:
        if col.endswith("_power_mw"):
            out[f"power_{col}_mean"] = float(df[col].mean())
            out[f"power_{col}_max"] = float(df[col].max())
    if "pom_5v_in_power_mw" in df.columns:
        out["power_vdd_in_mw_mean"] = float(df["pom_5v_in_power_mw"].mean())
        out["power_vdd_in_mw_max"] = float(df["pom_5v_in_power_mw"].max())
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze copied Jetson TensorRT benchmark logs.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)

    rows = []
    for trt_log in sorted((root / "benchmark").glob("*_trtexec_load.log")):
        order, case, mode = parse_name(trt_log)
        row = {"order": int(order), "case": case, "mode": mode, "trtexec_log": trt_log.name}
        row.update(parse_trtexec(trt_log))

        base = trt_log.name.replace("_trtexec_load.log", "")
        tegra = root / "resource" / f"{base}_tegrastats.log"
        power = root / "resource" / f"{base}_power.csv"
        if tegra.exists():
            row.update(parse_tegrastats(tegra))
        if power.exists():
            row.update(parse_power(power))
        rows.append(row)

    rows.sort(key=lambda r: r["order"])
    out_csv = root / "jetson_pareto_benchmark_summary.csv"
    fields = sorted({k for r in rows for k in r})
    for first in ["order", "case", "mode", "trtexec_log"]:
        fields.remove(first)
    fields = ["order", "case", "mode", "trtexec_log"] + fields
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    labels = [f"{r['case'].split('_')[0]} {r['mode']}" for r in rows]
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.bar(labels, df["latency_mean_ms"], color="#4c78a8", label="Latencia media")
    ax1.set_ylabel("Latencia media (ms)")
    ax1.tick_params(axis="x", rotation=35)
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(labels, df["throughput_qps"], color="#f58518", marker="o", label="Throughput")
    ax2.set_ylabel("Throughput (qps)")
    fig.tight_layout()
    fig.savefig(root / "jetson_latency_throughput.png", dpi=180)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(labels, df["power_vdd_in_mw_mean"], color="#59a14f")
    ax.set_ylabel("Potencia media POM_5V_IN (mW)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "jetson_power_mean.png", dpi=180)

    print(out_csv)


if __name__ == "__main__":
    main()
