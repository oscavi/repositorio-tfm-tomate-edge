import argparse
import csv
import re
from pathlib import Path


RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
SWAP_RE = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
CPU_RE = re.compile(r"CPU\s+\[([^\]]+)\]")
GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
EMC_RE = re.compile(r"EMC_FREQ\s+(\d+)%")
TEMP_RE = re.compile(r"(PLL|CPU|PMIC|GPU|AO|thermal)@([0-9.]+)C")
POWER_RE = re.compile(r"(POM_5V_IN|POM_5V_GPU|POM_5V_CPU|VDD_IN|VDD_CPU|VDD_GPU)\s+(\d+)/(\d+)")


def mean(values):
    return sum(values) / len(values) if values else None


def pct(values, p):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
    return ordered[idx]


def fmt(value):
    if value is None:
        return ""
    return f"{value:.3f}"


def parse_cpu(cpu_text):
    usages = []
    freqs = []
    for part in cpu_text.split(","):
        m = re.search(r"(\d+)%@(\d+)", part)
        if m:
            usages.append(float(m.group(1)))
            freqs.append(float(m.group(2)))
    return usages, freqs


def parse_file(path):
    samples = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        row = {}
        if m := RAM_RE.search(line):
            row["ram_used_mb"] = float(m.group(1))
            row["ram_total_mb"] = float(m.group(2))
        if m := SWAP_RE.search(line):
            row["swap_used_mb"] = float(m.group(1))
            row["swap_total_mb"] = float(m.group(2))
        if m := CPU_RE.search(line):
            usages, freqs = parse_cpu(m.group(1))
            row["cpu_avg_pct"] = mean(usages)
            row["cpu_max_pct"] = max(usages) if usages else None
            row["cpu_avg_mhz"] = mean(freqs)
        if m := GR3D_RE.search(line):
            row["gr3d_pct"] = float(m.group(1))
        if m := EMC_RE.search(line):
            row["emc_pct"] = float(m.group(1))
        for name, temp in TEMP_RE.findall(line):
            row[f"temp_{name.lower()}_c"] = float(temp)
        for name, instant, avg in POWER_RE.findall(line):
            row[f"power_{name.lower()}_mw"] = float(instant)
            row[f"power_{name.lower()}_avg_mw"] = float(avg)
        if row:
            samples.append(row)
    return samples


def summarize(samples):
    keys = sorted({key for sample in samples for key in sample})
    summary = {}
    for key in keys:
        values = [sample[key] for sample in samples if key in sample and sample[key] is not None]
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_max"] = max(values) if values else None
        summary[f"{key}_p95"] = pct(values, 95)
    summary["samples"] = len(samples)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Parse Jetson tegrastats logs into CSV summaries.")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for log in args.logs:
        samples = parse_file(log)
        row = {"log": Path(log).name}
        row.update(summarize(samples))
        rows.append(row)

    fields = sorted({k for row in rows for k in row})
    fields.remove("log")
    fields = ["log"] + fields
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) if isinstance(v, float) else v for k, v in row.items()})


if __name__ == "__main__":
    main()
