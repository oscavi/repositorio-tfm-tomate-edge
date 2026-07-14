import argparse
import csv
import re
from pathlib import Path


FLOAT = r"([0-9]+(?:\.[0-9]+)?)"
LOADED_SIZE_RE = re.compile(r"Loaded engine size:\s+" + FLOAT + r"\s+MB")
ENGINE_LOAD_RE = re.compile(r"Engine loaded in\s+" + FLOAT + r"\s+sec")
THROUGHPUT_RE = re.compile(r"Throughput:\s+" + FLOAT + r"\s+qps")
LATENCY_RE = re.compile(
    r"Latency: min = " + FLOAT + r" ms, max = " + FLOAT + r" ms, mean = "
    + FLOAT + r" ms, median = " + FLOAT + r" ms, percentile\(99%\) = "
    + FLOAT + r" ms"
)
GPU_COMPUTE_RE = re.compile(
    r"GPU Compute Time: min = " + FLOAT + r" ms, max = " + FLOAT
    + r" ms, mean = " + FLOAT + r" ms, median = " + FLOAT
    + r" ms, percentile\(99%\) = " + FLOAT + r" ms"
)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def parse_case_name(path):
    name = Path(path).name.replace("_trtexec_load.log", "")
    parts = name.split("_", 1)
    return parts[0], parts[1] if len(parts) == 2 else name


def parse_file(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    case_order, variant = parse_case_name(path)
    row = {
        "log": Path(path).name,
        "case_order": case_order,
        "variant": variant,
    }
    if m := LOADED_SIZE_RE.search(text):
        row["loaded_engine_size_mb"] = float(m.group(1))
    if m := ENGINE_LOAD_RE.search(text):
        row["engine_load_s"] = float(m.group(1))
    if m := THROUGHPUT_RE.search(text):
        row["throughput_qps"] = float(m.group(1))
    if m := LATENCY_RE.search(text):
        keys = ["latency_min_ms", "latency_max_ms", "latency_mean_ms", "latency_median_ms", "latency_p99_ms"]
        row.update({key: float(value) for key, value in zip(keys, m.groups())})
    if m := GPU_COMPUTE_RE.search(text):
        keys = ["gpu_min_ms", "gpu_max_ms", "gpu_mean_ms", "gpu_median_ms", "gpu_p99_ms"]
        row.update({key: float(value) for key, value in zip(keys, m.groups())})
    return row


def main():
    parser = argparse.ArgumentParser(description="Parse TensorRT trtexec benchmark summaries.")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [parse_file(log) for log in args.logs]
    rows.sort(key=lambda row: row["case_order"])

    fields = sorted({key for row in rows for key in row})
    for key in ["log", "case_order", "variant"]:
        fields.remove(key)
    fields = ["case_order", "variant", "log"] + fields

    with Path(args.out).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})


if __name__ == "__main__":
    main()
