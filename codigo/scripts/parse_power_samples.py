import argparse
import csv
from pathlib import Path


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
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(path):
    rows = read_rows(path)
    fields = [field for field in rows[0].keys() if field.endswith("_power_mw")] if rows else []
    summary = {"log": Path(path).name, "samples": len(rows)}

    timestamps = [as_float(row.get("timestamp_unix")) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    duration_s = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else None
    summary["duration_s"] = duration_s

    for field in fields:
        values = [as_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        prefix = field.replace("_power_mw", "")
        summary[f"{prefix}_power_mw_mean"] = mean(values)
        summary[f"{prefix}_power_mw_max"] = max(values) if values else None
        summary[f"{prefix}_power_mw_p95"] = pct(values, 95)
        if duration_s is not None and values:
            summary[f"{prefix}_energy_j_est"] = mean(values) / 1000.0 * duration_s
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize Jetson INA3221 power CSV samples.")
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [summarize(path) for path in args.csvs]
    fields = sorted({key for row in rows for key in row})
    fields.remove("log")
    fields = ["log"] + fields

    with Path(args.out).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})


if __name__ == "__main__":
    main()
