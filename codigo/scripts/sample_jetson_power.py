import argparse
import csv
import time
from pathlib import Path


DEFAULT_DEVICE_GLOB = "/sys/bus/iio/devices/iio:device*"


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def read_float(path):
    text = read_text(path)
    if not text:
        return None
    try:
        return float(text.split()[0])
    except ValueError:
        return None


def find_ina_device(device):
    if device:
        path = Path(device)
        if not path.exists():
            raise FileNotFoundError(f"Power sensor device not found: {device}")
        return path

    for path in sorted(Path("/sys/bus/iio/devices").glob("iio:device*")):
        name = read_text(path / "name")
        if name == "ina3221x" and any(path.glob("in_power*_input")):
            return path
    raise FileNotFoundError("No INA3221 IIO power sensor found under /sys/bus/iio/devices")


def discover_rails(device):
    rails = []
    for label_path in sorted(device.glob("rail_name_*")):
        idx = label_path.name.rsplit("_", 1)[-1]
        name = read_text(label_path) or f"rail_{idx}"
        rails.append((idx, name))
    if rails:
        return rails

    indexes = sorted(
        p.name.replace("in_power", "").replace("_input", "")
        for p in device.glob("in_power*_input")
        if p.name.replace("in_power", "").replace("_input", "").isdigit()
    )
    return [(idx, f"rail_{idx}") for idx in indexes]


def sample(device, rails):
    row = {
        "timestamp_unix": f"{time.time():.6f}",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for idx, name in rails:
        prefix = name.lower().replace(" ", "_")
        voltage_mv = read_float(device / f"in_voltage{idx}_input")
        current_ma = read_float(device / f"in_current{idx}_input")
        power_mw = read_float(device / f"in_power{idx}_input")
        row[f"{prefix}_voltage_mv"] = voltage_mv
        row[f"{prefix}_current_ma"] = current_ma
        row[f"{prefix}_power_mw"] = power_mw
    return row


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def main():
    parser = argparse.ArgumentParser(description="Sample Jetson INA3221 power rails into CSV.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--device", default=None, help="Optional IIO device path.")
    args = parser.parse_args()

    device = find_ina_device(args.device)
    rails = discover_rails(device)
    if not rails:
        raise RuntimeError(f"No rails discovered in {device}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    first = sample(device, rails)
    fields = list(first.keys())
    interval_s = max(args.interval_ms, 1) / 1000.0
    end = time.monotonic() + args.duration

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: fmt(v) for k, v in first.items()})
        while time.monotonic() < end:
            time.sleep(interval_s)
            row = sample(device, rails)
            writer.writerow({k: fmt(v) for k, v in row.items()})


if __name__ == "__main__":
    main()
