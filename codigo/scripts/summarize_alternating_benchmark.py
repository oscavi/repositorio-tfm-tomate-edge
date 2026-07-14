import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def case_id(log_name):
    return log_name.split("_", 1)[0]


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def stdev(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def fmt(value, digits=3):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def pct_delta(value, base):
    if value is None or base in (None, 0):
        return None
    return (value - base) / base * 100.0


def main():
    parser = argparse.ArgumentParser(description="Combine alternating TensorRT, tegrastats and power summaries.")
    parser.add_argument("--trtexec", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--power", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    trt_rows = read_csv(args.trtexec)
    res_rows = {case_id(row["log"]): row for row in read_csv(args.resources)}
    power_rows = {case_id(row["log"]): row for row in read_csv(args.power)}

    rows = []
    for trt in trt_rows:
        cid = trt["case_order"]
        res = res_rows.get(cid, {})
        power = power_rows.get(cid, {})
        row = {
            "case_order": cid,
            "variant": trt["variant"],
            "throughput_qps": as_float(trt.get("throughput_qps")),
            "latency_mean_ms": as_float(trt.get("latency_mean_ms")),
            "latency_p99_ms": as_float(trt.get("latency_p99_ms")),
            "gpu_mean_ms": as_float(trt.get("gpu_mean_ms")),
            "ram_used_mb_mean": as_float(res.get("ram_used_mb_mean")),
            "cpu_avg_pct_mean": as_float(res.get("cpu_avg_pct_mean")),
            "gr3d_pct_mean": as_float(res.get("gr3d_pct_mean")),
            "pom_5v_in_power_w_mean": as_float(power.get("pom_5v_in_power_mw_mean")) / 1000.0
            if as_float(power.get("pom_5v_in_power_mw_mean")) is not None else None,
            "pom_5v_in_energy_j_est": as_float(power.get("pom_5v_in_energy_j_est")),
        }
        rows.append(row)

    fields = list(rows[0].keys())
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()})

    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    metrics = [
        ("throughput_qps", "throughput qps", "higher"),
        ("latency_mean_ms", "latencia media ms", "lower"),
        ("latency_p99_ms", "latencia P99 ms", "lower"),
        ("ram_used_mb_mean", "RAM media MB", "lower"),
        ("cpu_avg_pct_mean", "CPU media %", "lower"),
        ("gr3d_pct_mean", "GPU GR3D media %", "lower"),
        ("pom_5v_in_power_w_mean", "potencia POM_5V_IN W", "lower"),
        ("pom_5v_in_energy_j_est", "energia estimada J", "lower"),
    ]

    base = {
        key: mean([row[key] for row in by_variant.get("base_fp16", [])])
        for key, _, _ in metrics
    }

    lines = [
        "# Benchmark alternado Jetson Nano",
        "",
        "Ensayo repetido y alternado con TensorRT FP16 para reducir sesgos de calentamiento, carga previa o deriva temporal. Cada tramo se ejecuto con `trtexec --loadEngine`, muestreo `tegrastats` y lectura INA3221.",
        "",
        "## Secuencia ejecutada",
        "",
        "| orden | variante | throughput qps | latencia media ms | P99 ms | RAM MB | CPU % | GPU % | W | J |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case_order} | {variant} | {throughput_qps} | {latency_mean_ms} | {latency_p99_ms} | "
            "{ram_used_mb_mean} | {cpu_avg_pct_mean} | {gr3d_pct_mean} | {pom_5v_in_power_w_mean} | "
            "{pom_5v_in_energy_j_est} |".format(
                **{key: fmt(value) if isinstance(value, float) else value for key, value in row.items()}
            )
        )

    lines.extend([
        "",
        "## Agregado por variante",
        "",
        "| variante | n | throughput medio qps | sd qps | latencia media ms | sd ms | delta qps vs base | delta latencia vs base | potencia W | energia J |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in ["base_fp16", "prune_fp16", "prune_recovery_fp16"]:
        group = by_variant.get(variant, [])
        qps = [row["throughput_qps"] for row in group]
        lat = [row["latency_mean_ms"] for row in group]
        power_w = [row["pom_5v_in_power_w_mean"] for row in group]
        energy = [row["pom_5v_in_energy_j_est"] for row in group]
        mqps = mean(qps)
        mlat = mean(lat)
        lines.append(
            f"| {variant} | {len(group)} | {fmt(mqps)} | {fmt(stdev(qps))} | {fmt(mlat)} | "
            f"{fmt(stdev(lat))} | {fmt(pct_delta(mqps, base['throughput_qps']))}% | "
            f"{fmt(pct_delta(mlat, base['latency_mean_ms']))}% | {fmt(mean(power_w))} | {fmt(mean(energy))} |"
        )

    lines.extend([
        "",
        "## Lectura experimental",
        "",
        "- La variante base FP16 es muy estable en las tres repeticiones: throughput medio cercano a 221 qps y latencia media alrededor de 4.51 ms.",
        "- La poda ligera de cabeza aumenta el throughput aproximadamente un 25-26% y reduce la latencia alrededor de un 20%, manteniendo un consumo medio parecido en el rail de entrada.",
        "- La version podada con recuperacion conserva la mejora de rendimiento, pero en la evaluacion externa TensorRT ya habia mostrado peor precision que el modelo base; por tanto se interpreta como alternativa de rendimiento, no como sustituto principal.",
        "- Este benchmark complementa las metricas de precision: para la decision final debe cruzarse rendimiento, energia y exactitud externa PlantDoc/Umair.",
        "",
    ])

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
