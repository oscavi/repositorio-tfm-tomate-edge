#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/mic-710ai/tfm-experimento}"
DURATION="${DURATION:-20}"
INTERVAL_MS="${INTERVAL_MS:-250}"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
PYTHON="${PYTHON:-python3}"

mkdir -p "$ROOT/logs/domainmix_pd5_umair_hard_int8/resources" "$ROOT/logs/domainmix_pd5_umair_hard_int8/power"
cd "$ROOT"

name="int8"
engine="engines/tomato_yolov8n_cls_domainmix_pd5_umair_hard_int8.engine"
tegra_log="logs/domainmix_pd5_umair_hard_int8/resources/${name}_tegrastats.log"
power_csv="logs/domainmix_pd5_umair_hard_int8/power/${name}_ina3221.csv"
trt_log="logs/domainmix_pd5_umair_hard_int8/resources/${name}_trtexec_load.log"

rm -f "$tegra_log" "$power_csv" "$trt_log"
sync

tegrastats --interval "$INTERVAL_MS" > "$tegra_log" &
tegra_pid=$!

"$PYTHON" scripts/sample_jetson_power.py \
  --out "$power_csv" \
  --interval-ms "$INTERVAL_MS" \
  --duration "$((DURATION + 3))" &
power_pid=$!

sleep 2
"$TRTEXEC" \
  --loadEngine="$engine" \
  --warmUp=1000 \
  --duration="$DURATION" \
  --iterations=1000 \
  2>&1 | tee "$trt_log"

kill "$tegra_pid" 2>/dev/null || true
wait "$tegra_pid" 2>/dev/null || true
wait "$power_pid" 2>/dev/null || true

echo "domainmix_pd5_umair_hard_int8 resource and power logs written under $ROOT/logs/domainmix_pd5_umair_hard_int8"
