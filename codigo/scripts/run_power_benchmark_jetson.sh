#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/mic-710ai/tfm-experimento}"
DURATION="${DURATION:-20}"
INTERVAL_MS="${INTERVAL_MS:-250}"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
PYTHON="${PYTHON:-python3}"

mkdir -p "$ROOT/logs/power"
cd "$ROOT"

run_case() {
  local name="$1"
  local engine="$2"
  local power_csv="logs/power/${name}_ina3221.csv"
  local trt_log="logs/power/${name}_trtexec_load.log"

  rm -f "$power_csv" "$trt_log"
  sync

  "$PYTHON" scripts/sample_jetson_power.py \
    --out "$power_csv" \
    --interval-ms "$INTERVAL_MS" \
    --duration "$((DURATION + 3))" &
  local power_pid=$!

  sleep 2
  "$TRTEXEC" \
    --loadEngine="$engine" \
    --warmUp=1000 \
    --duration="$DURATION" \
    --iterations=1000 \
    2>&1 | tee "$trt_log"

  wait "$power_pid" 2>/dev/null || true
}

run_case "fp32" "engines/tomato_yolov8n_cls_fp32.engine"
run_case "fp16" "engines/tomato_yolov8n_cls_fp16.engine"

echo "Power logs written under $ROOT/logs/power"
