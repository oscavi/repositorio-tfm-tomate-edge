#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/mic-710ai/tfm-experimento}"
DURATION="${DURATION:-30}"
INTERVAL_MS="${INTERVAL_MS:-250}"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

mkdir -p "$ROOT/logs/resources"
cd "$ROOT"

run_case() {
  local name="$1"
  local engine="$2"
  local resource_log="logs/resources/${name}_tegrastats.log"
  local trt_log="logs/resources/${name}_trtexec_load.log"

  rm -f "$resource_log" "$trt_log"
  sync
  tegrastats --interval "$INTERVAL_MS" > "$resource_log" &
  local stats_pid=$!
  sleep 2
  "$TRTEXEC" \
    --loadEngine="$engine" \
    --warmUp=1000 \
    --duration="$DURATION" \
    --iterations=1000 \
    2>&1 | tee "$trt_log"
  kill "$stats_pid" 2>/dev/null || true
  wait "$stats_pid" 2>/dev/null || true
}

run_case "fp32" "engines/tomato_yolov8n_cls_fp32.engine"
run_case "fp16" "engines/tomato_yolov8n_cls_fp16.engine"

echo "Resource logs written under $ROOT/logs/resources"
