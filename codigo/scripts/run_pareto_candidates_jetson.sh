#!/usr/bin/env bash
set -u

ROOT="/home/mic-710ai/tfm-experimento"
MODEL_ROOT="$ROOT/models/pareto_candidates"
ENGINE_ROOT="$ROOT/engines/pareto_candidates"
LOG_ROOT="$ROOT/logs/pareto_candidates_$(date +%Y%m%d_%H%M%S)"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
CALIB_DIR="$ROOT/datasets/calibration_int8_pd5_umair_hard_50pc"
WARMUP_MS="${WARMUP_MS:-1000}"
DURATION_S="${DURATION_S:-20}"
ITERATIONS="${ITERATIONS:-1000}"

mkdir -p "$ENGINE_ROOT" "$LOG_ROOT/build" "$LOG_ROOT/benchmark" "$LOG_ROOT/resource"

echo "log_root=$LOG_ROOT" | tee "$LOG_ROOT/run_metadata.txt"
date -Is | tee -a "$LOG_ROOT/run_metadata.txt"
df -h / | tee -a "$LOG_ROOT/run_metadata.txt"
uname -a | tee -a "$LOG_ROOT/run_metadata.txt"
python3 - <<'PY' 2>&1 | tee -a "$LOG_ROOT/run_metadata.txt"
try:
    import tensorrt as trt
    print("tensorrt_version=" + trt.__version__)
except Exception as exc:
    print("tensorrt_version_error=" + repr(exc))
try:
    import cv2
    print("opencv_version=" + cv2.__version__)
except Exception as exc:
    print("opencv_version_error=" + repr(exc))
PY

case_rows=(
  "p02_img320_lr5e4_wd5e4_augmed_e25 320 fp16"
  "p03_img416_lr5e4_wd5e4_augmed_e25 416 fp32"
  "p03_img416_lr5e4_wd5e4_augmed_e25 416 fp16"
  "p03_img416_lr5e4_wd5e4_augmed_e25 416 int8"
  "p05_img320_lr1e4_wd5e4_augmed_cos_e25 320 fp16"
  "p07_img320_lr5e4_wd1e3_aughigh_e25 320 fp32"
  "p07_img320_lr5e4_wd1e3_aughigh_e25 320 fp16"
  "p07_img320_lr5e4_wd1e3_aughigh_e25 320 int8"
  "p08_img320_lr5e4_wd5e4_dropout01_e25 320 fp16"
)

build_engine() {
  local name="$1"
  local imgsz="$2"
  local mode="$3"
  local onnx="$MODEL_ROOT/$name/$name.onnx"
  local engine="$ENGINE_ROOT/${name}_${mode}.engine"
  local build_log="$LOG_ROOT/build/${name}_${mode}_build.log"
  local cache="$ENGINE_ROOT/${name}_${mode}.cache"

  if [ -f "$engine" ]; then
    echo "SKIP existing engine: $engine" | tee "$build_log"
    return 0
  fi
  if [ ! -f "$onnx" ]; then
    echo "ERROR missing ONNX: $onnx" | tee "$build_log"
    return 1
  fi

  echo "Building $name $mode" | tee "$build_log"
  if [ "$mode" = "int8" ]; then
    if [ ! -d "$CALIB_DIR" ]; then
      echo "ERROR missing calibration dir: $CALIB_DIR" | tee -a "$build_log"
      return 1
    fi
    python3 "$ROOT/scripts/build_trt_int8_engine.py" \
      --onnx "$onnx" \
      --calib-dir "$CALIB_DIR" \
      --engine "$engine" \
      --cache "$cache" \
      --batch 8 \
      --imgsz "$imgsz" \
      --workspace 512 \
      --fp16 2>&1 | tee -a "$build_log"
  elif [ "$mode" = "fp16" ]; then
    "$TRTEXEC" --onnx="$onnx" --saveEngine="$engine" --explicitBatch --fp16 --workspace=512 --buildOnly 2>&1 | tee -a "$build_log"
  else
    "$TRTEXEC" --onnx="$onnx" --saveEngine="$engine" --explicitBatch --workspace=512 --buildOnly 2>&1 | tee -a "$build_log"
  fi
}

benchmark_engine() {
  local order="$1"
  local name="$2"
  local mode="$3"
  local engine="$ENGINE_ROOT/${name}_${mode}.engine"
  local case_id
  case_id="$(printf "%02d" "$order")_${name}_${mode}"
  local trt_log="$LOG_ROOT/benchmark/${case_id}_trtexec_load.log"
  local tegra_log="$LOG_ROOT/resource/${case_id}_tegrastats.log"
  local power_csv="$LOG_ROOT/resource/${case_id}_power.csv"
  local sample_duration=$((DURATION_S + 4))

  if [ ! -f "$engine" ]; then
    echo "SKIP benchmark, missing engine: $engine" | tee "$trt_log"
    return 0
  fi

  echo "Benchmark $case_id" | tee "$trt_log"
  ( tegrastats --interval 250 > "$tegra_log" 2>&1 ) &
  local tegra_pid=$!
  ( python3 "$ROOT/scripts/sample_jetson_power.py" --out "$power_csv" --interval-ms 250 --duration "$sample_duration" > "$LOG_ROOT/resource/${case_id}_power_sampler.log" 2>&1 ) &
  local power_pid=$!
  sleep 2
  "$TRTEXEC" --loadEngine="$engine" --warmUp="$WARMUP_MS" --duration="$DURATION_S" --iterations="$ITERATIONS" --percentile=99 2>&1 | tee -a "$trt_log"
  local trt_status=${PIPESTATUS[0]}
  sleep 1
  kill "$tegra_pid" 2>/dev/null || true
  wait "$power_pid" 2>/dev/null || true
  echo "trtexec_status=$trt_status" >> "$trt_log"
}

for row in "${case_rows[@]}"; do
  read -r name imgsz mode <<< "$row"
  build_engine "$name" "$imgsz" "$mode"
done

order=1
for row in "${case_rows[@]}"; do
  read -r name imgsz mode <<< "$row"
  benchmark_engine "$order" "$name" "$mode"
  order=$((order + 1))
done

find "$LOG_ROOT" -maxdepth 3 -type f | sort > "$LOG_ROOT/files.txt"
df -h / | tee -a "$LOG_ROOT/run_metadata.txt"
echo "DONE $LOG_ROOT"
