import argparse
import csv
import ctypes
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CudaDriver:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so")
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuMemFree_v2.argtypes = [ctypes.c_void_p]
        self._check(self.lib.cuInit(0), "cuInit")

    def _check(self, code, name):
        if code != 0:
            raise RuntimeError(f"{name} failed with CUDA driver code {code}")

    def mem_alloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), nbytes), "cuMemAlloc")
        return ptr

    def memcpy_htod(self, dst, src):
        src = np.ascontiguousarray(src)
        self._check(
            self.lib.cuMemcpyHtoD_v2(dst, src.ctypes.data_as(ctypes.c_void_p), src.nbytes),
            "cuMemcpyHtoD",
        )

    def memcpy_dtoh(self, dst, src):
        dst = np.ascontiguousarray(dst)
        self._check(
            self.lib.cuMemcpyDtoH_v2(dst.ctypes.data_as(ctypes.c_void_p), src, dst.nbytes),
            "cuMemcpyDtoH",
        )

    def mem_free(self, ptr):
        if ptr:
            self.lib.cuMemFree_v2(ptr)


def preprocess(path, imgsz):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(image)


def iter_dataset(root):
    root = Path(root)
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for img in sorted(class_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                yield img, class_dir.name


def softmax(x):
    x = x.astype(np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TensorRT classification engine on an image-folder dataset.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--names", required=True, help="Text file with one class name per output index.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = Path(args.names).read_text(encoding="utf-8").splitlines()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(args.engine).read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {args.engine}")
    context = engine.create_execution_context()
    driver = CudaDriver()

    input_idx = next(i for i in range(engine.num_bindings) if engine.binding_is_input(i))
    output_idx = next(i for i in range(engine.num_bindings) if not engine.binding_is_input(i))
    input_shape = tuple(engine.get_binding_shape(input_idx))
    output_shape = tuple(engine.get_binding_shape(output_idx))
    output_size = int(np.prod(output_shape))

    inp = np.zeros(input_shape, dtype=np.float32)
    out = np.zeros(output_size, dtype=np.float32)
    d_in = driver.mem_alloc(inp.nbytes)
    d_out = driver.mem_alloc(out.nbytes)
    bindings = [0] * engine.num_bindings
    bindings[input_idx] = int(d_in.value)
    bindings[output_idx] = int(d_out.value)

    rows = []
    correct = 0
    total = 0
    t0 = time.time()
    try:
        for img, true_name in iter_dataset(args.data):
            inp = preprocess(img, args.imgsz)
            driver.memcpy_htod(d_in, inp)
            if not context.execute_v2(bindings):
                raise RuntimeError("TensorRT execution failed")
            driver.memcpy_dtoh(out, d_out)
            scores = softmax(out.reshape(-1))
            pred_idx = int(np.argmax(scores))
            pred_name = names[pred_idx]
            conf = float(scores[pred_idx])
            ok = pred_name == true_name
            correct += int(ok)
            total += 1
            rows.append({
                "image": str(img),
                "true": true_name,
                "pred": pred_name,
                "confidence": f"{conf:.6f}",
                "correct": int(ok),
            })
    finally:
        driver.mem_free(d_in)
        driver.mem_free(d_out)

    elapsed = time.time() - t0
    summary = {
        "engine": str(args.engine),
        "data": str(args.data),
        "total": total,
        "correct": correct,
        "top1_accuracy": correct / total if total else None,
        "elapsed_s": elapsed,
        "images_per_s_python_runner": total / elapsed if elapsed > 0 else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "true", "pred", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
