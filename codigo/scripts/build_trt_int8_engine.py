import argparse
import ctypes
import os
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class CudaDriver:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so")
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
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

    def mem_free(self, ptr):
        if ptr:
            self.lib.cuMemFree_v2(ptr)


def list_images(root):
    paths = []
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
    return sorted(paths)


def preprocess(path, imgsz):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))


class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths, cache_file, batch_size=8, imgsz=224):
        super().__init__()
        self.image_paths = list(image_paths)
        self.cache_file = str(cache_file)
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.index = 0
        self.driver = CudaDriver()
        self.host = np.zeros((batch_size, 3, imgsz, imgsz), dtype=np.float32)
        self.device = self.driver.mem_alloc(self.host.nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.image_paths):
            return None
        batch = self.image_paths[self.index:self.index + self.batch_size]
        self.host.fill(0.0)
        for i, path in enumerate(batch):
            self.host[i] = preprocess(path, self.imgsz)
        self.index += len(batch)
        self.driver.memcpy_htod(self.device, self.host)
        return [int(self.device.value)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            return Path(self.cache_file).read_bytes()
        return None

    def write_calibration_cache(self, cache):
        Path(self.cache_file).write_bytes(cache)

    def __del__(self):
        try:
            self.driver.mem_free(self.device)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Build a TensorRT INT8 engine from ONNX using image calibration.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--workspace", type=int, default=512, help="Workspace in MiB.")
    parser.add_argument("--fp16", action="store_true", help="Allow FP16 fallback in addition to INT8.")
    args = parser.parse_args()

    image_paths = list_images(args.calib_dir)
    if not image_paths:
        raise RuntimeError(f"No calibration images found in {args.calib_dir}")
    print(f"Using {len(image_paths)} calibration images")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(Path(args.onnx).read_bytes()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.max_workspace_size = args.workspace * (1 << 20)
    config.set_flag(trt.BuilderFlag.INT8)
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = ImageEntropyCalibrator(
        image_paths=image_paths,
        cache_file=args.cache,
        batch_size=args.batch,
        imgsz=args.imgsz,
    )

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("TensorRT returned no engine")
    Path(args.engine).write_bytes(engine.serialize())
    print(f"Wrote {args.engine}")


if __name__ == "__main__":
    main()
