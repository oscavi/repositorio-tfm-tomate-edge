import torch
from ultralytics import YOLO


def main():
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device_count={torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"device_name={torch.cuda.get_device_name(0)}")
        print(f"capability={torch.cuda.get_device_capability(0)}")
        x = torch.randn(2048, 2048, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"cuda_matmul_ok={tuple(y.shape)} mean={y.float().mean().item():.6f}")

    model = YOLO("yolov8n.pt")
    print(f"ultralytics_model_loaded={model.model.__class__.__name__}")


if __name__ == "__main__":
    main()
