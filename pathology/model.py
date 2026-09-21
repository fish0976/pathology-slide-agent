import numpy as np


class DemoPredictor:
    mode = "demo"
    label = "颜色纹理演示分数（非肿瘤概率）"

    def predict(self, image):
        arr = np.asarray(image, dtype=np.float32)
        purple = (arr[:, :, 0] > arr[:, :, 1] + 15) & (arr[:, :, 2] > arr[:, :, 1] + 10)
        darkness = 1 - arr.mean() / 255
        return float(np.clip(0.55 * purple.mean() + 0.8 * darkness, 0, 1))


def build_network():
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 1),
    )


class TorchPredictor:
    mode = "torch"
    label = "研究模型阳性类别分数（未校准）"

    def __init__(self, path):
        import torch

        self.torch = torch
        self.net = build_network()
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()

    def predict(self, image):
        arr = np.asarray(image.resize((128, 128)), dtype=np.float32) / 255
        tensor = self.torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)
        with self.torch.inference_mode():
            return float(self.net(tensor).sigmoid().item())


def load_predictor(settings):
    if settings.model_backend == "demo":
        return DemoPredictor()
    if settings.model_backend == "torch":
        return TorchPredictor(settings.model_weights)
    if settings.model_backend == "foundation":
        from .foundation import FoundationPredictor

        return FoundationPredictor(settings.model_weights, settings.foundation_base_path or None)
    raise ValueError("MODEL_BACKEND 仅支持 demo、torch 或 foundation。")
