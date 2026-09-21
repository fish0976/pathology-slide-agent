"""DINOv2 vision foundation model with trainable query/value LoRA adapters.

The upstream model is a general vision encoder, not a language model or a
pathology-pretrained encoder. No remote model code is executed.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

BASE_MODEL = "facebook/dinov2-base"
BASE_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
BASE_FILES = {
    "config.json": "f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    "model.safetensors": "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
}


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale


class FoundationClassifier(nn.Module):
    def __init__(self, backbone, rank=4, alpha=8):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        targets = [
            (name, module)
            for name, module in backbone.named_modules()
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in {"query", "value"}
        ]
        if not targets:
            raise ValueError("Backbone has no supported query/value attention projections")
        for name, module in targets:
            parent, _, child = name.rpartition(".")
            setattr(backbone.get_submodule(parent), child, LoRALinear(module, rank, alpha))
        self.head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, pixel_values):
        return self.head(self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0])

    def adapter_state(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_adapter(self, state):
        expected = self.adapter_state()
        if state.keys() != expected.keys() or any(state[n].shape != expected[n].shape for n in expected):
            raise ValueError("Adapter keys/shapes do not match model")
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name in state:
                    parameter.copy_(state[name])


def load_base(base_path=None, local_only=False):
    from transformers import AutoImageProcessor, AutoModel

    if base_path:
        for name, expected in BASE_FILES.items():
            with (Path(base_path) / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Pinned base model checksum mismatch: {name}")
    source = str(base_path) if base_path else BASE_MODEL
    kwargs = {"local_files_only": local_only, "trust_remote_code": False}
    if not base_path:
        kwargs["revision"] = BASE_REVISION
    processor = AutoImageProcessor.from_pretrained(source, use_fast=False, **kwargs)
    backbone = AutoModel.from_pretrained(source, use_safetensors=True, **kwargs)
    if backbone.config.model_type != "dinov2":
        raise ValueError("Only DINOv2 backbones are supported")
    return backbone, processor


class FoundationPredictor:
    mode = "foundation"
    label = "DINOv2 LoRA 阳性类别研究分数（未校准）"
    handles_preprocessing = True

    def __init__(self, path, base_path=None):
        path = Path(path)
        metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if metadata["base_model"] != BASE_MODEL or metadata["base_revision"] != BASE_REVISION:
            raise ValueError("Unsupported base model/revision")
        adapter_path = path / "adapter.pt"
        if hashlib.sha256(adapter_path.read_bytes()).hexdigest() != metadata["adapter_sha256"]:
            raise ValueError("Adapter checksum mismatch")
        backbone, self.processor = load_base(base_path, local_only=True)
        self.net = FoundationClassifier(backbone, metadata["rank"], metadata["alpha"])
        self.net.load_adapter(torch.load(adapter_path, map_location="cpu", weights_only=True))
        self.net.eval()
        self.metadata = {k: metadata[k] for k in ("base_model", "base_revision", "method", "adapter_sha256")}

    def predict(self, image):
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.inference_mode():
            return float(self.net(inputs["pixel_values"]).sigmoid().item())
