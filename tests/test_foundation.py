"""Offline tests: tiny random DINOv2, never downloading a model in CI."""

import hashlib
import json

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pathology import foundation
from pathology.config import Settings
from pathology.pipeline import analyze


def tiny_base():
    config = transformers.Dinov2Config(
        image_size=28,
        patch_size=14,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=3,
    )
    backbone = transformers.Dinov2Model(config)
    processor = transformers.BitImageProcessor(
        size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28}
    )
    return backbone, processor


def test_lora_updates_without_changing_base_and_roundtrips(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    backbone, processor = tiny_base()
    original = {n: p.clone() for n, p in backbone.named_parameters()}
    net = foundation.FoundationClassifier(backbone, rank=2, alpha=4)
    initial = net.adapter_state()
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) < sum(
        p.numel() for p in net.parameters()
    )
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=0.01)
    image = Image.new("RGB", (28, 28), (100, 50, 150))
    inputs = processor(images=image, return_tensors="pt")["pixel_values"]
    for _ in range(2):
        optimizer.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(net(inputs), torch.ones(1, 1)).backward()
        optimizer.step()
    after = net.adapter_state()
    assert any(not torch.equal(initial[n], after[n]) for n in after if n.startswith("backbone."))
    for name, before in original.items():
        path = name.replace(".query.", ".query.base.").replace(".value.", ".value.base.")
        assert torch.equal(before, dict(backbone.named_parameters())[path])
    net.eval()
    expected = net(inputs).sigmoid().item()
    torch.save(after, tmp_path / "adapter.pt")
    metadata = {
        "base_model": foundation.BASE_MODEL,
        "base_revision": foundation.BASE_REVISION,
        "rank": 2,
        "alpha": 4,
        "method": "LoRA query/value + binary head",
        "adapter_sha256": hashlib.sha256((tmp_path / "adapter.pt").read_bytes()).hexdigest(),
    }
    (tmp_path / "model.json").write_text(json.dumps(metadata))

    def reload_base(*args, **kwargs):
        fresh, _ = tiny_base()
        fresh.load_state_dict(original)
        return fresh, processor

    monkeypatch.setattr(foundation, "load_base", reload_base)
    predictor = foundation.FoundationPredictor(tmp_path)
    assert predictor.predict(image) == pytest.approx(expected, abs=1e-6)
    image.save(tmp_path / "image.png")
    report = analyze(
        tmp_path / "image.png",
        tmp_path,
        Settings(data_dir=tmp_path, model_backend="foundation", model_weights=str(tmp_path)),
        {"level": 0, "tile_size": 28, "max_tiles": 1, "normalize": True},
    )
    assert report["mode"] == "foundation"
    assert report["normalization"] == "upstream_image_processor_no_stain_matching"
    assert report["tiles"][0]["score"] == pytest.approx(expected, abs=0.0001)
    assert report["model_provenance"]["adapter_sha256"] == metadata["adapter_sha256"]
    (tmp_path / "adapter.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        foundation.FoundationPredictor(tmp_path)


def test_adapter_rejects_incomplete_checkpoint():
    backbone, _ = tiny_base()
    net = foundation.FoundationClassifier(backbone)
    with pytest.raises(ValueError, match="keys/shapes"):
        net.load_adapter({})


def test_training_records_real_steps_and_serves_only_public_summary(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pathology.main import create_app
    from scripts import train_foundation

    lines = ["path,label,patient_id,split"]
    for split in ("train", "val", "test"):
        for label in (0, 1):
            name = f"{split}-{label}.png"
            Image.new("RGB", (28, 28), (90 + label * 100, 80, 120)).save(tmp_path / name)
            lines.append(f"{name},{label},{split}-{label},{split}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(lines))
    output = tmp_path / "run"
    monkeypatch.setattr(train_foundation, "load_base", lambda *_: tiny_base())
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_foundation",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
    )
    train_foundation.main()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["optimizer_steps"] == 1
    assert metrics["lora_weight_change_l2"] > 0
    assert metrics["test"]["n"] == 2
    assert metrics["samples"] == {"train": 2, "val": 2, "test": 2}
    assert metrics["patient_ids_available"]
    # This private-looking field must never appear in the HTTP summary.
    metrics["local_path"] = "private/patient-data"
    (output / "metrics.json").write_text(json.dumps(metrics))
    app = create_app(
        Settings(data_dir=tmp_path / "app", model_backend="foundation", model_weights=str(output))
    )
    with TestClient(app) as client:
        summary = client.get("/api/training").json()
    assert summary["state"] == "completed"
    assert summary["test"]["n"] == 2
    assert "local_path" not in summary
    with pytest.raises(SystemExit):
        train_foundation.main()
