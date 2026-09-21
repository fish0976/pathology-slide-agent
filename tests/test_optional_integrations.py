"""Real library smoke tests; skipped when optional extras are absent."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pathology.config import Settings
from pathology.model import TorchPredictor
from pathology.pipeline import analyze
from pathology.slides import Slide


def test_openslide_reads_tiled_pyramid(tmp_path):
    pytest.importorskip("openslide")
    tifffile = pytest.importorskip("tifffile")
    pixels = np.full((768, 1024, 3), [185, 110, 165], dtype=np.uint8)
    pixels[100:300, 200:500] = [95, 55, 130]
    path = tmp_path / "generated.tiff"
    with tifffile.TiffWriter(path) as writer:
        writer.write(pixels, tile=(256, 256), photometric="rgb", metadata=None)
        writer.write(pixels[::2, ::2], tile=(256, 256), photometric="rgb", subfiletype=1, metadata=None)
    with Slide(path) as slide:
        assert slide.dimensions == (1024, 768)
        assert slide.level_dimensions[1] == (512, 384)
        assert slide.read(200, 100, 1, (50, 50)).size == (50, 50)
    report = analyze(
        path,
        tmp_path,
        Settings(data_dir=tmp_path),
        {"level": 1, "tile_size": 128, "max_tiles": 12, "normalize": True},
    )
    assert report["slide"]["analysis_level"] == 1
    assert all(t["width"] == 256 for t in report["tiles"])
    assert report["sampling"]["total_grid_tiles"] == 12


def test_torch_training_checkpoint_and_inference(tmp_path):
    pytest.importorskip("torch")
    rows = ["path,label,patient_id,split"]
    for split in ("train", "val", "test"):
        for label in (0, 1):
            filename = f"{split}-{label}.png"
            Image.new("RGB", (64, 64), (220, 170, 195) if label == 0 else (130, 65, 150)).save(
                tmp_path / filename
            )
            rows.append(f"{filename},{label},{split}-{label},{split}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows), encoding="utf-8")
    output = tmp_path / "model.pt"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--manifest",
            str(manifest),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    assert result.returncode == 0, result.stderr
    metrics = json.loads(output.with_suffix(".metrics.json").read_text())
    assert metrics["n"] == 2
    assert metrics["unit"] == "patch"
    predictor = TorchPredictor(output)
    assert 0 <= predictor.predict(Image.new("RGB", (64, 64), (130, 65, 150))) <= 1
