import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pathology.config import Settings
from pathology.imaging import normalize_stain, quality
from pathology.main import create_app
from pathology.slides import tile_grid
from pathology.store import Store
from scripts.train import binary_metrics, read_manifest


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_upload_mb=1))) as c:
        yield c


def upload_image(client, color="white", size=(256, 256)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    response = client.post("/api/slides", files={"file": ("slide.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 201
    return response.json()["id"]


def finish(client, job_id):
    for _ in range(150):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.03)
    pytest.fail("analysis did not terminate")


def test_demo_end_to_end(client):
    job_id = client.post("/api/demo").json()["id"]
    assert client.post(f"/api/jobs/{job_id}/analyze", json={"max_tiles": 24}).status_code == 202
    job = finish(client, job_id)
    assert job["status"] == "completed"
    report = job["report"]
    assert report["mode"] == "demo"
    assert report["sampling"]["analyzed_tiles"] > 0
    assert 0 < report["sampling"]["grid_coverage"] < 1
    for tile in report["tiles"]:
        assert 0 <= tile["score"] <= 1
        assert tile["x"] + tile["width"] <= report["slide"]["width"]
    assert len(report["trace"]) == 5
    for asset in ("thumbnail.png", "heatmap.png"):
        response = client.get(f"/api/jobs/{job_id}/assets/{asset}")
        assert response.status_code == 200
        assert Image.open(io.BytesIO(response.content)).width <= 1200
    for format in ("json", "md"):
        response = client.get(f"/api/jobs/{job_id}/report?format={format}")
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
    for q, tool in [
        ("质量质控", "get_quality"),
        ("高分区域在哪里", "get_regions"),
        ("报告摘要", "get_report"),
    ]:
        response = client.post(f"/api/jobs/{job_id}/chat", json={"question": q})
        assert response.json()["tools"] == [tool]


def test_white_slide_produces_no_regions(client):
    job_id = upload_image(client)
    client.post(f"/api/jobs/{job_id}/analyze", json={})
    job = finish(client, job_id)
    assert job["status"] == "completed"
    assert job["report"]["regions"] == []
    assert job["report"]["sampling"]["analyzed_tiles"] == 0


@pytest.mark.parametrize(
    "options", [{"max_tiles": 0}, {"max_tiles": 100000}, {"tile_size": 2}, {"level": -1}]
)
def test_invalid_options(client, options):
    job_id = upload_image(client)
    assert client.post(f"/api/jobs/{job_id}/analyze", json=options).status_code == 422


def test_invalid_level_becomes_failed_job(client):
    job_id = upload_image(client)
    client.post(f"/api/jobs/{job_id}/analyze", json={"level": 2})
    assert finish(client, job_id)["status"] == "failed"


def test_upload_limits_and_corruption(client):
    assert client.post("/api/slides", files={"file": ("a.exe", b"x")}).status_code == 415
    assert client.post("/api/slides", files={"file": ("bad.png", b"not an image")}).status_code == 422
    assert client.post("/api/slides", files={"file": ("big.png", b"x" * 1_048_577)}).status_code == 413
    assert client.get("/api/jobs").json() == []


def test_not_ready_and_missing(client):
    job_id = upload_image(client)
    assert client.get(f"/api/jobs/{job_id}/report").status_code == 409
    assert client.post(f"/api/jobs/{job_id}/chat", json={"question": "hi"}).status_code == 409
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/assets/source.png").status_code == 404
    assert client.post(f"/api/jobs/{job_id}/chat", json={"question": ""}).status_code == 422


def test_gigapixel_grid_is_bounded_and_reaches_edges():
    grid = list(tile_grid(1_000_001, 850_001, 256, 128))
    assert len(grid) == 128
    assert len(set(grid)) == 128
    assert grid[0][:2] == (0, 0)
    x, y, w, h = grid[-1]
    assert (x + w, y + h) == (1_000_001, 850_001)


def test_small_edge_tiles():
    assert list(tile_grid(19, 13, 256, 128)) == [(0, 0, 19, 13)]


def test_stain_normalization_preserves_background():
    im = Image.new("RGB", (40, 40), "white")
    im.paste((180, 90, 150), (10, 10, 30, 30))
    result = np.asarray(normalize_stain(im))
    assert (result[0, 0] == [255, 255, 255]).all()
    assert np.isfinite(result).all()
    assert quality(Image.new("RGB", (2, 2), "white"))["focus_variance"] == 0


def test_restart_marks_interrupted_tasks(tmp_path):
    store = Store(tmp_path)
    store.save({"id": "a", "status": "running"})
    assert Store(tmp_path).get("a")["status"] == "failed"


def test_binary_metrics_known_confusion_matrix():
    m = binary_metrics([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1])
    assert m["tp"] == m["tn"] == m["fp"] == m["fn"] == 1
    assert m["accuracy"] == m["sensitivity"] == m["specificity"] == 0.5
    assert binary_metrics([0], [0.2])["sensitivity"] is None
    with pytest.raises(ValueError):
        binary_metrics([1], [float("nan")])


def test_patient_leakage_rejected(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,label,patient_id,split\na.png,0,p1,train\nb.png,1,p1,test\n")
    with pytest.raises(ValueError, match="leakage"):
        read_manifest(manifest)


def test_torch_mode_never_silently_uses_demo(tmp_path):
    from pathology.model import load_predictor

    with pytest.raises((ImportError, FileNotFoundError)):
        load_predictor(Settings(data_dir=tmp_path, model_backend="torch", model_weights="absent.pt"))
