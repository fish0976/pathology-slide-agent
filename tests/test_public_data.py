import csv

import numpy as np
import pytest

from scripts import prepare_pathmnist
from scripts.train import read_manifest


def test_public_subset_preserves_source_splits_and_labels(tmp_path, monkeypatch):
    arrays = {}
    for split in ("train", "val", "test"):
        arrays[f"{split}_images"] = np.arange(18 * 28 * 28 * 3, dtype=np.uint8).reshape(18, 28, 28, 3)
        arrays[f"{split}_labels"] = np.repeat(np.arange(9), 2).reshape(-1, 1)
    archive = tmp_path / "fixture.npz"
    np.savez_compressed(archive, **arrays)
    monkeypatch.setattr(prepare_pathmnist, "MD5", prepare_pathmnist.checksum(archive))
    output = tmp_path / "subset"
    metadata = prepare_pathmnist.extract_subset(archive, output, {"train": 1, "val": 1, "test": 1})
    rows = read_manifest(output / "manifest.csv", split_policy="official")
    assert len(rows) == 27
    assert metadata["patient_ids_available"] is False
    assert all(int(r["label"]) == int(int(r["source_label"]) == 8) for r in rows)
    assert {r["source_name"] for r in rows} == set(prepare_pathmnist.LABELS.values())
    assert all(not r["patient_id"] for r in rows)
    with pytest.raises(ValueError, match="patient"):
        read_manifest(output / "manifest.csv")
    with pytest.raises(FileExistsError):
        prepare_pathmnist.extract_subset(archive, output, {"train": 1, "val": 1, "test": 1})


def test_public_archive_checksum_is_mandatory(tmp_path):
    archive = tmp_path / "bad.npz"
    archive.write_bytes(b"wrong data")
    with pytest.raises(ValueError, match="checksum"):
        prepare_pathmnist.extract_subset(archive, tmp_path / "result", {"train": 1, "val": 1, "test": 1})
    assert not (tmp_path / "result").exists()


def test_official_policy_does_not_invent_patient_identity(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "patient_id", "split", "split_policy"])
        writer.writerow(["a.png", 0, "", "train", ""])
    with pytest.raises(ValueError, match="explicit"):
        read_manifest(manifest, "official")
