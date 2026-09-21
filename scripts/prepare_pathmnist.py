"""Download verified public PathMNIST data and retain its official splits.

The 28x28 images are public research patches, not this project's hospital data.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

URL = "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
MD5 = "a8b06965200029087d5bd730944a56c1"
LABELS = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}


def checksum(path):
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(path):
    if path.exists():
        if checksum(path) != MD5:
            raise ValueError(
                "Existing archive checksum mismatch; preserve it and choose another archive path."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    if temporary.exists():
        raise FileExistsError(
            "Partial download already exists; use a different archive path or remove it manually."
        )
    with httpx.stream("GET", URL, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("xb") as stream:
            received, next_notice = 0, 25_000_000
            for chunk in response.iter_bytes(1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                if received >= next_notice:
                    print(f"Downloaded {received / 1_000_000:.0f} MB", flush=True)
                    next_notice += 25_000_000
    if checksum(temporary) != MD5:
        raise ValueError("Downloaded archive checksum does not match the official release.")
    temporary.rename(path)


def extract_subset(archive, output, counts, seed=42):
    if output.exists():
        raise FileExistsError("Output already exists; choose a new folder to preserve existing data.")
    if checksum(archive) != MD5:
        raise ValueError("Use the official PathMNIST archive with the documented checksum.")
    if any(n < 1 for n in counts.values()):
        raise ValueError("Per-class sample counts must be positive.")
    rng = np.random.default_rng(seed)
    selections = []
    with np.load(archive, allow_pickle=False) as data:
        for split in ("train", "val", "test"):
            images, labels = data[f"{split}_images"], data[f"{split}_labels"].reshape(-1)
            if images.shape[1:] != (28, 28, 3) or len(labels) != len(images):
                raise ValueError("Unexpected image or label array shape.")
            for label in LABELS:
                candidates = np.flatnonzero(labels == label)
                if len(candidates) < counts[split]:
                    raise ValueError(f"Not enough images in {split}, class {label}.")
                indices = sorted(rng.choice(candidates, counts[split], replace=False).tolist())
                for index in indices:
                    selections.append((split, label, index, images[index].copy()))
    output.mkdir(parents=True)
    rows = []
    for split, label, index, pixels in selections:
        relative = f"patches/{split}/{label}_{index:06d}.png"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(destination)
        rows.append(
            {
                "path": relative,
                "label": int(label == 8),
                "patient_id": "",
                "split": split,
                "source_index": index,
                "source_label": label,
                "source_name": LABELS[label],
                "data_origin": "public_pathmnist",
                "split_policy": "official",
            }
        )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": "PathMNIST",
        "source": URL,
        "archive_md5": MD5,
        "release": "Zenodo 10519652",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "seed": seed,
        "split_policy": "official",
        "patient_ids_available": False,
        "samples": {split: counts[split] * 9 for split in counts},
        "task": "binary adaptation: class 8 vs classes 0-7",
        "positive_class": LABELS[8],
        "changes": "Stratified subset; lossless PNG export; added binary labels. Original labels retained.",
        "note": "28x28 research patches; no local hospital data. Patient-level independence cannot be audited from this archive.",
    }
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/pathmnist.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/pathmnist"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--val-per-class", type=int, default=30)
    parser.add_argument("--test-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.download:
        download_archive(args.archive)
    metadata = extract_subset(
        args.archive,
        args.output,
        {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class},
        args.seed,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
