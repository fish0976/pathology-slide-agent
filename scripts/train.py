"""Small CNN baseline. Manifest columns: path,label,patient_id,split.

Run: python scripts/train.py --manifest data/manifest.csv --epochs 5
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pathology.imaging import normalize_stain
from pathology.model import build_network


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest is empty")
    patients = {}
    for row in rows:
        if row.get("split") not in {"train", "val", "test"} or row.get("label") not in {"0", "1"}:
            raise ValueError("Each row needs split=train/val/test and label=0/1")
        patient = row.get("patient_id", "").strip()
        if not patient or (patient in patients and patients[patient] != row["split"]):
            raise ValueError("Missing patient_id or patient leakage across splits")
        patients[patient] = row["split"]
        row["path"] = str((path.parent / row["path"]).resolve())
    for split in ("train", "val", "test"):
        if {r["label"] for r in rows if r["split"] == split} != {"0", "1"}:
            raise ValueError(f"{split} must contain both classes")
    return rows


def binary_metrics(labels, scores, threshold=0.5):
    if not labels or len(labels) != len(scores):
        raise ValueError("Nonempty, matching labels and scores required")
    if not 0 <= threshold <= 1 or any(y not in (0, 1) for y in labels):
        raise ValueError("Invalid labels or threshold")
    if any(not np.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Scores must be finite and within [0,1]")
    pred = [s >= threshold for s in scores]
    tp = sum(y == 1 and p for y, p in zip(labels, pred))
    tn = sum(y == 0 and not p for y, p in zip(labels, pred))
    fp = sum(y == 0 and p for y, p in zip(labels, pred))
    fn = sum(y == 1 and not p for y, p in zip(labels, pred))
    ratio = lambda n, d: n / d if d else None
    return {
        "n": len(labels),
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def main():
    import torch
    from torch.utils.data import DataLoader, Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("weights/model.pt"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    torch.manual_seed(42)
    rows = read_manifest(args.manifest)

    class Patches(Dataset):
        def __init__(self, split):
            self.rows = [r for r in rows if r["split"] == split]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(row["path"]) as image:
                image = normalize_stain(image.convert("RGB")).resize((128, 128))
                arr = np.asarray(image, dtype=np.float32) / 255
            return torch.from_numpy(arr.transpose(2, 0, 1).copy()), torch.tensor([float(row["label"])])

    loaders = {
        s: DataLoader(Patches(s), batch_size=args.batch_size, shuffle=s == "train")
        for s in ("train", "val", "test")
    }
    net = build_network()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        net.train()
        for images, labels in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
        net.eval()
        val_loss = 0
        with torch.inference_mode():
            for images, labels in loaders["val"]:
                val_loss += criterion(net(images), labels).item() * len(images)
        val_loss /= len(loaders["val"].dataset)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), args.output)
        print(json.dumps({"epoch": epoch + 1, "val_loss": val_loss}))
    net.load_state_dict(torch.load(args.output, map_location="cpu", weights_only=True))
    net.eval()
    labels, scores = [], []
    with torch.inference_mode():
        for images, target in loaders["test"]:
            labels.extend(target.flatten().int().tolist())
            scores.extend(net(images).sigmoid().flatten().tolist())
    metrics = binary_metrics(labels, scores)
    metrics.update(
        unit="patch",
        model="small-cnn-baseline",
        seed=42,
        note="Independent patient split; patch metrics are not clinical validation.",
    )
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
