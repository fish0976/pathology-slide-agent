"""Small CNN baseline. Manifest columns: path,label,patient_id,split.

Run: python scripts/train.py --manifest data/manifest.csv --epochs 5
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pathology.imaging import normalize_stain
from pathology.model import build_network


def read_manifest(path, split_policy="patient"):
    if split_policy not in {"patient", "official"}:
        raise ValueError("Unknown split policy")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest is empty")
    patients = {}
    seen_paths = set()
    for row in rows:
        if row.get("split") not in {"train", "val", "test"} or row.get("label") not in {"0", "1"}:
            raise ValueError("Each row needs split=train/val/test and label=0/1")
        patient = row.get("patient_id", "").strip()
        if (split_policy == "patient" and not patient) or (
            patient and patient in patients and patients[patient] != row["split"]
        ):
            raise ValueError("Missing patient_id or patient leakage across splits")
        if split_policy == "official" and row.get("split_policy") != "official":
            raise ValueError("Official split mode requires explicit split_policy=official on every row")
        if patient:
            patients[patient] = row["split"]
        row["path"] = str((path.parent / row["path"]).resolve())
        if row["path"] in seen_paths:
            raise ValueError("Duplicate image path in manifest")
        seen_paths.add(row["path"])
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
    parser.add_argument("--split-policy", choices=["patient", "official"], default="patient")
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("epochs, batch-size and threads must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    rows = read_manifest(args.manifest, args.split_policy)

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
    train_rows = [r for r in rows if r["split"] == "train"]
    positives = sum(r["label"] == "1" for r in train_rows)
    positive_weight = (len(train_rows) - positives) / positives if args.balance_classes else 1.0
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]))
    best_loss = float("inf")
    history = []
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
        history.append({"epoch": epoch + 1, "val_loss": val_loss})
        print(json.dumps(history[-1]), flush=True)
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
        split_policy=args.split_policy,
        patient_ids_available=all(bool(r.get("patient_id")) for r in rows),
        data_origins=sorted({r.get("data_origin", "unspecified") for r in rows}),
        samples={split: sum(r["split"] == split for r in rows) for split in ("train", "val", "test")},
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        weights_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        epochs=args.epochs,
        positive_weight=positive_weight,
        history=history,
        note="Measured patch metrics only; not clinical validation. Missing patient IDs prevent patient-level split auditing.",
    )
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
