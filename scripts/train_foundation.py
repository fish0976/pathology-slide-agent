"""Fine-tune a pinned DINOv2 vision encoder using LoRA on local pathology tiles."""

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pathology.foundation import BASE_MODEL, BASE_REVISION, FoundationClassifier, load_base
from scripts.train import binary_metrics, read_manifest


class Patches(Dataset):
    def __init__(self, rows, processor):
        self.rows, self.processor = rows, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            pixels = self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return pixels, torch.tensor([float(row["label"])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("weights/dinov2-lora"))
    parser.add_argument("--base-path", type=Path, help="Optional local pinned model snapshot")
    parser.add_argument("--split-policy", choices=["patient", "official"], default="patient")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.rank, args.threads, args.lr, args.alpha) <= 0:
        parser.error("Training hyperparameters must be positive")
    if args.output.exists():
        parser.error("Output directory already exists; use a new experiment directory")
    rows = read_manifest(args.manifest, args.split_policy)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    backbone, processor = load_base(args.base_path)
    net = FoundationClassifier(backbone, args.rank, args.alpha).to(device)
    initial = net.adapter_state()
    trainable = [p for p in net.parameters() if p.requires_grad]
    counts = {
        "total": sum(p.numel() for p in net.parameters()),
        "trainable": sum(p.numel() for p in trainable),
    }
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    loaders = {
        s: DataLoader(Patches(r, processor), batch_size=args.batch_size, shuffle=s == "train")
        for s, r in splits.items()
    }
    positives = sum(r["label"] == "1" for r in splits["train"])
    positive_weight = (len(splits["train"]) - positives) / positives
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    args.output.mkdir(parents=True)
    status_path = args.output / "status.json"

    def status(**values):
        values["base_model"] = BASE_MODEL
        values["epochs"] = args.epochs
        values["updated_at"] = datetime.now(UTC).isoformat()
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary.replace(status_path)
        print(json.dumps(values), flush=True)

    history, best_loss, best_epoch, updates = [], float("inf"), 0, 0
    status(state="training", device=device, parameters=counts, epochs=args.epochs)
    try:
        for epoch in range(1, args.epochs + 1):
            net.train()
            train_loss = 0.0
            for step, (images, labels) in enumerate(loaders["train"], 1):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(net(images), labels)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                train_loss += loss.item() * len(images)
                if step == 1 or step % 20 == 0:
                    status(
                        state="training",
                        epoch=epoch,
                        epochs=args.epochs,
                        step=step,
                        steps=len(loaders["train"]),
                        optimizer_steps=updates,
                        loss=loss.item(),
                    )
            net.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for images, labels in loaders["val"]:
                    val_loss += criterion(net(images.to(device)), labels.to(device)).item() * len(images)
            val_loss /= len(splits["val"])
            history.append(
                {"epoch": epoch, "train_loss": train_loss / len(splits["train"]), "val_loss": val_loss}
            )
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                torch.save(net.adapter_state(), args.output / "adapter.pt")
            status(state="validating", **history[-1], best_epoch=best_epoch)
        net.load_adapter(torch.load(args.output / "adapter.pt", map_location="cpu", weights_only=True))
        net.eval()
        labels, scores = [], []
        with torch.inference_mode():
            for images, targets in loaders["test"]:
                scores.extend(net(images.to(device)).sigmoid().flatten().cpu().tolist())
                labels.extend(targets.flatten().int().tolist())
        final = net.adapter_state()
        lora_change = (
            sum((final[n] - initial[n]).square().sum().item() for n in final if n.startswith("backbone."))
            ** 0.5
        )
        if lora_change == 0:
            raise ValueError("LoRA weights did not change")
        metadata = {
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "method": "LoRA query/value + binary head",
            "rank": args.rank,
            "alpha": args.alpha,
            "preprocessing": "upstream_image_processor_no_stain_matching",
            "adapter_sha256": hashlib.sha256((args.output / "adapter.pt").read_bytes()).hexdigest(),
        }
        (args.output / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metrics = {
            **metadata,
            "test": binary_metrics(labels, scores),
            "unit": "patch",
            "seed": args.seed,
            "parameters": counts,
            "lora_weight_change_l2": lora_change,
            "optimizer_steps": updates,
            "history": history,
            "best_epoch": best_epoch,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "positive_weight": positive_weight,
            "samples": {s: len(r) for s, r in splits.items()},
            "split_policy": args.split_policy,
            "patient_ids_available": all(bool(r.get("patient_id")) for r in rows),
            "data_origins": sorted({r.get("data_origin", "unspecified") for r in rows}),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "device": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "completed_at": datetime.now(UTC).isoformat(),
            "note": "General-vision foundation model adapted to pathology, not an LLM. Small public-data pilot; not clinical validation.",
        }
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        status(state="completed", test=metrics["test"], elapsed_seconds=metrics["elapsed_seconds"])
    except BaseException as exc:
        status(state="failed", error_type=type(exc).__name__)
        raise


if __name__ == "__main__":
    main()
