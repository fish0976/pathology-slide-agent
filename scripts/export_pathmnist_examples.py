"""Export nine attributed test images and an actual public-data experiment record."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np

from scripts.prepare_pathmnist import LABELS, MD5


def export_examples(dataset, metrics_path, output):
    import torch

    metadata = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metadata.get("archive_md5") != MD5 or metrics.get("data_origins") != ["public_pathmnist"]:
        raise ValueError("Only the verified public PathMNIST experiment may be exported")
    if metrics.get("manifest_sha256") != hashlib.sha256((dataset / "manifest.csv").read_bytes()).hexdigest():
        raise ValueError("Experiment metrics do not belong to this dataset manifest")
    if output.exists():
        raise FileExistsError("Example directory already exists")
    with (dataset / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [
        next(r for r in rows if r["split"] == "test" and int(r["source_label"]) == label) for label in LABELS
    ]
    (output / "images").mkdir(parents=True)
    index = []
    for label, row in zip(LABELS, selected):
        filename = f"images/class_{label}.png"
        source = (dataset / row["path"]).resolve()
        if not source.is_relative_to(dataset.resolve()):
            raise ValueError("Manifest path leaves the public dataset directory")
        shutil.copyfile(source, output / filename)
        index.append(
            {
                "file": filename,
                "official_split": "test",
                "official_index": int(row["source_index"]),
                "original_label": label,
                "original_name": LABELS[label],
                "binary_label": int(label == 8),
            }
        )
    (output / "index.json").write_text(
        json.dumps({"source": metadata, "images": index}, indent=2), encoding="utf-8"
    )
    record = {
        "experiment": "public-pathmnist-small-cnn",
        "dataset": metadata,
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "clinical_validation": False,
    }
    (output / "experiment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    lines = [
        "# 公开 PathMNIST 样本与训练记录",
        "",
        "这里的九张图片来自官方 PathMNIST 测试划分，每个原始类别一张。它们是公开研究图块，不是自有医院数据。",
        "",
        "图像原始尺寸为 28×28；下表仅放大显示，没有增加图像细节。",
        "",
        "| 原标签 | 样本 | 来源索引（test） |",
        "| --- | --- | --- |",
    ]
    for item in index:
        lines.append(
            f"| {item['original_label']} · {item['original_name']} | "
            f'<img src="{item["file"]}" width="84" height="84" alt="{item["original_name"]}"> | '
            f"{item['official_index']} |"
        )
    lines.extend(
        [
            "",
            "## 来源与使用许可",
            "",
            "PathMNIST / MedMNIST v2：Yang, Shi, Wei 等；上游病理数据：Kather, Halama, Marx。",
            (
                "[官方数据](https://zenodo.org/records/10519652) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · "
                "[完整署名与修改说明](../../THIRD_PARTY_NOTICES.md)"
            ),
            "",
            "选择子集并无损导出 PNG，保留来源索引和九分类标签；训练时增加类别 8 对其余类别的二分类目标。",
            "",
            "## 本次实际运行",
            "",
            f"训练 / 验证 / 测试：{metadata['samples']['train']} / {metadata['samples']['val']} / {metadata['samples']['test']} 张。",
            f"小型 CNN 训练 {metrics['epochs']} 轮，根据验证损失选模；测试阈值固定为 0.5。",
            "",
            "| 指标 | 实测值 |",
            "| --- | --- |",
        ]
    )
    for name in ("accuracy", "sensitivity", "specificity", "precision", "f1", "tp", "tn", "fp", "fn"):
        value = metrics[name]
        display = (
            "未定义（分母为零）"
            if value is None
            else f"{value:.4f}"
            if isinstance(value, float)
            else str(value)
        )
        lines.append(f"| {name} | {display} |")
    if metrics["tp"] == 0 and metrics["fn"] > 0:
        lines.extend(
            [
                "",
                "**本次基线没有检出阳性样本。准确率受负类占比影响，不能据此称模型有效。**",
                "这次运行仅验证公开数据、训练、选模、评估和结果追溯链路；后续模型改进需要新的预先定义实验。",
            ]
        )
    lines.extend(
        [
            "",
            "完整参数、训练曲线、校验值和环境版本见 [experiment.json](experiment.json)。",
            "这些数字是本项目运行所得，未采用论文的成绩；不是原始九分类基准成绩，也不是临床诊断准确率。",
            "NPZ 不提供患者身份索引，采用原始官方划分，不声称已经重新核验患者级独立性。",
            "",
            "复现步骤见 [公开数据文档](../../docs/public-data.md)。模型权重与完整数据包保存在本地，不加入仓库。",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/pathmnist"))
    parser.add_argument("--metrics", type=Path, default=Path("weights/pathmnist.metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("examples/pathmnist"))
    args = parser.parse_args()
    record = export_examples(args.dataset, args.metrics, args.output)
    print(json.dumps({"examples": 9, "test_samples": record["metrics"]["n"]}))


if __name__ == "__main__":
    main()
