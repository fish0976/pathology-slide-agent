# 公开病理数据与论文

医院原始切片、标注和内部训练资料不随本仓库发布。本项目使用已有开放许可的 PathMNIST 数据做工程复现，不将公开数据表述为自有医院数据。

## 数据来源

采用 MedMNIST v2 的 PathMNIST，文件来自官方 Zenodo 发布记录 **10519652**。数据为 28×28 RGB 病理图块，包含九类组织；适合轻量分类实验，不替代高分辨率 WSI。

- [官方数据下载](https://zenodo.org/records/10519652/files/pathmnist.npz?download=1)
- [官方标签、划分、许可与校验值](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)
- [上游 NCT-CRC-HE-100K / CRC-VAL-HE-7K 数据记录](https://zenodo.org/records/1214456)

下载文件大小约 206 MB，MD5 为 `a8b06965200029087d5bd730944a56c1`。脚本从官方地址下载并校验；完整压缩包只保存到本地，不提交 GitHub。

## 引用与许可

PathMNIST 标注为 [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)。本仓库转载的样本保留原标签、原划分与样本索引，标明无损 PNG 导出和二分类标签改编。图片许可独立于代码的 MIT License，不表示原作者认可本项目。

1. Yang, J., Shi, R., Wei, D., et al. **MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification**. *Scientific Data* 10, 41 (2023). [论文](https://doi.org/10.1038/s41597-022-01721-8)。本项目使用其标准化图块和划分方式。
2. Kather, J. N., et al. **Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study**. *PLOS Medicine* (2019). [论文](https://doi.org/10.1371/journal.pmed.1002730)。PathMNIST 的上游病理图块来源于该研究对应的数据资源。本项目没有复现该论文的生存预测任务。

## 本项目如何使用

- 保留官方 train / val / test，分别固定随机抽取每个原始类别 100 / 30 / 30 张，共 900 / 270 / 270 张。
- 原始九分类标签完整保留。为匹配本项目现有二分类接口，仅把类别 8 映射为 1，其余类别映射为 0；0 不代表“无肿瘤患者”或“良性”。
- 不虚构患者 ID。公开 NPZ 不提供每个图块的患者标识，清单该字段留空；训练显式指定 `--split-policy official`，报告注明无法从此文件审计患者级划分。
- 当前小型 CNN 使用加权 BCE、训练集类别权重、验证集选模以及固定 0.5 阈值。测试集只用于最终指标，不参与选模。
- `examples/pathmnist/` 包含九张带索引的公开测试样本和这次运行的指标，完整训练集及权重仍留在本地。

## 复现

```bash
pip install -e ".[ml]"
python scripts/prepare_pathmnist.py --download
python scripts/train.py --manifest data/pathmnist/manifest.csv --split-policy official --balance-classes --epochs 5 --output weights/pathmnist.pt
python -m scripts.export_pathmnist_examples
```

准备脚本不会覆盖已有目录；重复实验可指定 `--output data/pathmnist-run2`，已有校验正确的下载包会复用。训练权重、指标与校验值保存在 `weights/`。

用于本地推理时可设置 `MODEL_BACKEND=torch`、`MODEL_WEIGHTS=weights/pathmnist.pt` 并重启服务，但 28×28 图块实验的结果不能直接解释为 WSI 病变检出能力。默认工作台仍保留独立的规则演示模式，DeepSeek 负责报告证据问答。

## 结果口径

示例指标来自实际运行，不抄录论文准确率。它们只描述选定公开子集上的二分类图块实验，不能与原始九分类榜单直接比较，也不构成医院数据验证。尤其要同时看阳性召回、特异度和混淆矩阵，避免将大量负类带来的高准确率误判为模型能力。
