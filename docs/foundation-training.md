# 视觉基础模型的病理微调

这条训练路线使用 **DINOv2 ViT-B/14 + LoRA + 二分类头**。DINOv2 是约 8600 万参数的通用视觉基础模型，本项目把它适配到病理图块任务。这里没有从零预训练，也没有训练 DeepSeek 的语言模型权重。DeepSeek API 仍负责基于分析报告的工具调用问答。

## 模型与方法

- 上游：[Meta DINOv2](https://github.com/facebookresearch/dinov2)，[模型卡](https://huggingface.co/facebook/dinov2-base)。权重遵循 Apache-2.0，代码仓库的 MIT 许可不覆盖上游权重。
- 固定版本：`f9e44c814b77203eaa57a6bdbbd535f21ede1415`。只加载 safetensors，不执行远程模型代码。
- 冻结原始编码器，在每层 attention 的 query/value 线性层加入 LoRA，默认 rank=4、alpha=8，同时训练新的二分类头。
- 前向计算：`W(x) + (alpha/rank) * B(A(x))`。A 随机初始化，B 初始化为零；因此初始适配器不改变原始投影。
- 使用上游图像处理器（短边缩放 256、中心裁剪 224、ImageNet 通道归一化），不叠加项目的 RGB 染色匹配。训练与推理复用同一处理方式。
- AdamW、训练集计算的正类权重、梯度裁剪；按验证集加权 BCE 选择最佳轮次，再使用固定 0.5 阈值评估测试集一次。
- 保存可训练参数和校验值，不重复保存整个基础模型。实验记录包括参数量、优化步数、LoRA 权重变化、划分、随机种子、损失和测试混淆矩阵。

## 运行

使用 Python 3.11+。先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与显卡驱动兼容的版本。CPU 也能执行，但较慢。模型缓存、临时下载和训练环境需要放在有足够空间的磁盘。

```bash
pip install -e ".[foundation,dev]"
python scripts/prepare_pathmnist.py --download
python -m scripts.train_foundation --manifest data/pathmnist/manifest.csv --split-policy official --epochs 3 --batch-size 4 --device auto --output weights/dinov2-lora
```

首次训练从 Hugging Face 下载固定版本。可设置 `HF_HOME` 改变缓存位置；或用 `--base-path` 指定本地快照目录，目录必须含固定版本的 `config.json`、`preprocessor_config.json`、`model.safetensors`，代码会校验三个文件的 SHA256。

`status.json` 随训练更新。成功完成后输出：

| 文件 | 内容 |
| --- | --- |
| `adapter.pt` | LoRA 与分类头的训练后权重 |
| `model.json` | 基础模型版本、LoRA 配置和权重校验值 |
| `metrics.json` | 实际训练与测试结果 |
| `status.json` | 训练状态、进度或失败类型 |

每次实验使用新输出目录，防止覆盖既有结果。中断的实验可以保留已选出的适配器供排错，但只有完整成功的实验才生成部署所需 `model.json`；目前不支持断点恢复。

## 接入分析工作台

在本机 `.env` 设置：

```dotenv
MODEL_BACKEND=foundation
MODEL_WEIGHTS=weights/dinov2-lora
# 如使用本地快照，填写其目录；否则留空，复用训练时的 Hugging Face 缓存
FOUNDATION_BASE_PATH=
```

使用安装了 foundation 依赖的 Python 重启服务。推理只读本地权重/缓存，缺失会明确失败，不会退回规则分数。工作台显示模型模式、训练摘要和研究限制；报告保留模型版本与适配器 SHA256。`/api/training` 提供当前配置实验的只读进度/指标，不暴露本地路径或密钥，也不能从网页任意启动训练。

## 本次实验的适用范围

公开 PathMNIST 子集：训练 900、验证 270、测试 270 张，保留官方划分及原始来源索引。标签为原类别 8（结直肠腺癌上皮）对其余类别；其余类别不能一概称为“健康组织”。

原始图片只有 28×28，放大到模型输入尺寸不会恢复细胞细节。这是验证训练与部署链路的小规模实验；图块测试结果不能解释为整张切片或患者诊断性能。数据不提供逐图块患者 ID，无法自行审计患者级独立性。医院数据尚未接入，不能声称已做医院训练或临床验证。

加入医院数据时使用 `path,label,patient_id,split` 清单并保留默认 `--split-policy patient`。先按患者/机构划分，再切块；不要把同一患者图块放到多个集合。原始数据、清单、权重和密钥留在本地。公开仓库只发布获准公开的数据样本与实验汇总。

参见 [公开数据来源与许可](public-data.md)。
