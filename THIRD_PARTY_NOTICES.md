# Third-party data notices

## DINOv2 vision foundation model

The optional foundation backend uses Meta's [DINOv2](https://github.com/facebookresearch/dinov2), distributed as [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base), pinned to revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`. Upstream model weights are Apache-2.0 licensed and are downloaded separately; they are not relicensed as MIT or bundled in this Git repository. LoRA modifies the attention projections for a local pathology experiment. This does not imply endorsement by Meta.

Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023. [Paper](https://arxiv.org/abs/2304.07193). [Upstream license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).

## PathMNIST image samples

Scope: PNG files under `examples/pathmnist/images/` and their source labels/index records.

Dataset: **PathMNIST**, part of **MedMNIST v2**, by Jiancheng Yang, Rui Shi, Donglai Wei and collaborators.

Distribution: [MedMNIST v2 / Zenodo record 10519652](https://zenodo.org/records/10519652).

Upstream data: **NCT-CRC-HE-100K / CRC-VAL-HE-7K**, by Jakob Nikolas Kather, Niels Halama and Alexander Marx. [Zenodo record 1214456](https://zenodo.org/records/1214456).

License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**. [License and terms](https://creativecommons.org/licenses/by/4.0/). These data are not relicensed under the repository's MIT code license.

Changes made by this project: selected a small reproducible subset; exported the existing 28×28 RGB arrays as lossless PNG; retained source labels, split names and indices; added a binary class-8-versus-rest target for the baseline. No claim of endorsement by the original authors is made.

Publications:

- Yang, J., Shi, R., Wei, D., et al. MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data 10, 41 (2023). https://doi.org/10.1038/s41597-022-01721-8
- Kather, J. N., et al. Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study. PLOS Medicine (2019). https://doi.org/10.1371/journal.pmed.1002730

These samples are public research data, not the repository owner's private hospital data. No clinical-use validation is supplied by this project.
