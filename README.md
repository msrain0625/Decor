# DeCoR: Gradient-Relieved Multi-path Prototype Learning for Long-Tailed Scene Graph Generation

This repository provides the implementation of **DeCoR**, a gradient-relieved multi-path prototype learning framework for long-tailed Scene Graph Generation (SGG).

DeCoR improves prototype-based predicate prediction with two complementary modules:

* **Multi-path Prototype Branch (MPB)** expands single-path prototype matching into multiple prototype pathways.
* **Gradient Relief Head (GRH)** introduces a detached sparse correction route for selected high-pressure predicate groups.

> **Current release.**
> This release currently supports the main **Visual Genome PredCls** setting, which directly evaluates the proposed predicate prediction head. Scripts and configurations for SGCls, SGDet, and GQA are being organized.

---

## Installation

Our tested environment uses **Python 3.12.3**, **PyTorch 2.7.0 + CUDA 12.8**, and an **NVIDIA A6000** GPU.

Please install PyTorch according to your CUDA version. For CUDA 12.8, you may use:

```bash
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining dependencies and build the project:

```bash
pip install -r requirements.txt
python setup.py build develop
```

---

## Dataset Preparation

The current release uses **Visual Genome** under the PredCls protocol. Please prepare the dataset with the following structure:

```text
DeCoR/
  datasets/
    vg/
      VG-SGG-dicts-with-attri.json
      VG-SGG.h5
      image_data.json
      glove.6B.200d.pt
```

If your dataset is stored elsewhere, create a symbolic link:

```bash
ln -s /path/to/your/datasets datasets
```

Please check that the following files exist:

```bash
ls datasets/vg/VG-SGG-dicts-with-attri.json
ls datasets/vg/VG-SGG.h5
ls datasets/vg/image_data.json
```

---

## Pretrained Detector

Please place the pretrained detector checkpoint as:

```text
DeCoR/
  checkpoints/
    pretrained_faster_rcnn/
      model_final.pth
```

Alternatively, create a symbolic link:

```bash
mkdir -p checkpoints/pretrained_faster_rcnn
ln -s /path/to/pretrained_detector/model_final.pth \
  checkpoints/pretrained_faster_rcnn/model_final.pth
```

---

## Model Weights

We provide the checkpoint and log for the released VG PredCls setting. Due to random seeds and hardware differences, reproduced results may have minor variations.

| Model           | Setting    | mR@50 | mR@100 | F@50 | F@100 | Checkpoint                                                                                               | Log                                                                                                      |
| --------------- | ---------- | ----- | ------ | ---- | ----- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| PE-Net baseline | VG PredCls | 31.5  | 33.8   | 42.4 | 45.0  | -                                                                                                        | -                                                                                                        |
| DeCoR           | VG PredCls | 36.8  | 39.1   | 43.9 | 46.3  | [model_final.pth](https://drive.google.com/file/d/1gkNewhyzcVQxLWESmfy72FWjo9GmS5Yl/view?usp=drive_link) | [test_result.txt](https://drive.google.com/file/d/1PwD4vKRHozXCsVmpYayXB6NVLxPYnS3S/view?usp=drive_link) |

Please place the downloaded checkpoint as:

```text
checkpoints/decor_vg_predcls/model_final.pth
```

---

## Training

To train DeCoR on Visual Genome PredCls:

```bash
bash shell/train_predcls.sh
```

The script uses:

```text
configs/decor_vg_predcls.yaml
```

and saves checkpoints to:

```text
checkpoints/decor_vg_predcls/
```

---

## TEST

To evaluate the released checkpoint on Visual Genome PredCls:

```bash
bash shell/test_predcls.sh
```

The evaluation reports Recall@K, mean Recall@K, and F@K.

---

## Release Scope

This repository currently includes:

* Visual Genome PredCls training and evaluation scripts
* Multi-path Prototype Branch (MPB)
* Gradient Relief Head (GRH)
* Configuration for the released VG PredCls setting
* Checkpoint and evaluation log for DeCoR on VG PredCls

The remaining protocols, including SGCls, SGDet, and GQA, are being cleaned and will be released progressively.

---

## Acknowledgement

This codebase is developed based on the PE-Net / Scene Graph Benchmark framework. We thank the authors of the original repositories for their contributions to the SGG community.

---

## Citation

If you find this project useful, please cite our paper:

```bibtex
@misc{decor2026,
  title        = {DeCoR: Gradient-Relieved Multi-path Prototype Learning for Long-Tailed Scene Graph Generation},
  author       = {DeCoR Authors},
  year         = {2026},
  note         = {Manuscript under review}
}
