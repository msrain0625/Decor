# DeCoR: Gradient-Relieved Multi-path Prototype Learning for Long-Tailed Scene Graph Generation

This repository provides the implementation of **DeCoR**, a gradient-relieved multi-path prototype learning framework for long-tailed Scene Graph Generation (SGG).

DeCoR improves prototype-based predicate prediction with two modules:

* **Multi-path Prototype Branch (MPB)**: expands single-path prototype matching into multiple visual prototype pathways.
* **Gradient Relief Head (GRH)**: introduces a detached sparse correction route for selected high-pressure predicate groups.

> **Current release.**
> This repository currently provides the cleaned training and evaluation code for the main **Visual Genome PredCls** setting, which directly evaluates the proposed predicate prediction head. Scripts and configurations for SGCls, SGDet, and GQA are being organized and will be released progressively.

---

## News

* `[2026.xx.xx]` Initial release of DeCoR for VG PredCls.
* `[2026.xx.xx]` Model weights and logs will be available through Google Drive and OneDrive.

---

## Installation

Please follow the environment setup of the Scene Graph Benchmark / PE-Net codebase.

```bash
conda create -n decor python=3.8 -y
conda activate decor

pip install -r requirements.txt
python setup.py build develop
```

If your environment follows the original PE-Net or Scene-Graph-Benchmark setup, you can reuse the same dataset preprocessing and detector checkpoints.

---

## Dataset

We currently support **Visual Genome PredCls** in the cleaned release.

Please prepare the Visual Genome dataset following the common SGG preprocessing protocol. The expected directory structure is:

```text
datasets/
  vg/
    VG-SGG-dicts-with-attri.json
    VG-SGG.h5
    image_data.json
    imagedb_1024.h5
    glove.6B.200d.pt
```

The pretrained object detector checkpoint should be placed as:

```text
checkpoints/
  pretrained_faster_rcnn/
    model_final.pth
```

---

## Model Weights

We provide checkpoints and logs for the released VG PredCls setting. Due to random seeds and hardware differences, reproduced results may have minor variations from those reported in the paper.

| Model           | Setting    | mR@50 | mR@100 | F@50 | F@100 | Google Drive | OneDrive | Log  |
| --------------- | ---------- | ----: | -----: | ---: | ----: | ------------ | -------- | ---- |
| PE-Net baseline | VG PredCls |  31.5 |   33.8 | 42.4 |  45.0 | TODO         | TODO     | TODO |
| DeCoR           | VG PredCls |  36.8 |   39.2 | 43.9 |  46.3 | TODO         | TODO     | TODO |

Please replace `TODO` with the actual download links after uploading the model files.

Recommended checkpoint structure:

```text
checkpoints/
  decor_vg_predcls/
    model_final.pth
    config.yaml
    log.txt
```

---

## Training

We provide scripts for training DeCoR under the VG PredCls setting.

```bash
export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1

MODEL_NAME="DeCoR_VG_PredCls"

mkdir -p checkpoints/${MODEL_NAME}

python tools/relation_train_net.py \
  --config-file configs/decor_vg_predcls.yaml \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR DeCoRPredictor \
  MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR checkpoints/${MODEL_NAME}
```

The high-pressure predicate groups used by GRH are stored in:

```text
configs/groups/vg_predcls_high_pressure_groups.json
```

---

## Evaluation

To evaluate a trained DeCoR checkpoint on VG PredCls:

```bash
export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1

MODEL_NAME="DeCoR_VG_PredCls"

python tools/relation_test_net.py \
  --config-file configs/decor_vg_predcls.yaml \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR DeCoRPredictor \
  MODEL.WEIGHT checkpoints/${MODEL_NAME}/model_final.pth \
  MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR checkpoints/${MODEL_NAME} \
  TEST.ALLOW_LOAD_FROM_CACHE False
```

The evaluation reports Recall@K, mean Recall@K, and F@K.

---

## Current Release Scope

The current cleaned release focuses on:

* Visual Genome PredCls training
* Visual Genome PredCls evaluation
* Multi-path Prototype Branch (MPB)
* Gradient Relief Head (GRH)
* High-pressure predicate group configuration for VG PredCls
* DeCoR checkpoint and log for VG PredCls

The following components are being organized:

* SGCls scripts and configurations
* SGDet scripts and configurations
* GQA scripts and configurations
* Additional checkpoints and logs

---

## Project Structure

```text
DeCoR/
  configs/
    decor_vg_predcls.yaml
    groups/
      vg_predcls_high_pressure_groups.json

  tools/
    relation_train_net.py
    relation_test_net.py

  maskrcnn_benchmark/
    modeling/
      roi_heads/
        relation_head/
          roi_relation_predictors.py
          loss.py
          relation_head.py

  checkpoints/
    pretrained_faster_rcnn/
    decor_vg_predcls/

  README.md
  requirements.txt
  setup.py
```

---

## Notes

* DeCoR is designed to improve the predicate prediction head.
* PredCls is the main protocol for validating predicate recognition because it removes object detection and object classification errors.
* The released VG PredCls code is the cleaned and reproducible setting for the main predicate prediction experiment.
* SGCls, SGDet, and GQA use task-specific configurations and are being cleaned before release.

---

## Acknowledgement

This codebase is developed based on the Scene Graph Benchmark / PE-Net framework. We thank the authors of the original repositories for their contributions to the SGG community.

---

## Citation

If you find this project useful, please cite our paper:

```bibtex
@inproceedings{decor2026,
  title     = {DeCoR: Gradient-Relieved Multi-path Prototype Learning for Long-Tailed Scene Graph Generation},
  author    = {Anonymous Authors},
  booktitle = {PRCV},
  year      = {2026}
}
```

Please also consider citing PE-Net if you use the prototype-based SGG framework:

```bibtex
@inproceedings{zheng2023prototype,
  title     = {Prototype-based Embedding Network for Scene Graph Generation},
  author    = {Zheng, Chaofan and Lyu, Xinyu and Gao, Lianli and Dai, Bo and Song, Jingkuan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {22783--22792},
  year      = {2023}
}
```
