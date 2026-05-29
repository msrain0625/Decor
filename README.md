# DeCoR: Gradient-Relieved Multi-path Prototype Learning for Long-Tailed Scene Graph Generation

This repository provides the official implementation of **DeCoR**, a gradient-relieved multi-path prototype learning framework for long-tailed Scene Graph Generation (SGG).

DeCoR improves prototype-based predicate prediction with two complementary modules:

* **Multi-path Prototype Branch (MPB)** expands single-path prototype matching into multiple visual prototype pathways.
* **Gradient Relief Head (GRH)** introduces a detached sparse correction route for selected high-pressure predicate groups.

> **Current release.**
> This repository currently provides the cleaned training and evaluation code for the main **Visual Genome PredCls** setting, which directly evaluates the proposed predicate prediction head. Scripts and configurations for SGCls, SGDet, and GQA are being organized and will be released progressively.

---

## News

* `[2026.xx.xx]` Initial release of DeCoR for Visual Genome PredCls.
* `[2026.xx.xx]` Checkpoints and logs will be available through Google Drive and OneDrive.

---

## Installation

# Install PyTorch according to your CUDA version first.
# Our tested environment uses PyTorch 2.7.0 + CUDA 12.8.
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
python setup.py build develop

---

## Dataset Preparation

The current cleaned release supports **Visual Genome PredCls**.

Please prepare Visual Genome following the common SGG preprocessing protocol. The expected dataset structure is:

```text
DeCoR/
  datasets/
    vg/
      VG-SGG-dicts-with-attri.json
      VG-SGG.h5
      image_data.json
      imagedb_1024.h5
      glove.6B.200d.pt
```

If your dataset is stored elsewhere, you can create a symbolic link:

```bash
ln -s /path/to/your/datasets datasets
```

For example:

```bash
ln -s /root/autodl-tmp/PENET/datasets datasets
```

Please make sure the following files exist:

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

You may also create a symbolic link:

```bash
mkdir -p checkpoints/pretrained_faster_rcnn
ln -s /path/to/pretrained_detector/model_final.pth \
  checkpoints/pretrained_faster_rcnn/model_final.pth
```

---

## Model Weights

We provide checkpoints and logs for the released VG PredCls setting. Due to random seeds and hardware differences, reproduced results may have minor variations from the reported numbers.

| Model           | Setting    | mR@50 | mR@100 | F@50 | F@100 | Google Drive | Log  |
| --------------- | ---------- | ----: | -----: | ---: | ----: | ------------ | ---- |
| PE-Net baseline | VG PredCls |  31.5 |   33.8 | 42.4 |  45.0 | -            | -    |
| DeCoR           | VG PredCls |  36.8 |   39.2 | 43.9 |  46.3 | [model_final.pth](https://drive.google.com/file/d/1gkNewhyzcVQxLWESmfy72FWjo9GmS5Yl/view?usp=drive_link)        | [test_result.txt](https://drive.google.com/file/d/1PwD4vKRHozXCsVmpYayXB6NVLxPYnS3S/view?usp=drive_link) |

Recommended checkpoint structure:

```text
DeCoR/
  checkpoints/
    decor_vg_predcls/
      model_final.pth
      config.yaml
      log.txt
```

---

## Training on VG PredCls

We provide the cleaned training script for Visual Genome PredCls.

```bash
bash shell/train_predcls.sh
```

The script is equivalent to:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py \
  --config-file configs/decor_vg_predcls.yaml \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
  DTYPE float32 \
  SOLVER.IMS_PER_BATCH 8 \
  TEST.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 60000 \
  SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS '(28000,48000)' \
  SOLVER.VAL_PERIOD 5000 \
  SOLVER.CHECKPOINT_PERIOD 5000 \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  OUTPUT_DIR checkpoints/decor_vg_predcls \
  INPUT.MIN_SIZE_TRAIN '(600,)' \
  INPUT.MAX_SIZE_TRAIN 1000 \
  INPUT.MIN_SIZE_TEST 600 \
  INPUT.MAX_SIZE_TEST 1000 \
  DATALOADER.NUM_WORKERS 4
```

The high-pressure predicate groups used by GRH are stored in:

```text
configs/groups/vg_predcls_high_pressure_groups.json
```

---

## Evaluation on VG PredCls

To evaluate a trained checkpoint:

```bash
bash shell/test_predcls.sh
```

A typical evaluation command is:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py \
  --config-file configs/decor_vg_predcls.yaml \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
  MODEL.WEIGHT checkpoints/decor_vg_predcls/model_final.pth \
  OUTPUT_DIR checkpoints/decor_vg_predcls \
  TEST.IMS_PER_BATCH 1
```

The evaluation reports Recall@K, mean Recall@K, and F@K.

---

## Current Release Scope

The current release includes:

* Visual Genome PredCls training code
* Visual Genome PredCls evaluation code
* Multi-path Prototype Branch (MPB)
* Gradient Relief Head (GRH)
* VG PredCls high-pressure group configuration
* Checkpoint and log placeholders for VG PredCls

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

  shell/
    train_predcls.sh
    test_predcls.sh

  tools/
    relation_train_net.py
    relation_test_net.py

  hetsgg/
    data/
    modeling/
      roi_heads/
        relation_head/
          relation_head.py
          roi_relation_predictors.py
          loss.py

  proto/

  checkpoints/
    pretrained_faster_rcnn/
    decor_vg_predcls/

  datasets/
    vg/

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

This codebase is developed based on the PE-Net / Scene Graph Benchmark framework. We thank the authors of the original repositories for their contributions to the SGG community.

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
