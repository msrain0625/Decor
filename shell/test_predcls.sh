#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py \
    --config-file 'configs/mp_vg.yaml' \
    --eval-only \
    MODEL.WEIGHT "checkpoints/predcls-0.5/model_final.pth" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    DTYPE float32 \
    TEST.IMS_PER_BATCH 1 \
    SOLVER.IMS_PER_BATCH 8 \
    DATALOADER.NUM_WORKERS 4 \
    INPUT.MIN_SIZE_TEST 600 \
    INPUT.MAX_SIZE_TEST 1000