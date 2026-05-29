# Dataset Preparation

Decor is evaluated on three datasets: **Visual Genome (VG-150)**, **GQA-200**.

## Visual Genome (VG-150)

1. Download the VG images [part1 (9 GB)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 GB)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract to `datasets/vg/VG_100K/`.

2. Download the [scene graph annotations](https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed) and place under `datasets/vg/`:
   - `VG-SGG-with-attri.h5`
   - `VG-SGG-dicts-with-attri.json`
   - `image_data.json`

3. Download the pretrained Faster R-CNN detector from [OneDrive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/lirj2_shanghaitech_edu_cn/EhKLGFbrRfhEka0gBbfcNeIBOYMUNRftCnp6JhPueMIOJQ?e=x1owcV) → `detector_model/vg/pretrained_faster_rcnn/model_final.pth`

4. Download GloVe vectors `glove.6B.300d.pt` → `datasets/vg/`

```
datasets/vg/
├── VG_100K/
├── VG-SGG-with-attri.h5
├── VG-SGG-dicts-with-attri.json
├── image_data.json
└── glove.6B.300d.pt
```

## GQA-200

Following [SHA-GCL](https://github.com/dongxingning/SHA-GCL-for-SGG), GQA-200 contains 200 object categories and 100 predicate categories.

1. Download GQA images from the [GQA official page](https://cs.stanford.edu/people/dorarad/gqa/download.html) → `datasets/gqa/images/`

2. Download GQA-200 annotations from [SHA-GCL](https://github.com/dongxingning/SHA-GCL-for-SGG) → `datasets/gqa/`

3. Download the GQA detector from [Google Drive](https://drive.google.com/file/d/1vimG6TZPs0pJwL2MpLM2f2pLGUuiaDNn) → `detector_model/gqa/gqa_det.pth`

```
datasets/gqa/
├── images/
├── GQA_200_Train.json
├── GQA_200_Test.json
└── GQA_200_ID_Info.json
```
