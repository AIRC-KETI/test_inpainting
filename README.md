# Test model
This repository includes the implementation for inpainting test.

This repo is not yet complete.

## Availabe Metrics
In this test, the following metrics are evaluated.

- [X] L1 (SSL 1-1 metric)
- [X] L2
- [X] Structural SIMilarity (SSIM, SSL 2-3 metric)
- [X] Peak Signal-to-Noise Ratio (PSNR)
- [X] Learned Perceptual Image Patch Similarity (LPIPS, SSL 2-1 metric)
- [X] Inception Score (IS)
- [X] Frechet Inception Distance (FID, SSL 2-2 metric) 
- [ ] The size of the smallest part that can be reconstructed (SSL 1-2 metric, COCO and VG only)
- [ ] Restoration success rate of noise / part deletion video (SSL 3 metric, COCO and VG only)

## Requirements
please see [INSTALL.md](INSTALL.md).

## TODO
- [X] COCO validation with ours
- [X] VG validation with ours
- [X] validations with other methods
- [ ] Valdiations with other metrics
- [ ] Link for download test datasets

## Data Preparation
Download COCO dataset to datasets/coco
```bash
bash scripts/download_coco.sh
```
Download VG dataset to datasets/vg
```bash
bash scripts/download_vg.sh
python scripts/preprocess_vg.py
```
## Model List
- [X] STALostGAN (Ours)
- [X] LostGAN
- [ ] CAL2IM
- [ ] CIAFILL
- [ ] Hyein et al.

## Test Model

Run the test script:

```bash
python test_model.py \
--dataset [DATASET] --out_path [OUT_DIR] --ckpt_path [CKPT_DIR] --model_name [MODEL_NAME]
```
#### Note for test your own model

1. Your own model should be in: 
```bash
${ROOT}
├── data
├── model
    └── [your model.py]
├── model_layout2img
├── scripts
├── utils
├── utils_layout2img
├── INSTALL.md
├── README.md
├── requirements.txt
├── test_model.py
└── test_samples.py
```

2. test model should get two or more input with 'dict' type content

| key                | type      | range                 | shape              | descrption                                    |
|--------------------|-----------|-----------------------|--------------------|-----------------------------------------------|
| image_contents     | float     | [-1, 1]                | [Height,Width, 3]  | masked input (gray color)                     |
| mask               | float,int | [0, 1]                | [Height, Width, 1] | mask(1 to mask in each pixel)                 |
| label (optional)   | int       | [0, # of label index) | [# of objects]     | object index (defined in each dataset)        |
| bbox (optional)    | float     | [0, 1]                | [# of objects, 4]  | bound box with (x, y, w, h) relative position |
| triples (optional) | int       | [0,# of triples)      | [# of triples, 3]  | triples (index defined in each dataset)       |

3. The result of the model should also be output in the form of a dict.

| key                | type      | range                 | shape              | descrption                                    |
|--------------------|-----------|-----------------------|--------------------|-----------------------------------------------|
| image_contents     | float     | [-1,1]                | [Height,Width, 3]  |inpaintied results                     |

## Test Samples

Run the test script:

```bash
python test_samples.py --dataset [DATASET] --real_path [DATASET PATH] --fake_path [FAKE PATH] --out_path [OUT PATH]
```
Note 1: Measuring IS or FID is recommended by generating more than 50,000 samples [WGAN-GP, TTUR]

Note 2: The data type is converted from float to uint8 when saving the image. For accurate measurement, use the [test_model.py](/test_model.py) whenever possible.


## Test Datasets for Various Tasks

#### Task 1: Hallucinating Visual Instances with Total Absensia

* Download link: [COCO] [VG]

#### Task 2: Inpainting

* Download link: [Places2](http://places2.csail.mit.edu/download.html)
* Download link:  [CelebA-HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ)
* Download link:  [IrregularMask](https://www.dropbox.com/s/01dfayns9s0kevy/test_mask.zip?dl=0)

#### Task 3: HVITA+Inpainting

* Download link: [COCO 128x128](https://drive.google.com/file/d/11xapK9GCIP-iZuvn8julofitQ_EPpNbh/view?usp=sharing)
* Download link: [VG 128x128](https://drive.google.com/file/d/1ONn3-sABfuFhjZj81X3VQqHvV-24ShuY/view?usp=sharing)


## Reference
If you find this repo helpful, please consider citing:

```

```

## Acknowledgements

This repository is based on [LostGAN](https://github.com/WillSuen/LostGANs) and [CAL2IM](https://github.com/wtliao/layout2img.
). The propsoed modules can be applied in the [layout2img](https://github.com/zhaobozb/layout2im).
