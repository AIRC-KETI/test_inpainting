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

### Model List
- [X] TripleLostGAN (Ours)
- [X] LostGAN
- [ ] CAL2IM
- [ ] CIAFILL
- [ ] Hyein et al.

## Test Samples

Run the test script:

```bash
python test_samples.py --dataset [DATASET] --real_path [DATASET PATH] --fake_path [FAKE PATH] --out_path [OUT PATH]
```
Note 1: Measuring IS or FID is recommended by generating more than 50,000 samples [WGAN-GP, TTUR]

Note 2: The data type is converted from float to uint8 when saving the image. For accurate measurement, use the [test_model.py](/test_model.py) whenever possible.

## Test Model with Categories

Run the test script:

```bash
python test_model_with_categories.py \
--dataset [DATASET] --out_path [OUT_DIR] --ckpt_path [CKPT_DIR] --model_name [MODEL_NAME]
```
Note: This test is for hallucinating visual instances with potal absensia. In coco and vg, the object corresponding to the label index defined in each task is deleted from the image, and the object is restored by inputting the index.

## Test Datasets for Various Tasks

#### Task 1: Hallucinating Visual Instances with Parital Absentia (HVIPA)

This task aims to restore an object when it has been partially or completely erased from the image.

The smaller the remaining ratio, the higher the percentage of objects erased.

* Example

Real | Rect mask | Rect masked image | Segmentation mask | Segmentation masked image
:---:|:---:|:---:|:---:|:---:
![](./readme_asset/000000_252219_real.jpg) | ![](./readme_asset/000000_252219_rect_mask.jpg) | ![](./readme_asset/000000_252219_rect_masked_image.jpg) | ![](./readme_asset/000000_252219_seg_mask.jpg) | ![](./readme_asset/000000_252219_seg_masked_image.jpg)

| Remain ratio     |   50  |   45  |   40  |   20  |   0 (HVITA)*  |
|:----------------:|:-----:|:-----:|:-----:|:-----:|:--------------:|
| COCO             | [COCO_50](https://drive.google.com/file/d/1jdsiFTPUJy6PPPmCJZTJ3u-y1LRY0ptp/view?usp=sharing) | [COCO_45](https://drive.google.com/file/d/1y7tzmoSyoGgDm6EwUtYfGUyzyILc-gwz/view?usp=sharing) | ![]() | ![]() | [COCO_00](https://drive.google.com/file/d/1vjlXbsG7k1jHuP_soqDi9HAXqhT5u4OF/view?usp=sharing) |
| Visual Genome    | [VG_50](https://drive.google.com/file/d/1rG7X9fGa9tptoBBgrSh7fxnLtRFdarH1/view?usp=sharing) | [VG_45](https://drive.google.com/file/d/1gdEn_Gf-3UiAq7dyjPghGpNhhRc2883_/view?usp=sharing) | [VG_40](https://drive.google.com/file/d/1Ca9nl7VfIo4KHl42yWqlNSUkJAowwB0Q/view?usp=sharing) | ![]() | [VG_00](https://drive.google.com/file/d/1om43Uwyynpx2wkPexyhjmUt4Uja_35Dt/view?usp=sharing) |

* HVITA*: Hallucinating Visual Instances with 'Total' Absensia

```bash
{data}_{resolution}_hvita
├── real
│   ├── 000
│   │   ├── xxxxxx.jpg
│   │   ├── xxxxxx.jpg        
│   │   └── ...           
│   ├── 001
│   │   ├── xxxxxx.jpg
│   │   ├── xxxxxx.jpg
│   │   └── ...       
│   └── ...(the last index number)
├── rect_mask
│   ├── 000
│   │   ├── xxxxxx.jpg
│   │   ├── xxxxxx.jpg        
│   │   └── ...           
│   ├── 001
│   │   ├── xxxxxx.jpg
│   │   ├── xxxxxx.jpg
│   │   └── ...       
│   └── ...(the last index number)
├── rect_masked_image
│   ├── 000
│   │   ├── xxxxxx.jpg
│   │   ├── xxxxxx.jpg        
│   │   └── ...           
│   ├── 001
│   │   ├── xxxxxx.jpg
│   │   ├── xxxxxx.jpg
│   │   └── ...       
│   └── ...(the last index number)
├── seg_mask
│   ├── 000
│   │   ├── xxxxxx.jpg
│   │   ├── xxxxxx.jpg        
│   │   └── ...           
│   ├── 001
│   │   ├── xxxxxx.jpg
│   │   ├── xxxxxx.jpg
│   │   └── ...       
│   └── ...(the last index number)
└── seg_masked_image
    ├── 000
    │   ├── xxxxxx.jpg
    │   ├── xxxxxx.jpg        
    │   └── ...           
    ├── 001
    │   ├── xxxxxx.jpg
    │   ├── xxxxxx.jpg
    │   └── ...       
    └── ...(the last index number)
```

#### Task 2: Hallucinating Multiple instances with partial Absentia 

This task aims to restore two or more objects when it has been partially or completely erased from the image.

The smaller the remaining ratio, the higher the percentage of objects erased.

| Remain ratio     |   50  |   45  |   40  |   20  |   0 (HVITA)*  |
|:----------------:|:-----:|:-----:|:-----:|:-----:|:--------------:|
| COCO             | [COCO_50](https://drive.google.com/file/d/1jdsiFTPUJy6PPPmCJZTJ3u-y1LRY0ptp/view?usp=sharing) | [COCO_45](https://drive.google.com/file/d/1y7tzmoSyoGgDm6EwUtYfGUyzyILc-gwz/view?usp=sharing) | ![]() | ![]() | [COCO_00](https://drive.google.com/file/d/1vjlXbsG7k1jHuP_soqDi9HAXqhT5u4OF/view?usp=sharing) |
| Visual Genome    | [VG_50](https://drive.google.com/file/d/1rG7X9fGa9tptoBBgrSh7fxnLtRFdarH1/view?usp=sharing) | [VG_45](https://drive.google.com/file/d/1gdEn_Gf-3UiAq7dyjPghGpNhhRc2883_/view?usp=sharing) | [VG_40](https://drive.google.com/file/d/1Ca9nl7VfIo4KHl42yWqlNSUkJAowwB0Q/view?usp=sharing) | ![]() | [VG_00](https://drive.google.com/file/d/1om43Uwyynpx2wkPexyhjmUt4Uja_35Dt/view?usp=sharing) |

#### Task 3: Inpainting

* Download link: [Places2](http://places2.csail.mit.edu/download.html)
* Download link:  [CelebA-HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ)
* Download link:  [IrregularMask](https://www.dropbox.com/s/01dfayns9s0kevy/test_mask.zip?dl=0)

#### Task 4: HVITA+Inpainting

* Download link: [COCO 128x128](https://drive.google.com/file/d/11xapK9GCIP-iZuvn8julofitQ_EPpNbh/view?usp=sharing)
* Download link: [VG 128x128](https://drive.google.com/file/d/1ONn3-sABfuFhjZj81X3VQqHvV-24ShuY/view?usp=sharing)

Note: A set of random tests mixed with a random rectangular mask or a mask in which the object is completely absent.

## Reference
If you find this repo helpful, please consider citing:

```

```

## Acknowledgements

This repository is based on [LostGAN](https://github.com/WillSuen/LostGANs) and [CAL2IM](https://github.com/wtliao/layout2img.
). The propsoed modules can be applied in the [layout2img](https://github.com/zhaobozb/layout2im).
