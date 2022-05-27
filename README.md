# Test model
This repository includes the implementation for inpainting test.

In this test, the following metrics are evaluated.

*  L1
*  L2
*  SSIM
*  PSNR
*  LPIPS
*  IS
*  FID
- [ ] SSL 1-2 metric
- [ ] SSL 3 metric

This repo is not yet complete.

## Requirements
please see [INSTALL.md](INSTALL.md).

## TODO
- [X] COCO validation with ours
- [X] VG validation with ours
- [ ] validations with other methods
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
- [X] TSA2IM (Ours)
- [ ] LostGAN
- [ ] CAL2IM
- [ ] CIAFILL
- [ ] Hyein et al.

## Test Model

Note: Your own model should be in: 
```bash
${ROOT}
├── data
├── model
|   └── [your model.py]
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

### Input and Output type
Note 1: test model should get two or more input with 'dict' type content

| key                | type      | range                 | shape              | descrption                                    |
|--------------------|-----------|-----------------------|--------------------|-----------------------------------------------|
| image_contents     | float     | [-1,1]                | [Height,Width, 3]  | masked input (gray color)                     |
| mask               | float,int | [0, 1]                | [Height, Width, 1] | mask(1 to mask in each pixel)                 |
| label (optional)   | int       | [0, # of label index) | [# of objects]     | object index (defined in each dataset)        |
| bbox (optional)    | float     | [0, 1]                | [# of objects, 4]  | bound box with (x, y, w, h) relative position |
| triples (optional) | int       | [0,# of triples)      | [# of triples, 3]  | triples (index defined in each dataset)       |

Note 2: The result of the model should also be output in the form of a dict.

| key                | type      | range                 | shape              | descrption                                    |
|--------------------|-----------|-----------------------|--------------------|-----------------------------------------------|
| image_contents     | float     | [-1,1]                | [Height,Width, 3]  |inpaintied results                     |

```bash
python test_model.py --dataset [DATASET] --out_path [OUT_DIR] --ckpt_path [CKPT_DIR] --model_name [MODEL_NAME]
```

## Test Samples
Note 1: Measuring IS or FID is recommended by generating more than 50,000 samples [WGAN-GP, TTUR]

Note 2: The data type is converted from float to uint8 when saving the image. For accurate measurement, use the [test_model.py](/test_model.py) whenever possible.

```bash
python test_samples.py --dataset [DATASET] --out_path [OUT_DIR] --model_path [MODEL_DIR]
```

## Test Datasets for Various Tasks
Comming soon.
<li>Task 1: Hallucinating Visual Instances with Total Absensia (HVITA)</li>
<ul><li>COCO</li>
<li>VG</li></ul>
<li>Task 2: Inpainting</li>
<ul><li>Places2</li>
<li>CelebA</li>
</ul>
<li>Task 3: HVITA+Inpainting</li>
<ul><li> [COCO](/data) </li>
<li> [VG](/data) </li></ul>

## Reference
If you find this repo helpful, please consider citing:

```

```

## Acknowledgements

This repository is based on [LostGAN](https://github.com/WillSuen/LostGANs) and [CAL2IM](https://github.com/wtliao/layout2img.
). The propsoed modules can be applied in the [layout2img](https://github.com/zhaobozb/layout2im).
