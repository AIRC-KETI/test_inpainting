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

```bash
python test_model.py --dataset [DATASET] --out_path [OUT_DIR] --model_path [MODEL_DIR]
```

## Test Samples
Note 1: Measuring IS or FID is recommended by generating more than 50,000 samples [WGAN-GP, TTUR]

Note 2: The data type is converted from float to uint8 when saving the image. For accurate measurement, use the test model whenever possible.

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
