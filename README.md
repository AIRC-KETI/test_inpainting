# Test model
This repository includes the implementation for test.

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
please see INSTALL.md .

## TODO
- [X] COCO validation with ours
- [X] VG validation with ours
- [ ] validations with other methods
- [ ] Valdiations with other metrics

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
## Test

```bash
python test_model.py --dataset [DATASET] --out_path [OUT_DIR] --model_path [MODEL_DIR]
```

## Reference

If you find this repo helpful, please consider citing:

```
@inproceedings{he2021context,
  title={Context-Aware Layout to Image Generation with Enhanced Object Appearance},
  author={He, Sen and Liao, Wentong and Yang, Michael and Yang, Yongxin and Song, Yi-Zhe and Rosenhahn, Bodo and Xiang, Tao},
  booktitle={CVPR},
  year={2021}
}
```

## Acknowledgements

This repository is based on [LostGAN](https://github.com/WillSuen/LostGANs) and [CAL2IM](https://github.com/wtliao/layout2img.
). The propsoed modules can be applied in the [layout2img](https://github.com/zhaobozb/layout2im).
