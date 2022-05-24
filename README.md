# Test model
This repository includes the implementation for test.

This repo is not completely.

## Requirements
please see INSTALL.md .

## TODO
- [v] instruction for coco validation with ours
- [ ] instruction for vg validation with ours
- [ ] instruction for coco validation with other methods
- [ ] instruction for vg validation with other methods

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
