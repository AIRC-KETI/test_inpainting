## Installation

#### Requiements
* matplotlib
* numpy
* scikit-image
* opencv-python
* tensorboardX
* scipy
* Cython
* pycocotools
* tqdm
* piq
* h5py 
* six
* pytorch
* torchvision
```
pip install -r requiements.txt

# setup for roi_layers
python setup.py build develop
```

#### Data Preparation
Download COCO dataset to `datasets/coco`
```
bash scripts/download_coco.sh
```
Download VG dataset to `datasets/vg`
```
bash scripts/download_vg.sh
python scripts/preprocess_vg.py
```
