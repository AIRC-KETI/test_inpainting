import os
from tkinter.messagebox import NO
import PIL
from torch.utils.data import Dataset
from collections import defaultdict
import glob
import torch
import torchvision.transforms as T
import json

class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir, instances_json=None, stuff_json=None,
                 stuff_only=True, image_size=(299, 299), mask_size=16,
                 max_samples=None, normalize_images=True,
                 include_relationships=True, min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8, left_right_flip=False,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None):
        super(Dataset, self).__init__()

        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_samples = max_samples
        self.max_objects_per_image = max_objects_per_image
        self.normalize_images = normalize_images
        self.include_relationships = include_relationships
        self.left_right_flip = left_right_flip
        self.set_image_size(image_size)
        self.instances_json = instances_json

        instances_data = None
        if instances_json is not None and stuff_json != '':
            with open(instances_json, 'r') as f:
                instances_data = json.load(f)

            stuff_data = None
            if stuff_json is not None and stuff_json != '':
                with open(stuff_json, 'r') as f:
                    stuff_data = json.load(f)

            if instances_data:
                self.image_ids = []
                self.image_id_to_filename = {}
                self.image_id_to_size = {}
                for image_data in instances_data['images']:
                    image_id = image_data['id']
                    filename = image_data['file_name']
                    width = image_data['width']
                    height = image_data['height']
                    self.image_ids.append(image_id)
                    self.image_id_to_filename[image_id] = filename
                    self.image_id_to_size[image_id] = (width, height)

                self.vocab = {
                    'object_name_to_idx': {},
                    'pred_name_to_idx': {},
                }
                object_idx_to_name = {}
                all_instance_categories = []
                for category_data in instances_data['categories']:
                    category_id = category_data['id']
                    category_name = category_data['name']
                    all_instance_categories.append(category_name)
                    object_idx_to_name[category_id] = category_name
                    self.vocab['object_name_to_idx'][category_name] = category_id
            all_stuff_categories = []
            if stuff_data:
                for category_data in stuff_data['categories']:
                    category_name = category_data['name']
                    category_id = category_data['id']
                    all_stuff_categories.append(category_name)
                    object_idx_to_name[category_id] = category_name
                    self.vocab['object_name_to_idx'][category_name] = category_id

            if instance_whitelist is None:
                instance_whitelist = all_instance_categories
            if stuff_whitelist is None:
                stuff_whitelist = all_stuff_categories
            category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

            # Add object data from instances
            self.image_id_to_objects = defaultdict(list)
            for object_data in instances_data['annotations']:
                image_id = object_data['image_id']
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                # box_area = object_data['area'] / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                    self.image_id_to_objects[image_id].append(object_data)

            # Add object data from stuff
            if stuff_data:
                image_ids_with_stuff = set()
                for object_data in stuff_data['annotations']:
                    image_id = object_data['image_id']
                    image_ids_with_stuff.add(image_id)
                    _, _, w, h = object_data['bbox']
                    W, H = self.image_id_to_size[image_id]
                    box_area = (w * h) / (W * H)
                    # box_area = object_data['area'] / (W * H)
                    box_ok = box_area > min_object_size
                    object_name = object_idx_to_name[object_data['category_id']]
                    category_ok = object_name in category_whitelist
                    other_ok = object_name != 'other' or include_other
                    if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                        self.image_id_to_objects[image_id].append(object_data)

                if stuff_only:
                    new_image_ids = []
                    for image_id in self.image_ids:
                        if image_id in image_ids_with_stuff:
                            new_image_ids.append(image_id)
                    self.image_ids = new_image_ids

                    all_image_ids = set(self.image_id_to_filename.keys())
                    image_ids_to_remove = all_image_ids - image_ids_with_stuff
                    for image_id in image_ids_to_remove:
                        self.image_id_to_filename.pop(image_id, None)
                        self.image_id_to_size.pop(image_id, None)
                        self.image_id_to_objects.pop(image_id, None)

            # COCO category labels start at 1, so use 0 for __image__
            self.vocab['object_name_to_idx']['__image__'] = 0

            # Build object_idx_to_name
            name_to_idx = self.vocab['object_name_to_idx']
            assert len(name_to_idx) == len(set(name_to_idx.values()))
            max_object_idx = max(name_to_idx.values())
            idx_to_name = ['NONE'] * (1 + max_object_idx)
            for name, idx in self.vocab['object_name_to_idx'].items():
                idx_to_name[idx] = name
            self.vocab['object_idx_to_name'] = idx_to_name

            # Prune images that have too few or too many objects
            new_image_ids = []
            total_objs = 0
            for image_id in self.image_ids:
                num_objs = len(self.image_id_to_objects[image_id])
                total_objs += num_objs
                if min_objects_per_image <= num_objs <= max_objects_per_image:
                    new_image_ids.append(image_id)
            self.image_ids = new_image_ids

            self.vocab['pred_idx_to_name'] = [
                '__in_image__',
                'left of',
                'right of',
                'above',
                'below',
                'inside',
                'surrounding',
            ]
            self.vocab['pred_name_to_idx'] = {}
            for idx, name in enumerate(self.vocab['pred_idx_to_name']):
                self.vocab['pred_name_to_idx'][name] = idx

        else:
            self.image_id_to_filename = glob.glob(image_dir+'/*')
            self.image_ids = glob.glob(image_dir+'/*')

    def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        '''
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        '''
        self.transform = T.Compose(transform)
        self.image_size = image_size
    
    def __len__(self):
        if self.max_samples is None:
            if self.left_right_flip:
                return len(self.image_ids)*2
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)
    
    def __getitem__(self, index):    
        flip = False
        if index >= len(self.image_ids):
            index = index - len(self.image_ids)
            flip = True
        image_id = self.image_ids[index]
        if self.instances_json is not None:
            filename = self.image_id_to_filename[image_id]
            image_path = os.path.join(self.image_dir, filename)
        if self.instances_json is None:
            image_path = self.image_id_to_filename[index]
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                if flip:
                    image = PIL.ImageOps.mirror(image)
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))
        return {'images': image}

class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
              H, W = size
              self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)


IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


import math
import random
import h5py
import numpy as np

class ImageOnlyDatasetVG(Dataset):
    def __init__(self, vocab_json, h5_path, image_dir, image_size=(256, 256),
                 normalize_images=True, max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True,
                 left_right_flip=False):
        super(ImageOnlyDatasetVG, self).__init__()

        self.image_dir = image_dir
        self.image_size = image_size
        with open(vocab_json, 'r') as f:
            self.vocab = json.load(f)
        self.num_objects = len(self.vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.left_right_flip = left_right_flip
        self.normalize_images = normalize_images
        self.include_relationships = include_relationships

        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)

        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        if self.left_right_flip:
            return num * 2
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        flip = False
        if index >= self.data['object_names'].size(0):
            index = index - self.data['object_names'].size(0)
            flip = True
        # print('[*] {}, {}'.format(self.image_dir, self.image_paths[index].decode("utf-8")))
        img_path = os.path.join(self.image_dir, self.image_paths[index].decode("utf-8"))

        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                if flip:
                    image = PIL.ImageOps.mirror(image)
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size
        return {'images': image}


class COCOPairDataset(Dataset):
    def __init__(self, image_dir, fake_dir, instances_json=None, stuff_json=None,
                 stuff_only=True, image_size=(299, 299), mask_size=16,
                 max_samples=None, normalize_images=False,
                 include_relationships=True, min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8, left_right_flip=False,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None):
        super(Dataset, self).__init__()

        self.image_dir = image_dir
        self.fake_dir = fake_dir
        self.mask_size = mask_size
        self.max_samples = max_samples
        self.max_objects_per_image = max_objects_per_image
        self.normalize_images = normalize_images
        self.include_relationships = include_relationships
        self.left_right_flip = left_right_flip
        self.set_image_size(image_size)
        self.instances_json = instances_json

        instances_data = None
        if instances_json is not None and stuff_json != '':
            with open(instances_json, 'r') as f:
                instances_data = json.load(f)

            stuff_data = None
            if stuff_json is not None and stuff_json != '':
                with open(stuff_json, 'r') as f:
                    stuff_data = json.load(f)

            if instances_data:
                self.image_ids = []
                self.image_id_to_filename = {}
                self.image_id_to_size = {}
                for image_data in instances_data['images']:
                    image_id = image_data['id']
                    filename = image_data['file_name']
                    width = image_data['width']
                    height = image_data['height']
                    self.image_ids.append(image_id)
                    self.image_id_to_filename[image_id] = filename
                    self.image_id_to_size[image_id] = (width, height)

                self.vocab = {
                    'object_name_to_idx': {},
                    'pred_name_to_idx': {},
                }
                object_idx_to_name = {}
                all_instance_categories = []
                for category_data in instances_data['categories']:
                    category_id = category_data['id']
                    category_name = category_data['name']
                    all_instance_categories.append(category_name)
                    object_idx_to_name[category_id] = category_name
                    self.vocab['object_name_to_idx'][category_name] = category_id
            all_stuff_categories = []
            if stuff_data:
                for category_data in stuff_data['categories']:
                    category_name = category_data['name']
                    category_id = category_data['id']
                    all_stuff_categories.append(category_name)
                    object_idx_to_name[category_id] = category_name
                    self.vocab['object_name_to_idx'][category_name] = category_id

            if instance_whitelist is None:
                instance_whitelist = all_instance_categories
            if stuff_whitelist is None:
                stuff_whitelist = all_stuff_categories
            category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

            # Add object data from instances
            self.image_id_to_objects = defaultdict(list)
            for object_data in instances_data['annotations']:
                image_id = object_data['image_id']
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                # box_area = object_data['area'] / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                    self.image_id_to_objects[image_id].append(object_data)

            # Add object data from stuff
            if stuff_data:
                image_ids_with_stuff = set()
                for object_data in stuff_data['annotations']:
                    image_id = object_data['image_id']
                    image_ids_with_stuff.add(image_id)
                    _, _, w, h = object_data['bbox']
                    W, H = self.image_id_to_size[image_id]
                    box_area = (w * h) / (W * H)
                    # box_area = object_data['area'] / (W * H)
                    box_ok = box_area > min_object_size
                    object_name = object_idx_to_name[object_data['category_id']]
                    category_ok = object_name in category_whitelist
                    other_ok = object_name != 'other' or include_other
                    if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                        self.image_id_to_objects[image_id].append(object_data)

                if stuff_only:
                    new_image_ids = []
                    for image_id in self.image_ids:
                        if image_id in image_ids_with_stuff:
                            new_image_ids.append(image_id)
                    self.image_ids = new_image_ids

                    all_image_ids = set(self.image_id_to_filename.keys())
                    image_ids_to_remove = all_image_ids - image_ids_with_stuff
                    for image_id in image_ids_to_remove:
                        self.image_id_to_filename.pop(image_id, None)
                        self.image_id_to_size.pop(image_id, None)
                        self.image_id_to_objects.pop(image_id, None)

            # COCO category labels start at 1, so use 0 for __image__
            self.vocab['object_name_to_idx']['__image__'] = 0

            # Build object_idx_to_name
            name_to_idx = self.vocab['object_name_to_idx']
            assert len(name_to_idx) == len(set(name_to_idx.values()))
            max_object_idx = max(name_to_idx.values())
            idx_to_name = ['NONE'] * (1 + max_object_idx)
            for name, idx in self.vocab['object_name_to_idx'].items():
                idx_to_name[idx] = name
            self.vocab['object_idx_to_name'] = idx_to_name

            # Prune images that have too few or too many objects
            new_image_ids = []
            total_objs = 0
            for image_id in self.image_ids:
                num_objs = len(self.image_id_to_objects[image_id])
                total_objs += num_objs
                if min_objects_per_image <= num_objs <= max_objects_per_image:
                    new_image_ids.append(image_id)
            self.image_ids = new_image_ids

            self.vocab['pred_idx_to_name'] = [
                '__in_image__',
                'left of',
                'right of',
                'above',
                'below',
                'inside',
                'surrounding',
            ]
            self.vocab['pred_name_to_idx'] = {}
            for idx, name in enumerate(self.vocab['pred_idx_to_name']):
                self.vocab['pred_name_to_idx'][name] = idx
            self.fake_id_to_filename = glob.glob(fake_dir+'/*')
            self.fake_ids = glob.glob(fake_dir+'/*')
        else:
            self.image_id_to_filename = glob.glob(image_dir+'/*')
            self.image_ids = glob.glob(image_dir+'/*')
            self.fake_id_to_filename = glob.glob(fake_dir+'/*')
            self.fake_ids = glob.glob(fake_dir+'/*')

    def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        self.transform = T.Compose(transform)
        self.image_size = image_size
    
    def __len__(self):
        if self.max_samples is None:
            return max(len(self.image_ids), len(self.fake_ids))
        return [len(self.image_ids), self.max_samples, len(self.fake_ids)].sort()[0]
    
    def __getitem__(self, index):    
        f_index = index
        real_index = index
        while real_index >= len(self.image_ids):
            real_index = real_index - len(self.image_ids)

        image_id = self.image_ids[real_index]
        if self.instances_json is not None:
            filename = self.image_id_to_filename[image_id]
            image_path = os.path.join(self.image_dir, filename)
            fake_path = self.fake_id_to_filename[f_index]
        if self.instances_json is None:
            image_path = self.image_id_to_filename[real_index]
            fake_path = self.fake_id_to_filename[f_index]
        with open(image_path, 'rb') as r:
            with PIL.Image.open(r) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))
        with open(fake_path, 'rb') as f:
            with PIL.Image.open(f) as fake:
                fake = self.transform(fake.convert('RGB'))
        return {'images': image, 'fakes': fake}


class VGPairDataset(Dataset):
    def __init__(self, vocab_json, h5_path, image_dir, fake_dir, image_size=(256, 256),
                 normalize_images=False, max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True):
        super(VGPairDataset, self).__init__()

        self.image_dir = image_dir
        self.fake_dir = fake_dir
        self.image_size = image_size
        with open(vocab_json, 'r') as f:
            self.vocab = json.load(f)
        self.num_objects = len(self.vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.normalize_images = normalize_images
        self.include_relationships = include_relationships

        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)

        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))
        self.fake_list = glob.glob(self.fake_dir+'/*')

    def __len__(self):
        num = self.data['object_names'].size(0)
        fake_num = len(self.fake_list)
        return max(num, fake_num)

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        real_index = index
        while real_index >= self.data['object_names'].size(0):
            real_index = real_index - self.data['object_names'].size(0)

        img_path = os.path.join(self.image_dir, self.image_paths[real_index].decode("utf-8"))
        fake_path = self.fake_list[index]
        with open(img_path, 'rb') as r:
            with PIL.Image.open(r) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))
        
        with open(fake_path, 'rb') as f:
            with PIL.Image.open(f) as fake_image:
                WW, HH = fake_image.size
                fake_image = self.transform(fake_image.convert('RGB'))
        H, W = self.image_size
        return {'images': image, 'fakes': fake_image}