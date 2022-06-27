import argparse
import os
import pickle
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from pycocotools.coco import COCO
from collections import OrderedDict

from utils.util import *
from utils.save_image import *
from data.vg_size_free import *
from data.cocostuff_loader_size_free import *
from model.resnet_generator_app_v2 import *
from model.rcnn_discriminator_app import *
import model.vis as vis
from imageio import imwrite
from model.sync_batchnorm import DataParallelWithCallback
from utils.logger import setup_logger
from tqdm import tqdm
import faulthandler
import utils.bilinear as bil
import json
import glob
import re
import piq


def main(args):
    # parameters
    z_dim = 128
    pred_classes = 7 if args.dataset == 'coco' else 7
    num_classes = 184 if args.dataset == 'coco' else 179
    num_obj = 8 if args.dataset == 'coco' else 8

    if args.ratio < 1.e-6:
        args.out_path = os.path.join(args.out_path, args.dataset)
    else:
        args.out_path = os.path.join(args.out_path, args.dataset+'_'+str(int(100*args.ratio))+'_'+args.direction)
    num_gpus = torch.cuda.device_count()
    num_workers = 2
    if num_gpus > 1:
        parallel = True
        num_workers = num_workers * num_gpus
    else:
        parallel = False

    # data loader
    device = torch.device('cuda')
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'real/')):
        os.makedirs(os.path.join(args.out_path, 'real/'))
    if not os.path.exists(os.path.join(args.out_path, 'real/')):
        os.makedirs(os.path.join(args.out_path, 'real/'))
    if not os.path.exists(os.path.join(args.out_path, 'real/')):
        os.makedirs(os.path.join(args.out_path, 'real/'))
    if not os.path.exists(os.path.join(args.out_path, 'samples/')):
        os.makedirs(os.path.join(args.out_path, 'samples/'))
    if not os.path.exists(os.path.join(args.out_path, 'categories/')):
        os.makedirs(os.path.join(args.out_path, 'categories/'))
    if not os.path.exists(os.path.join(args.out_path, 'categories/real')):
        os.makedirs(os.path.join(args.out_path, 'categories/real'))
    if not os.path.exists(os.path.join(args.out_path, 'categories/seg_masked_image')):
        os.makedirs(os.path.join(args.out_path, 'categories/seg_masked_image'))
    if not os.path.exists(os.path.join(args.out_path, 'categories/rect_masked_image')):
        os.makedirs(os.path.join(args.out_path, 'categories/rect_masked_image'))
    if not os.path.exists(os.path.join(args.out_path, 'categories/seg_mask')):
        os.makedirs(os.path.join(args.out_path, 'categories/seg_mask'))
    if not os.path.exists(os.path.join(args.out_path, 'categories/rect_mask')):
        os.makedirs(os.path.join(args.out_path, 'categories/rect_mask'))

    for i in range(200):
        if not os.path.exists(os.path.join(args.out_path, 'categories/real/{:03d}'.format(i))):
            os.makedirs(os.path.join(args.out_path, 'categories/real/{:03d}'.format(i)))
        if not os.path.exists(os.path.join(args.out_path, 'categories/seg_masked_image/{:03d}'.format(i))):
            os.makedirs(os.path.join(args.out_path, 'categories/seg_masked_image/{:03d}'.format(i)))
        if not os.path.exists(os.path.join(args.out_path, 'categories/rect_masked_image/{:03d}'.format(i))):
            os.makedirs(os.path.join(args.out_path, 'categories/rect_masked_image/{:03d}'.format(i)))
        if not os.path.exists(os.path.join(args.out_path, 'categories/seg_mask/{:03d}'.format(i))):
            os.makedirs(os.path.join(args.out_path, 'categories/seg_mask/{:03d}'.format(i)))
        if not os.path.exists(os.path.join(args.out_path, 'categories/rect_mask/{:03d}'.format(i))):
            os.makedirs(os.path.join(args.out_path, 'categories/rect_mask/{:03d}'.format(i)))
    
    if args.dataset == "coco":
        pair_dataset = CocoSceneGraphDataset('datasets/coco/val2017/',
                                    instances_json='datasets/coco/annotations/instances_val2017.json',
                                    stuff_json='datasets/coco/annotations/stuff_val2017.json', left_right_flip=False)
    elif args.dataset == 'vg':
        pair_dataset = VgSceneGraphDataset(vocab_json='data/tmp/vocab.json', h5_path='data/tmp/preprocess_vg/test.h5',
                                   image_dir='datasets/vg/images/')

    dataloader = torch.utils.data.DataLoader(
        pair_dataset, batch_size=1,
        drop_last=False, shuffle=False, num_workers=num_workers)
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            process_mask(idx, data, args)


def process_mask(idx, data, args):
    obj = data['objs']
    for i in range(obj.size(1)):  # box, image, mask, idx
        save_mask(idx, data, args, i)


def save_mask(idx, data, args, i):
    image = data['images']
    obj = data['objs']
    box = data['boxes']
    if args.dataset == 'coco':
        mask = data['masks']
    else:
        mask = data['images']
    bbox = torch.unsqueeze(box[:, i, :], 1)
    mmask = torch.unsqueeze(mask[:, i, :, :], 1).type(torch.FloatTensor)
    rect_mask = make_mask(bbox, image, mmask, args, is_rect=True)
    rect_hvita = image * (1. - rect_mask) + 0.5 * rect_mask

    if args.dataset == 'coco':
        seg_mask = make_mask(bbox, image, mmask, args, is_rect=False)
        seg_hvita = image * (1. - seg_mask) + 0.5 * seg_mask

    name = "{:06d}".format(data['image_id'].item()) if args.dataset == 'coco' else str(data['image_id']).replace(
        "\\", "_").replace(".jpg", "")
    torchvision.utils.save_image(image,
                                 "{}/categories/real/{:03d}/{:06d}_{}.jpg".format(
                                     args.out_path, obj[0, i].item(), idx, name))
    torchvision.utils.save_image(rect_hvita,
                                 "{}/categories/rect_masked_image/{:03d}/{:06d}_{}.jpg".format(
                                     args.out_path, obj[0, i].item(), idx, name))
    if args.dataset == 'coco':
        torchvision.utils.save_image(seg_hvita,
                                     "{}/categories/seg_masked_image/{:03d}/{:06d}_{}.jpg".format(
                                         args.out_path, obj[0, i].item(), idx, name))
        torchvision.utils.save_image(seg_mask,
                                     "{}/categories/seg_mask/{:03d}/{:06d}_{}.jpg".format(
                                         args.out_path, obj[0, i].item(), idx, name))
    torchvision.utils.save_image(rect_mask,
                                 "{}/categories/rect_mask/{:03d}/{:06d}_{}.jpg".format(
                                     args.out_path, obj[0, i].item(), idx, name))


def make_mask(box, image, mask, args, is_rect=True):
    if is_rect:
        return make_rect_mask(box, image, args, False)
    else:
        return make_seg_mask(mask, box, args)


def make_rect_mask(bbox, image, args, is_train=False):
    ratio = args.ratio
    if args.direction == 'left':
        re_box = torch.cat((bbox[:,:,0], bbox[:,:,1], (1.-ratio) * bbox[:,:,2], bbox[:,:,3]), -1)
    elif args.direction == 'right':
        re_box = torch.cat((ratio * bbox[:,:,2] + bbox[:,:,0], bbox[:,:,1], bbox[:,:,2], bbox[:,:,3]), -1)
    elif args.direction == 'upper':
        re_box = torch.cat((bbox[:,:,0], bbox[:,:,1], bbox[:,:,2], (1.-ratio) * bbox[:,:,3]), -1)
    else:
        re_box = torch.cat((bbox[:,:,0], ratio * bbox[:,:,3] + bbox[:,:,1], bbox[:,:,2], bbox[:,:,3]), -1)

    return bil.bbox2_mask(torch.unsqueeze(re_box, 0), image, is_train)


def make_seg_mask(mask, box, args):
    if args.ratio <= 1.e-6:
        return mask
    else:
        return make_seg_mask_ratio(mask, box, args)


def make_seg_mask_ratio(mask, bbox, args):
    count = torch.sum(mask, axis=(1, 2, 3))
    start_x, start_y = bbox[:,:,0] + 0.5 * bbox[:,:,2], bbox[:,:,1] + 0.5 * bbox[:,:,3]
    start = (start_x, start_y)
    for i in range(16):
        bool_mask = make_bool_mask(start, mask, bbox, args)
        temp_mask = mask * bool_mask
        bool_count = torch.sum(temp_mask, axis=(1, 2, 3))
        # print(bool_count, count, args.ratio, torch.sum(bool_mask, axis=(1, 2, 3)))
        if abs((bool_count/count) - (1.-args.ratio)) <= 0.02:
            break
        elif (bool_count/count) > (1.-args.ratio):  # too many
            if args.direction == 'left':
                start = ((start_x-(1/pow(1.25, i))*0.5*bbox[:,:,1]), start_y)
            elif args.direction == 'right':
                start = ((start_x+(1/pow(1.25, i))*0.5*bbox[:,:,1]), start_y)
            elif args.direction == 'upper':
                start = (start_x, start_y - (1/pow(1.25, i))*0.5*bbox[:,:,3])
            else:
                start = (start_x, start_y + (1/pow(1.25, i))*0.5*bbox[:,:,3])
        else:  # too small
            if args.direction == 'left':
                start = ((start_x + (1 / pow(1.25, i))*0.5*bbox[:, :, 1]), start_y)
            elif args.direction == 'right':
                start = ((start_x - (1 / pow(1.25, i))*0.5*bbox[:, :, 1]), start_y)
            elif args.direction == 'upper':
                start = (start_x, start_y + (1 / pow(1.25, i))*0.5*bbox[:, :, 3])
            else:
                start = (start_x, start_y - (1 / pow(1.25, i))*0.5*bbox[:, :, 3])

    return temp_mask


def make_bool_mask(start, mask, bbox, args):
    start_x, start_y = start[0], start[-1]

    if args.direction == 'left':
        temp_bbox = torch.cat((torch.zeros_like(bbox[:, :, 0]),
                               torch.zeros_like(bbox[:, :, 1]),
                               start_x,
                               torch.ones_like(bbox[:, :, 1]),), -1)
    elif args.direction == 'right':
        temp_bbox = torch.cat((start_x,
                               torch.zeros_like(bbox[:, :, 1]),
                               torch.ones_like(bbox[:, :, 1]),
                               torch.ones_like(bbox[:, :, 1]),), -1)
    elif args.direction == 'upper':
        temp_bbox = torch.cat((torch.zeros_like(bbox[:, :, 0]),
                               torch.zeros_like(bbox[:, :, 1]),
                               torch.ones_like(bbox[:, :, 1]),
                               start_y,), -1)
    else:
        temp_bbox = torch.cat((torch.zeros_like(bbox[:, :, 0]),
                               start_y,
                               torch.ones_like(bbox[:, :, 1]),
                               torch.ones_like(bbox[:, :, 1]),), -1)

    return bil.bbox2_mask(torch.unsqueeze(temp_bbox, 0), mask, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--out_path', type=str, default='./size_free',
                        help='path to output files')
    parser.add_argument('--ratio', type=float, default=0.,
                        help='remained image ratio of each mask')
    parser.add_argument('--direction', type=str, default='left',
                        choices=['left', 'right', 'upper', 'below'],
                        help='where the parts were deleted from the image')

    args = parser.parse_args()
    main(args)

# python generate_datasets.py
