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
    if args.ratio < 1.e-6 and args.obj_count == 1 and args.group != 'triple':
        args.out_path = os.path.join(args.out_path, args.dataset)
    elif args.group == 'triple':
        args.out_path = os.path.join(args.out_path, args.dataset + '_triple_' + str(int(args.obj_count)))
    elif args.obj_count == 1 and args.group != 'triple':
        args.out_path = os.path.join(args.out_path, args.dataset+'_'+str(int(100*args.ratio))+'_'+args.direction)
    elif args.obj_count > 1:
        args.out_path = os.path.join(args.out_path, args.dataset+'_multiple_'+str(int(args.obj_count)))
    else:
        args.out_path = os.path.join(args.out_path, args.dataset)
    num_gpus = torch.cuda.device_count()
    num_workers = 2
    if num_gpus > 1:
        parallel = True
        num_workers = num_workers * num_gpus
    else:
        parallel = False

    # data loader
    device = torch.device('cuda')
    print(args.out_path)
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'real/')):
        os.makedirs(os.path.join(args.out_path, 'real/'))
    if not os.path.exists(os.path.join(args.out_path, 'seg_masked_image/')):
        os.makedirs(os.path.join(args.out_path, 'seg_masked_image/'))
    if not os.path.exists(os.path.join(args.out_path, 'rect_masked_image/')):
        os.makedirs(os.path.join(args.out_path, 'rect_masked_image/'))
    if not os.path.exists(os.path.join(args.out_path, 'seg_mask/')):
        os.makedirs(os.path.join(args.out_path, 'seg_mask/'))
    if not os.path.exists(os.path.join(args.out_path, 'rect_mask/')):
        os.makedirs(os.path.join(args.out_path, 'rect_mask/'))
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
    if args.obj_count == 1 and args.group != 'triple':
        for i in range(obj.size(1)):  # box, image, mask, idx
            save_mask(idx, data, args, i)
    else:
        save_multiple_mask(idx, data, args)


def save_mask(idx, data, args, i):
    image = data['images']
    obj = data['objs']
    box = data['boxes']
    if args.dataset == 'coco':
        mask = data['masks']
        mmask = torch.unsqueeze(mask[:, i, :, :], 1).type(torch.FloatTensor)
    else:
        mask = data['images']
        mmask = mask
    bbox = torch.unsqueeze(box[:, i, :], 1)

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


def save_multiple_mask(idx, data, args):
    image = data['images']
    obj = data['objs']
    box = data['boxes']
    triple = data['triples']
    if args.dataset == 'coco':
        mask = data['masks']
    else:
        mask = data['images']

    if args.group == 'random':
        obj_list = list(range(0, obj.size(1)))
        obj_list = [x for x in obj_list if (obj[:,x].item() != 0)]
        obj_count = args.obj_count
        if args.obj_count > len(obj_list):
            obj_count = len(obj_list)
        selected_list = random.sample(obj_list, obj_count)
    else:
        s, p, o = triple.chunk(3, dim=-1)  # [B,# of triples, 1]
        s, p, o = [x.squeeze(-1) for x in [s, p, o]]  # [B, # of triples]
        s_obj = torch.gather(obj, -1, s)
        o_obj = torch.gather(obj, -1, o)
        sel_triple = []
        for i in range(s_obj.size(1)):
            if s_obj[:,i] != 0 and o_obj[:,i] != 0:
                sel_triple.append([s[:,i], o[:,i]])

        obj_count = args.obj_count
        if args.obj_count > len(sel_triple):
            obj_count = len(sel_triple)
        selected_list = random.sample(sel_triple, obj_count)
        selected_list = [x for xs in selected_list for x in xs]
    rect_mask = torch.zeros(image.size(0), 1, image.size(-2), image.size(-1))

    for i in selected_list:
        if args.group == 'random':
            bbox = torch.unsqueeze(box[:, i, :], 1)
        else:
            bbox = box[:, i, :]
        mmask = mask
        rect_mask = rect_mask + make_mask(bbox, image, mmask, args, is_rect=True)

    rect_mask = torch.minimum(rect_mask, torch.ones_like(rect_mask))
    rect_hvita = image * (1. - rect_mask) + 0.5 * rect_mask

    if args.dataset == 'coco':
        seg_mask = torch.zeros(image.size(0), 1, image.size(-2), image.size(-1))
        for i in selected_list:
            if args.group == 'random':
                bbox = torch.unsqueeze(box[:, i, :], 1)
            else:
                bbox = box[:, i, :]
            if args.group == 'random':
                mmask = torch.unsqueeze(mask[:, i, :, :], 1).type(torch.FloatTensor)
            else:
                mmask = mask[:, i, :, :].type(torch.FloatTensor)

            seg_mask = seg_mask + make_mask(bbox, image, mmask, args, is_rect=False)

        seg_mask = torch.minimum(seg_mask, torch.ones_like(seg_mask))
        seg_hvita = image * (1. - seg_mask) + 0.5 * seg_mask

    name = "{:06d}".format(data['image_id'].item()) if args.dataset == 'coco' else str(data['image_id']).replace(
        "\\", "_").replace(".jpg", "")
    torchvision.utils.save_image(image,
                                 "{}/real/{}_{:06d}_{}.jpg".format(
                                     args.out_path, str([obj[0,x].item() for x in selected_list]), idx, name))
    torchvision.utils.save_image(rect_hvita,
                                 "{}/rect_masked_image/{}_{:06d}_{}.jpg".format(
                                     args.out_path, str([obj[0, x].item() for x in selected_list]), idx, name))
    torchvision.utils.save_image(rect_mask,
                                 "{}/rect_mask/{}_{:06d}_{}.jpg".format(
                                     args.out_path, str([obj[0, x].item() for x in selected_list]), idx, name))
    if args.dataset == 'coco':
        torchvision.utils.save_image(seg_hvita,
                                     "{}/seg_masked_image/{}_{:06d}_{}.jpg".format(
                                         args.out_path, str([obj[0, x].item() for x in selected_list]), idx, name))
        torchvision.utils.save_image(seg_mask,
                                     "{}/seg_mask/{}_{:06d}_{}.jpg".format(
                                         args.out_path, str([obj[0, x].item() for x in selected_list]), idx, name))


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
    parser.add_argument('--obj_count', type=int, default=1,
                        help='# of erased objects')
    parser.add_argument('--group', type=str, default='random',
                        choices=['random', 'triple'],
                        help='# of erased objects')
    args = parser.parse_args()
    main(args)

# python generate_datasets.py
