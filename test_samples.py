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
from data.cocostuff_loader_my import *
from data.vg_direction import *
from model.resnet_generator_app_v2 import *
from model.rcnn_discriminator_app import *
from data.image_only_loader import *
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
    img_size = args.img_size
    z_dim = 128
    pred_classes = 7 if args.dataset == 'coco' else 7
    num_classes = 184 if args.dataset == 'coco' else 179
    num_obj = 8 if args.dataset == 'coco' else 8

    args.out_path = os.path.join(args.out_path, args.dataset, str(args.img_size))

    num_gpus = torch.cuda.device_count()
    num_workers = 2
    if num_gpus > 1:
        parallel = True
        args.batch_size = args.batch_size * num_gpus
        num_workers = num_workers * num_gpus
    else:
        parallel = False

    # data loader
    my_path = 'D:/layout2img_ours'
    device = torch.device('cuda')
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.makedirs(os.path.join(args.out_path, 'model/'))
    if not os.path.exists(os.path.join(args.out_path, 'samples/')):
        os.makedirs(os.path.join(args.out_path, 'samples/'))
    if not os.path.exists(os.path.join(args.out_path, 'grid_samples/')):
        os.makedirs(os.path.join(args.out_path, 'grid_samples/'))
    
    fake_dir = '{}/samples/'.format(args.out_path)
    if args.dataset == "coco":
        pair_dataset = COCOPairDataset(my_path + './datasets/coco/val2017/',
                                    fake_dir=my_path + './test_tsa_v3/coco/128/samples',
                                    instances_json=my_path+'./datasets/coco/annotations/instances_val2017.json',
                                    stuff_json=my_path+'./datasets/coco/annotations/stuff_val2017.json', image_size=(128, 128), left_right_flip=True)
    elif args.dataset == 'vg':
        pair_dataset = ImageOnlyDatasetVG(vocab_json=my_path+'./datasets/vg/vocab.json', h5_path=my_path+'./datasets/vg/test.h5',
                                   image_dir=my_path+'./datasets/vg/images/',
                                   image_size=(img_size, img_size), max_objects=7, left_right_flip=True)

    dataloader = torch.utils.data.DataLoader(
        pair_dataset, batch_size=args.batch_size,
        drop_last=False, shuffle=False, num_workers=num_workers)
    
    test_l1, test_l2, test_ssim, test_psnr, test_lpips, test_is, test_fid = 0, 0, 0, 0, 0, 0, 0
    inception_v3 = Inceptionv3OnlyFeature().to(device)
    ssim, psnr, lpips = piq.ssim, piq.psnr, piq.LPIPS()
    count = 0
    batch_count = 0
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            real_images, fake_images = data['images'].to(device), data['fakes'].to(device)
            # torchvision.utils.save_image(real_images, "{}/piq/real/{}_real_{:06d}.jpg".format(args.out_path, args.dataset, idx))
            # torchvision.utils.save_image(fake_images, "{}/piq/fake/{}_fake_{:06d}.jpg".format(args.out_path, args.dataset, idx))
            test_l1 = test_l1 + torch.mean(torch.abs(real_images-fake_images))
            test_l2 = test_l2 + torch.mean(torch.square(real_images-fake_images))
            test_ssim = test_ssim + ssim(real_images, fake_images)
            test_psnr = test_psnr + psnr(real_images, fake_images)
            test_lpips = test_lpips + lpips(real_images, fake_images)
            count = count + real_images.size(0)
            batch_count = batch_count + 1

            real_images = 2. * F.interpolate(real_images, size=(299, 299), mode='nearest') - 1.
            fake_images = 2. * F.interpolate(fake_images, size=(299, 299), mode='nearest') - 1.

            if idx == 0:
                real_feats, real_feats_1000 = inception_v3(real_images)
                fake_feats, ins_feat = inception_v3(fake_images)
            else:
                temp_real_feats, temp_real_feats_1000 = inception_v3(real_images)
                real_feats = torch.cat((real_feats, temp_real_feats), 0)
                real_feats_1000 = torch.cat((real_feats_1000, temp_real_feats_1000), 0)
                temp_fake_feats, temp_ins_feat = inception_v3(fake_images)
                fake_feats = torch.cat((fake_feats, temp_fake_feats), 0)
                ins_feat = torch.cat((ins_feat, temp_ins_feat), 0)
    
    fake_feats = torch.squeeze(fake_feats)
    test_is = inception_score(ins_feat)

    real_feats = torch.squeeze(real_feats)
    print(torch.var_mean(real_feats, unbiased=False))  # [0.1045, 0.3143]
    print(torch.var_mean(fake_feats, unbiased=False))  # [0.1105, 0.3413]
    test_fid = compute_metric(real_feats, fake_feats)

    print('[*] WARNING: The generated samples must be loaded sequentially.')
    print('[*] l1: {} %'.format(100. * test_l1/(batch_count+1.e-6)))
    print('[*] l2: {} %'.format(100. * test_l2/(batch_count+1.e-6)))
    print('[*] ssim: {}'.format(test_ssim/(batch_count+1.e-6)))
    print('[*] psnr: {}'.format(test_psnr/(batch_count+1.e-6)))
    print('[*] lpips: {}'.format(test_lpips/(batch_count+1.e-6)))
    print('[*] is: {}'.format(test_is))
    print('[*] fid: {}'.format(test_fid))
    f= open(args.out_path + "/quantitative_results.txt","w+")
    f.write('[*] l1: {} %\n'.format(100. * test_l1/(batch_count+1.e-6)))
    f.write('[*] l2: {} %\n'.format(100. * test_l2/(batch_count+1.e-6)))
    f.write('[*] ssim: {} \n'.format(test_ssim/(batch_count+1.e-6)))
    f.write('[*] psnr: {} \n'.format(test_psnr/(batch_count+1.e-6)))
    f.write('[*] lpips: {} \n'.format(test_lpips/(batch_count+1.e-6)))
    f.write('[*] IS: {} \n'.format(test_is))
    f.write('[*] FID: {} \n'.format(test_fid))
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='mini-batch size of training data. Default: 16')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='number of total training epoch')
    parser.add_argument('--total_epoch', type=int, default=1,
                        help='number of total training epoch')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for generator')
    parser.add_argument('--out_path', type=str, default='./outputs/tmp/our_d/',
                        help='path to output files')
    parser.add_argument('--model_path', type=str, default='./my_outputs/model/',
                        help='path to output files')
    parser.add_argument('--img_size', type=str, default=128,
                        help='generated image size')
    args = parser.parse_args()
    main(args)

# python test_samples.py --dataset coco --out_path D:/layout2img_ours/test_tsa_v3/ --model_path D:/layout2img_ours/tsa_v3/coco/128/model/
# python test_model.py --dataset vg --out_path  D:/layout2img_ours/test_tsa_v3/ --model_path  D:/layout2img_ours/tsa_v3/vg/128/model/
