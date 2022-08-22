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
from model.generator_my import *
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

def get_dataset(dataset, my_path, img_size):
    if dataset == "coco":
        data = CocoSceneGraphDataset(image_dir=my_path+'/coco/val2017/',
                                     instances_json=my_path+'/coco/annotations/instances_val2017.json',
                                     stuff_json=my_path+'/coco/annotations/stuff_val2017.json',
                                     stuff_only=True, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == 'vg':
        data = VgSceneGraphDataset(vocab_json=my_path+'/vg/vocab.json', h5_path=my_path+'/vg/test.h5',
                                   image_dir=my_path+'/vg/images/',
                                   image_size=(img_size, img_size), max_objects=30, left_right_flip=False)
    return data


def main(args):
    # parameters
    img_size = args.img_size
    pred_classes = 7 if args.dataset == 'coco' else 46
    num_classes = 184 if args.dataset == 'coco' else 179
    num_obj = 8 if args.dataset == 'coco' else 31

    args.out_path = os.path.join(args.out_path, args.dataset, str(args.img_size))

    num_gpus = torch.cuda.device_count()
    print(num_gpus)
    num_workers = 2
    if num_gpus > 1:
        parallel = True
        args.batch_size = args.batch_size * num_gpus
        num_workers = num_workers * num_gpus
    else:
        parallel = False

    # data loader
    train_data = get_dataset(args.dataset, args.data_path, img_size)

    dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        drop_last=False, shuffle=False, num_workers=num_workers)

    # Load model
    device = torch.device('cuda')
    netG = eval(args.model_name)(num_classes=num_classes, pred_classes=pred_classes, output_dim=3).to(device)
    assert os.path.isfile(args.ckpt_path) is True
    state_dict = torch.load(args.ckpt_path)
    print('[*] load_{}'.format(args.ckpt_path))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)
    netG.to(device)

    if parallel:
        netG = DataParallelWithCallback(netG)

    # make dirs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.makedirs(os.path.join(args.out_path, 'model/'))
    if not os.path.exists(os.path.join(args.out_path, 'samples/')):
        os.makedirs(os.path.join(args.out_path, 'samples/'))
    if not os.path.exists(os.path.join(args.out_path, 'masks/')):
        os.makedirs(os.path.join(args.out_path, 'masks/'))
    if not os.path.exists(os.path.join(args.out_path, 'masked_images/')):
        os.makedirs(os.path.join(args.out_path, 'masked_images/'))
    if not os.path.exists(os.path.join(args.out_path, 'grid_samples/')):
        os.makedirs(os.path.join(args.out_path, 'grid_samples/'))
    writer = SummaryWriter(os.path.join(args.out_path, 'log'))

    logger = setup_logger("lostGAN", args.out_path, 0)
    logger.info(netG)

    test_l1, test_l2, test_ssim, test_psnr, test_lpips, test_is, test_fid = 0, 0, 0, 0, 0, 0, 0
    inception_v3 = Inceptionv3OnlyFeature().to(device)
    ssim, psnr, lpips = piq.ssim, piq.psnr, piq.LPIPS()
    count = 0
    batch_count = 0
    
    for epoch in range(args.start_epoch, args.total_epoch):
        netG.eval()
        with torch.no_grad():
            for idx, data in enumerate(tqdm(dataloader)):
                real_images, label, bbox, triples = data
                real_images, label, bbox, triples = real_images.to(device), label.long().to(device).unsqueeze(-1), bbox.float(), triples.cuda()
                randomly_selected = torch.randint(bbox.size(1), (1,))
                selected_bbox = torch.unsqueeze(bbox[:,randomly_selected.item(), :], 1)
                mask = bil.bbox2_mask(selected_bbox, real_images)  # 1 for mask, 0 for non-mask
                mask = mask.cuda()
                masked_images = real_images * (1.-mask)

                # generate images
                content = {'image_contents': masked_images, 'mask': mask, 'label': label.squeeze(dim=-1), 'bbox': bbox, 'triples': triples}
                fake_images_dict = netG(content)
                fake_images = fake_images_dict['image_contents'] * mask + masked_images * (1.-mask)
                
                for i in range(real_images.size(0)):
                    # torchvision.utils.save_image((fake_images[i] - torch.min(fake_images[i]))/(torch.max(fake_images[i]) - torch.min(fake_images[i])+1.e-6), "{}/samples/{}_fake_{:06d}_{:06d}_{:06d}.jpg".format(args.out_path, args.dataset, epoch, idx, i))
                    torchvision.utils.save_image((masked_images[i]+1.)/2., "{}/masked_images/{}_fake_{:06d}_{:06d}_{:06d}.jpg".format(args.out_path, args.dataset, epoch, idx, i), value_range=(0., 1.))
                    torchvision.utils.save_image(mask[i], "{}/masks/{}_fake_{:06d}_{:06d}_{:06d}.jpg".format(args.out_path, args.dataset, epoch, idx, i), value_range=(0., 1.))
                    torchvision.utils.save_image((fake_images[i]+1.)/2., "{}/samples/{}_fake_{:06d}_{:06d}_{:06d}.jpg".format(args.out_path, args.dataset, epoch, idx, i), value_range=(0., 1.))

                real_images = (real_images + 1.)/2.
                fake_images = (fake_images + 1.)/2.

                # metric check
                test_l1 = test_l1 + torch.mean(torch.abs(real_images-fake_images))
                test_l2 = test_l2 + torch.mean(torch.square(real_images-fake_images))
                test_ssim = test_ssim + ssim(real_images, fake_images)
                test_psnr = test_psnr + psnr(real_images, fake_images)
                test_lpips = test_lpips + lpips(real_images, fake_images)
                count = count + real_images.size(0)
                batch_count = batch_count + 1

                real_images = 2. * F.interpolate(real_images, size=(299, 299), mode='nearest') - 1.
                fake_images = 2. * F.interpolate(fake_images, size=(299, 299), mode='nearest') - 1.

                if idx == 0 and epoch == args.start_epoch:
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
    test_is_ = inception_score(ins_feat[:5000])
    real_feats = torch.squeeze(real_feats)
    print(torch.var_mean(fake_feats, unbiased=False))  # [0.1105, 0.3413]
    print(torch.var_mean(real_feats, unbiased=False))  # [0.1045, 0.3143]
    test_fid = compute_metric(real_feats, fake_feats)
    print(batch_count)
    print('[*] l1: {} %'.format(100. * test_l1/(batch_count+1.e-6)))
    print('[*] l2: {} %'.format(100. * test_l2/(batch_count+1.e-6)))
    print('[*] ssim: {}'.format(test_ssim/(batch_count+1.e-6)))
    print('[*] psnr: {}'.format(test_psnr/(batch_count+1.e-6)))
    print('[*] lpips: {}'.format(test_lpips/(batch_count+1.e-6)))
    print('[*] IS: {} {} \n'.format(test_is, test_is_))
    print('[*] FID: {} \n'.format(test_fid))

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
    parser.add_argument('--data_path', type=str, default='./datasets',
                        help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='mini-batch size of training data. Default: 16')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='number of total training epoch')
    parser.add_argument('--total_epoch', type=int, default=18,
                        help='number of total training epoch')
    parser.add_argument('--out_path', type=str, default='./outputs/tmp/our_d/',
                        help='path to output files')
    parser.add_argument('--ckpt_path', type=str, default='D:/layout2img_ours/tsa_v3/coco/128/model/',
                        help='path to checkpoint file')
    parser.add_argument('--model_name', type=str, default='ResnetGenerator128_inpaint_triple_v2',
                        help='file_name')
    parser.add_argument('--img_size', type=int, default=128,
                        help='generated image size')
    parser.add_argument('--metric', type=str, nargs='+', default='l1 l2 ssim psnr lpips is fid')
    args = parser.parse_args()
    main(args)

# python test_model.py --dataset coco --data_path D:/layout2img_ours/datasets/ --out_path D:/layout2img_ours/test_tsa_v3/ --ckpt_path D:/layout2img_ours/tsa_v3/coco/128/model/G_86.pth --model_name ResnetGenerator128_inpaint_triple_v2
# python test_model.py --dataset vg --data_path D:/layout2img_ours/datasets/ --out_path  D:/layout2img_ours/test_tsa_v3/ --ckpt_path D:/layout2img_ours/tsa_v3/vg/128/model/G_47.pth --model_name ResnetGenerator128_inpaint_triple_v2

# model name list
# ResnetGenerator128_inpaint_triple_v2
# ResnetGenerator128_inpaint_subject

# python test_model.py --dataset coco --data_path D:/layout2img_ours/datasets/ --out_path D:/layout2img_ours/subject_only/ --ckpt_path D:/layout2img_ours/subject_only/coco/128/model/G_181.pth --model_name ResnetGenerator128_inpaint_subject

    '''
    print('[*] l1: {} %'.format(100. * test_l1/(batch_count+1.e-6)))
    print('[*] l2: {} %'.format(100. * test_l2/(batch_count+1.e-6)))
    print('[*] ssim: {}'.format(test_ssim/(batch_count+1.e-6)))
    print('[*] psnr: {}'.format(test_psnr/(batch_count+1.e-6)))
    print('[*] lpips: {}'.format(test_lpips/(batch_count+1.e-6)))
    
    fake_dir = '{}/samples/'.format(args.out_path)
    if args.dataset == "coco":
        real_dataset = ImageOnlyDataset(args.data_path + './coco/val2017/',
                                    instances_json=args.data_path+'./coco/annotations/instances_val2017.json',
                                    stuff_json=args.data_path+'./coco/annotations/stuff_val2017.json', image_size=(128, 128), left_right_flip=False)
    elif args.dataset == 'vg':
        real_dataset = ImageOnlyDatasetVG(vocab_json=args.data_path+'./vg/vocab.json', h5_path=args.data_path+'./vg/test.h5',
                                   image_dir=args.data_path+'./vg/images/',
                                   image_size=(img_size, img_size), max_objects=7, left_right_flip=False)
    
    fake_dataset = ImageOnlyDataset(fake_dir, image_size=(299, 299), left_right_flip=False)
    
    real_dataloader = torch.utils.data.DataLoader(
        real_dataset, batch_size=args.batch_size,
        drop_last=False, shuffle=False, num_workers=num_workers)

    fake_dataloader = torch.utils.data.DataLoader(
        fake_dataset, batch_size=args.batch_size,
        drop_last=False, shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for idx, data in enumerate(tqdm(fake_dataloader)):
            images = data['images'].to(device)
            if idx == 0:
                fake_feats, ins_feat = inception_v3(images)
            else:
                temp_fake_feats, temp_ins_feat = inception_v3(images)
                fake_feats = torch.cat((fake_feats, temp_fake_feats), 0)
                ins_feat = torch.cat((ins_feat, temp_ins_feat), 0)
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(real_dataloader)):
            images = data['images'].to(device)
            images = F.interpolate(images, size=(299, 299), mode='nearest')
            if idx == 0:
                real_feats, real_feats_1000 = inception_v3(images)
            else:
                temp_real_feats, temp_real_feats_1000 = inception_v3(images)
                real_feats = torch.cat((real_feats, temp_real_feats), 0)
                real_feats_1000 = torch.cat((real_feats_1000, temp_real_feats_1000), 0)
    '''