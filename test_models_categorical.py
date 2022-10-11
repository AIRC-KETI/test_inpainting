import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
from data.vg_fake_pred import *
from model.resnet_generator_app_v2 import *
from model.generator_my import *
from data.image_only_loader import *
from model.deepfill_v2_tf import *
from model.ocgan_inpaint import *
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
from torchvision.utils import draw_bounding_boxes

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

def load_models(i, netG, args):
    if args.model_name[i] =='DeepFillv2':
        state_dict = torch.load(args.ckpt_path[i])['G']
        print('[*] loaded_{}'.format(args.ckpt_path[i]))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'generator.'+ k  # remove `module.`nvidia
            new_state_dict[name] = v

        model_dict = netG.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        return model_dict if len(pretrained_dict) != 0 else state_dict
    else:
        state_dict = torch.load(args.ckpt_path[i])
        print('[*] loaded_{}'.format(args.ckpt_path[i]))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`nvidia
            new_state_dict[name] = v

        model_dict = netG.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        return model_dict if len(pretrained_dict) != 0 else state_dict


def main(args):
    # parameters
    img_size = args.img_size
    pred_classes = 7 if args.dataset == 'coco' else 46
    num_classes = 184 if args.dataset == 'coco' else 179
    num_obj = 8 if args.dataset == 'coco' else 31

    args.out_path = os.path.join(args.out_path, args.dataset, str(args.img_size))
    model_count = len(args.model_name)
    num_gpus = torch.cuda.device_count()
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
        drop_last=True, shuffle=False, num_workers=num_workers)

    # Load model
    # list
    # deepfillv2  == states_tf_places2.pth
    # ResnetGenerator128_inpaint_subject
    # ResnetGenerator128_subject
    # HVITA_with_GT_label
    # TripleGenerator_v2
    device = torch.device('cuda')
    netG = []
    for i in range(model_count):
        if args.model_name[i] == 'DeepFillv2':
            print('[*] {}'.format(args.model_name[i]))
            temp_netG = eval(args.model_name[i])(cnum_in=5, cnum=48, return_flow=False).to(device)  # for DeepFillv2
        elif args.model_name[i] == 'TripleGenerator_v2' or args.model_name[i] == 'ResnetGenerator128_hvita_triple' or args.model_name[i] == 'OCGANGenerator':
            print('[**] {}'.format(args.model_name[i]))
            temp_netG = eval(args.model_name[i])(num_classes=num_classes, pred_classes=pred_classes, output_dim=3).to(device)  # Triplelostgan
        else:  # ResnetGenerator128_hvita ResnetGenerator128_gen_subject ResnetGenerator128_inpaint_subject
            print('[***] {}'.format(args.model_name[i]))
            temp_netG = eval(args.model_name[i])(num_classes=num_classes, pred_classes=7, output_dim=3).to(device)  # for lostgan, hvita
        netG.append(temp_netG)

    assert (os.path.isfile(args.ckpt_path[0]) is True) and (len(args.ckpt_path) == len(args.model_name))

    for i in range(model_count):
        model_dict = load_models(i, netG[i], args)
        netG[i].load_state_dict(model_dict)
        netG[i].to(device)

    if parallel:
        for i in range(model_count):
            netG[i] = DataParallelWithCallback(netG[i])

    # make dirs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'grid_samples/')):
        os.makedirs(os.path.join(args.out_path, 'grid_samples/'))
    
    vocab_json = args.data_path+'/vg/vocab.json'
    with open(vocab_json, 'r') as f:
        vocab = json.load(f)
    obj_len = len(vocab['object_idx_to_name'])
    for j in range(obj_len):
        if not os.path.exists(os.path.join(args.out_path, 'grid_samples/{:03d}_{}/'.format(j, vocab['object_idx_to_name'][j]))):
            os.makedirs(os.path.join(args.out_path, 'grid_samples/{:03d}_{}/'.format(j, vocab['object_idx_to_name'][j])))
    if not os.path.exists(os.path.join(args.out_path, 'scene_graph/')):
        os.makedirs(os.path.join(args.out_path, 'scene_graph/'))
    writer = SummaryWriter(os.path.join(args.out_path, 'log'))

    logger = setup_logger("lostGAN", args.out_path, 0)
    logger.info(netG)

    test_l1, test_l2, test_ssim, test_psnr, test_lpips, test_is, test_is_, test_fid, test_scenefid, remain_ratio = [[0]*obj_len]*model_count, [[0]*obj_len]*model_count, [[0]*obj_len]*model_count, [[0]*obj_len]*model_count, [[0]*obj_len]*model_count, [[0]*obj_len]*model_count, [[0]*obj_len]*model_count, [[0]*obj_len]*model_count, [[0]*obj_len]*model_count, [[1]*obj_len]*model_count
    inception_v3 = Inceptionv3OnlyFeature().to(device)
    ssim, psnr, lpips = piq.ssim, piq.psnr, piq.LPIPS()
    count = [0]*obj_len
    batch_count = 0
    zeros = torch.zeros([1], dtype=torch.int)
    ones = torch.ones([1], dtype=torch.int)
    o_126 = 126 * ones
    o_127 = 127 * ones
    o_254 = 254 * ones
    o_255 = 255 * ones
    mean_mask_ratio = 0.

    for i in range(model_count):
        netG[i].eval()
    for epoch in range(args.start_epoch, args.total_epoch):
        with torch.no_grad():
            for idx, data in enumerate(tqdm(dataloader)):
                real_images, label, bbox, triples = data
                real_images, label, bbox, triples = real_images.to(device), label.long().to(device).unsqueeze(-1), bbox.float(), triples.cuda()
                for j in range(1):
                    selected_bbox = torch.unsqueeze(bbox[:,j,:], 1)
                    mask, sel_xyxy = bil.bbox2_mask(selected_bbox, real_images, mode=45)  # 1 for mask, 0 for non-mask
                    mask = mask.cuda()
                    masked_images = real_images * (1.-mask)
                    temp_remain_ratio = 1. - mask.mean()
                    b, h, w = real_images.size(0), real_images.size(2), real_images.size(3)
                    mean_mask_ratio = mean_mask_ratio + mask.mean() * b
                    # generate images
                    z = torch.randn(b, num_obj, 128).to(device)
                    batch_randomly_selected = torch.LongTensor([j]).repeat(real_images.size(0))
                    hw = torch.ones([b], dtype=torch.int).to(device) * args.img_size
                    content = {'image_contents': masked_images, 'mask': mask, 'label': label.squeeze(dim=-1), 'bbox': bbox.cuda(), 'triples': triples, 'z': z, 'batch_randomly_selected': batch_randomly_selected, 'hw': hw, 'spatial': None, 'avg': None}

                    for i in range(model_count):  # [model_count, b, 3, h, w]
                        fake_images_dict = netG[i](content)
                        temp_fake_images = (fake_images_dict['image_contents'] * mask + masked_images * (1.-mask)).unsqueeze(0)
                        if i == 0:
                            fake_images = temp_fake_images
                        else:
                            fake_images = torch.cat((fake_images, temp_fake_images), 0)

                    x1, y1, x2, y2 = (real_images.size(3) * bbox[:,j,0]).int(), (real_images.size(2) * bbox[:,j,1]).int(), (real_images.size(3) * (bbox[:,j,2]+bbox[:,j,0])).int(), (real_images.size(2) * (bbox[:,j,3]+bbox[:,j,1])).int()  # [B, 1]
                    x1, y1, x2, y2 = torch.minimum(torch.maximum(zeros, x1), x2-1), torch.minimum(torch.maximum(zeros, y1), y2-1), torch.minimum(torch.maximum(x1+1, x2), o_255), torch.minimum(torch.maximum(y1+1, y2), o_255)
                    for i in range(real_images.size(0)):
                        if epoch == args.start_epoch and not os.path.exists("{}/scene_graph/{}_sg_{:06d}_{:06d}_{:06d}.png".format(args.out_path, args.dataset, epoch, idx, i)):
                            img = vis.draw_scene_graph(label[i,:,0], triples[i], vocab)
                            imwrite("{}/scene_graph/{}_sg_{:06d}_{:06d}_{:06d}.png".format(args.out_path, args.dataset, epoch, idx, i), img)

                        selected_xyminmax = sel_xyxy[i] * torch.Tensor([[w, h, w, h]])
                        bbox_image = draw_bounding_boxes((255 * 0.5 * (masked_images[i]+1.).cpu()).type(torch.uint8), selected_xyminmax, labels=[vocab['object_idx_to_name'][label[i, j].item()]]).cuda()
                        bbox_image = 2. * (bbox_image.float()/255.) - 1.
                        image_batch = torch.stack(((2.*mask[i]-1.).repeat(3, 1, 1), bbox_image))
                        for k in range(model_count):
                            image_batch = torch.cat((image_batch, torch.unsqueeze(fake_images[k][i], 0)), 0)

                        image_batch = torch.cat((image_batch, torch.unsqueeze(real_images[i], 0)), 0)
                        torchvision.utils.save_image((image_batch+1.)/2., "{}/{}_grid_{:04d}_{:04d}_{:04d}_{:03d}.jpg".format(os.path.join(args.out_path, 'grid_samples/{:03d}_{}/'.format(label[i, j].item(), vocab['object_idx_to_name'][label[i, j].item()])), args.dataset, epoch, idx, i, j), value_range=(0., 1.))
                    metric_real_images = (real_images + 1.)/2.
                    metric_fake_images = (fake_images + 1.)/2.
                    # metric check
                    
                    for i in range(model_count):
                        for ii in range(real_images.size(0)):
                            patch_real = metric_real_images[ii,:,y1[ii]:y2[ii],x1[ii]:x2[ii]].unsqueeze(0)
                            patch_fake = metric_fake_images[i,ii,:,y1[ii]:y2[ii],x1[ii]:x2[ii]].unsqueeze(0)
                            if patch_real.size(2) < 1 or patch_real.size(3) < 1:
                                pass
                            else:
                                temp_real_patch = F.interpolate(patch_real, (h, w))
                                temp_fake_patch = F.interpolate(patch_fake, (h, w))
                                temp_test_l1 = torch.mean(torch.abs(temp_real_patch - temp_fake_patch))
                                test_l1[i][label[ii,j].item()] = test_l1[i][label[ii,j].item()] + temp_test_l1
                                test_l2[i][label[ii,j].item()] = test_l2[i][label[ii,j].item()] + torch.mean(torch.square(temp_real_patch-temp_fake_patch))  # [model_count]
                                if (temp_test_l1 <= 0.05 * b and remain_ratio[i][label[ii,j].item()] > temp_remain_ratio):
                                    remain_ratio[i][label[ii,j].item()] = temp_remain_ratio
                                test_ssim[i][label[ii,j].item()] = test_ssim[i][label[ii,j].item()] + ssim(temp_real_patch, temp_fake_patch)
                                test_psnr[i][label[ii,j].item()] = test_psnr[i][label[ii,j].item()] + psnr(temp_real_patch, temp_fake_patch)
                                test_lpips[i][label[ii,j].item()] = test_lpips[i][label[ii,j].item()] + lpips(temp_real_patch, temp_fake_patch)
                                if i == 0:
                                    count[label[ii, j].item()] = count[label[ii, j].item()] + 1
                                else:
                                    pass
                                if i == 0:
                                    batch_fake_patch = temp_fake_patch
                                elif i == model_count-1:
                                    torchvision.save_image((torch.cat((patch_real, batch_fake_patch), 0)+1.)/2., "{}/{}_grid_patch_{:04d}_{:04d}_{:04d}_{:03d}.jpg".format(os.path.join(args.out_path, 'grid_samples/{:03d}_{}/'.format(label[i, j].item(), vocab['object_idx_to_name'][label[i, j].item()])), args.dataset, epoch, idx, i, j), value_range=(0., 1.))
                                else:
                                    batch_fake_patch = torch.cat((batch_fake_patch, temp_fake_patch), 0)

                    batch_count = batch_count + 1
                    '''
                    inc_real_images = F.interpolate(real_images, size=(299, 299), mode='nearest')  # [b, c, h, w]
                    inc_fake_images = F.interpolate(fake_images, size=(3, 299, 299), mode='nearest')  # [model_count, b, c, h, w]
                    if idx == 0 and epoch == args.start_epoch:
                        real_feats, real_feats_1000 = inception_v3(inc_real_images)  # [b, -1]
                        fake_feats, ins_feat = inception_v3(inc_fake_images.view(model_count * b, 3, 299, 299))
                        fake_feats, ins_feat = fake_feats.view(model_count, b, -1), ins_feat.view(model_count, b, -1)  # [model_count, b, -1]
                    else:
                        temp_real_feats, temp_real_feats_1000 = inception_v3(inc_real_images)  # [b, -1]
                        real_feats, real_feats_1000 = torch.cat((real_feats, temp_real_feats), 0), torch.cat((real_feats_1000, temp_real_feats_1000), 0)    # [model_count, b, -1]
                        temp_fake_feats, temp_ins_feat = inception_v3(inc_fake_images.view(model_count * b, 3, 299, 299))
                        temp_fake_feats, temp_ins_feat = temp_fake_feats.view(model_count, b, -1), temp_ins_feat.view(model_count, b, -1)
                        fake_feats, ins_feat = torch.cat((fake_feats, temp_fake_feats), 1), torch.cat((ins_feat, temp_ins_feat), 1)  # [model_count, b, -1]
                    # sceneFID... 

                    if epoch < args.start_epoch+1:
                        j = randomly_selected
                        x1, y1, x2, y2 = (real_images.size(3) * bbox[:,j,0]).int(), (real_images.size(2) * bbox[:,j,1]).int(), (real_images.size(3) * (bbox[:,j,2]+bbox[:,j,0])).int(), (real_images.size(2) * (bbox[:,j,3]+bbox[:,j,1])).int()  # [B, 1]
                        x1, y1, x2, y2 = torch.min(torch.max(zeros, x1), o_254), torch.min(torch.max(zeros, y1), o_254), torch.min(torch.max(ones, x2), o_255), torch.min(torch.max(ones, y2), o_255)
                        for k in range(label.size(0)):  # in batch, resize it
                            if k == 0:
                                real_images_patch = 2. * F.interpolate(torch.unsqueeze(real_images[k, :, y1[k]:y2[k], x1[k]:x2[k]], 0), size=(299, 299), mode='nearest') - 1.  # [NCHW]
                            else:
                                real_images_patch = torch.cat((real_images_patch, 2. * F.interpolate(torch.unsqueeze(real_images[k, :, y1[k]:y2[k], x1[k]:x2[k]], 0), size=(299, 299), mode='nearest') - 1.), 0)
                        real_images_patch = real_images_patch.repeat(model_count, 1, 1, 1)
                        
                        for k in range(label.size(0)):  # in batch, resize it
                            for l in range(model_count):
                                    if k == 0 and l == 0:
                                        fake_images_patch = 2. * F.interpolate(torch.unsqueeze(fake_images[l,k, :, y1[k]:y2[k], x1[k]:x2[k]], 0), size=(299, 299), mode='nearest') - 1.  # [NCHW]
                                    else:
                                        fake_images_patch = torch.cat((fake_images_patch, 2. * F.interpolate(torch.unsqueeze(fake_images[l*b+k, :, y1[k]:y2[k], x1[k]:x2[k]], 0), size=(299, 299), mode='nearest') - 1.), 0)

                        if idx == 0 and epoch == args.start_epoch:
                            real_scene_feats, real_scene_feats_1000 = inception_v3(real_images_patch)
                            fake_scene_feats, ins_scene_feat = inception_v3(fake_images_patch)
                        else:
                            temp_real_feats, temp_real_feats_1000 = inception_v3(real_images)
                            real_scene_feats = torch.cat((real_scene_feats, temp_real_feats), 0)
                            real_scene_feats_1000 = torch.cat((real_scene_feats_1000, temp_real_feats_1000), 0)
                            temp_fake_feats, temp_ins_feat = inception_v3(fake_images)
                            fake_scene_feats = torch.cat((fake_scene_feats, temp_fake_feats), 0)
                            ins_scene_feat = torch.cat((ins_scene_feat, temp_ins_feat), 0)

                        for j in range(label.size(1)):
                            x1, y1, x2, y2 = (real_images.size(3) * bbox[:,j,0]).int(), (real_images.size(2) * bbox[:,j,1]).int(), (real_images.size(3) * (bbox[:,j,2]+bbox[:,j,0])).int(), (real_images.size(2) * (bbox[:,j,3]+bbox[:,j,1])).int()  # [B, 1]
                            x1, y1, x2, y2 = torch.min(torch.max(zeros, x1), o_254), torch.min(torch.max(zeros, y1), o_254), torch.min(torch.max(ones, x2), o_255), torch.min(torch.max(ones, y2), o_255)
                            for k in range(label.size(0)):
                                if k == 0:
                                    real_images_patch = 2. * F.interpolate(torch.unsqueeze(real_images[k, :, y1[k]:y2[k], x1[k]:x2[k]], 0), size=(299, 299), mode='nearest') - 1.  # [NCHW]
                                    fake_images_patch = 2. * F.interpolate(torch.unsqueeze(fake_images[k, :, y1[k]:y2[k], x1[k]:x2[k]], 0), size=(299, 299), mode='nearest') - 1.  # [NCHW]
                                else:
                                    real_images_patch = torch.cat((real_images_patch, 2. * F.interpolate(torch.unsqueeze(real_images[k, :, y1[k]:y2[k], x1[k]:x2[k]], 0), size=(299, 299), mode='nearest') - 1.), 0)
                                    fake_images_patch = torch.cat((fake_images_patch, 2. * F.interpolate(torch.unsqueeze(fake_images[k, :, y1[k]:y2[k], x1[k]:x2[k]], 0), size=(299, 299), mode='nearest') - 1.), 0)                

                            if idx == 0 and epoch == args.start_epoch:
                                real_scene_feats, real_scene_feats_1000 = inception_v3(real_images_patch)
                                fake_scene_feats, ins_scene_feat = inception_v3(fake_images_patch)
                            else:
                                temp_real_feats, temp_real_feats_1000 = inception_v3(real_images)
                                real_scene_feats = torch.cat((real_scene_feats, temp_real_feats), 0)
                                real_scene_feats_1000 = torch.cat((real_scene_feats_1000, temp_real_feats_1000), 0)
                                temp_fake_feats, temp_ins_feat = inception_v3(fake_images)
                                fake_scene_feats = torch.cat((fake_scene_feats, temp_fake_feats), 0)
                                ins_scene_feat = torch.cat((ins_scene_feat, temp_ins_feat), 0)
                        '''
    # fake_feats = torch.squeeze(fake_feats)
    # real_feats = torch.squeeze(real_feats)
    # fake_scene_feats = torch.squeeze(fake_scene_feats)
    # real_scene_feats = torch.squeeze(real_scene_feats)
    # split_size = int(ins_feat.size(0)/model_count)
    '''
    for i in range(model_count):
        test_is[i] = inception_score(ins_feat[i])
        test_is_[i] = inception_score(ins_feat[i]) if ins_feat.size(0) >= 5000 else 0
        test_fid[i] = compute_metric(real_feats, fake_feats[i])
        # test_scenefid[i] = compute_metric(real_scene_feats[split_size*i:split_size*(i+1)-1], fake_scene_feats[split_size*i:split_size*(i+1)-1])
    '''
    for i in range(model_count):
        print('[ ** ] model name: {}'.format(args.model_name[i]))
        print('[*] mean mask ratio: {} %'.format(100. * mean_mask_ratio/(sum(count)+1.e-6)))
        print('[*] count: {} \n'.format(sum(count)+1.e-6))
        print('[*] l1: {} %'.format(100. * sum(test_l1[i])/(sum(count)+1.e-6)))
        print('[*] l2: {} %'.format(100. * sum(test_l2[i])/(sum(count)+1.e-6)))
        print('[*] ssim: {}'.format(sum(test_ssim[i])/(sum(count)+1.e-6)))
        print('[*] psnr: {}'.format(sum(test_psnr[i])/(sum(count)+1.e-6)))
        print('[*] lpips: {}'.format(sum(test_lpips[i])/(sum(count)+1.e-6)))
        for j in range(obj_len):
            print('================[{}]================'.format(vocab['object_idx_to_name'][j]))
            print('[*] count: {} \n'.format(count[j]+1.e-6))
            print('[*] l1: {} %'.format(100. * test_l1[i][j]/(count[j]+1.e-6)))
            print('[*] l2: {} %'.format(100. * test_l2[i][j]/(count[j]+1.e-6)))
            print('[*] ssim: {}'.format(test_ssim[i][j]/(count[j]+1.e-6)))
            print('[*] psnr: {}'.format(test_psnr[i][j]/(count[j]+1.e-6)))
            print('[*] lpips: {}'.format(test_lpips[i][j]/(count[j]+1.e-6)))

    f= open(args.out_path + "/quantitative_results.txt","w+")
    for i in range(model_count):
        f.write('=====================================================\n')
        f.write('[ ** ] model name: {} %\n'.format(args.model_name[i]))
        f.write('[*] mean mask ratio: {} %\n'.format(100. * mean_mask_ratio/(sum(count)+1.e-6)))
        f.write('[*] l1: {} %\n'.format(100. * sum(test_l1[i])/(sum(count)+1.e-6)))
        f.write('[*] l2: {} %\n'.format(100. * sum(test_l2[i])/(sum(count)+1.e-6)))
        f.write('[*] ssim: {}\n'.format(sum(test_ssim[i])/(sum(count)+1.e-6)))
        f.write('[*] psnr: {}\n'.format(sum(test_psnr[i])/(sum(count)+1.e-6)))
        f.write('[*] lpips: {}\n'.format(sum(test_lpips[i])/(sum(count)+1.e-6)))
        for j in range(obj_len):
            f.write('================[{}]================'.format(vocab['object_idx_to_name'][j]))
            f.write('[*] count: {} \n'.format(count[j]+1.e-6))
            f.write('[*] l1: {} %\n'.format(100. * test_l1[i][j]/(count[j]+1.e-6)))
            f.write('[*] l2: {} %\n'.format(100. * test_l2[i][j]/(count[j]+1.e-6)))
            f.write('[*] ssim: {} \n'.format(test_ssim[i][j]/(count[j]+1.e-6)))
            f.write('[*] psnr: {} \n'.format(test_psnr[i][j]/(count[j]+1.e-6)))
            f.write('[*] lpips: {} \n'.format(test_lpips[i][j]/(count[j]+1.e-6)))

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
    parser.add_argument('--total_epoch', type=int, default=10,
                        help='number of total training epoch')
    parser.add_argument('--out_path', type=str, default='./outputs/tmp/our_d/',
                        help='path to output files')
    parser.add_argument('--ckpt_path', type=str, nargs='+', default='D:/layout2img_ours/tsa_v3/coco/128/model/',
                        help='path to checkpoint file')
    parser.add_argument('--model_name', type=str, nargs='+', default='ResnetGenerator128_inpaint_triple_v2',
                        help='file_name')
    parser.add_argument('--img_size', type=int, default=128,
                        help='generated image size')
    parser.add_argument('--metric', type=str, nargs='+', default='l1 l2 ssim psnr lpips is fid')
    args = parser.parse_args()
    main(args)

'''
python test_models.py \
--dataset vg --out_path gang \
--ckpt_path ./pretrained/lostgan.pth ./pretrained/lostgan_.pth ./pretrained/hvita.pth ./pretrained/states_tf_places2.pth ./outputs/tmp/our_d/vg/256/model/G_393.pth \
--model_name ResnetGenerator128_gen_subject ResnetGenerator128_inpaint_subject ResnetGenerator128_hvita DeepFillv2 TripleGenerator_v2

python test_models_.py --dataset vg --out_path square --ckpt_path ./pretrained/lostgan.pth ./pretrained/lostgan_.pth ./pretrained/hvita.pth ./pretrained/states_tf_places2.pth ./outputs/tmp/our_d/vg/256/model/G_393.pth --model_name ResnetGenerator128_gen_subject ResnetGenerator128_inpaint_subject ResnetGenerator128_hvita DeepFillv2 TripleGenerator_v2 --

python test_models_.py --dataset vg --out_path gang --ckpt_path ./pretrained/lostgan.pth ./pretrained/lostgan_.pth ./pretrained/hvita.pth ./pretrained/states_tf_places2.pth --model_name ResnetGenerator128_gen_subject ResnetGenerator128_inpaint_subject ResnetGenerator128_hvita DeepFillv2

CUDA_VISIBLE_DEVICES=2,3 python test_models_.py --dataset vg --out_path september/ocgan_inpaint_test --ckpt_path ./pretrained/hvita_0905.pth ./pretrained/hvita_triple.pth ./pretrained/lostgan.pth ./pretrained/lostgan_.pth ./pretrained/states_tf_places2.pth ./pretrained/G_393.pth ./september/ocgan_inpaint/vg/128/model/G_199.pth --model_name ResnetGenerator128_gen_subject ResnetGenerator128_inpaint_subject ResnetGenerator128_hvita DeepFillv2 TripleGenerator_v2 OCGANGenerator
'''