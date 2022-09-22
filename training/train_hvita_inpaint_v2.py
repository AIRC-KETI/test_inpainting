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
from data.vg import *
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

def create_status_json(writepath, status_json):
    with open(writepath, "w", encoding='utf-8') as status_json_file:
        json.dump(status_json, status_json_file, indent=4)
        # print('[*] save status_json')

def make_secne_graph(label, triples, json_, vocab, epoch, idx):
    scene_json = []
    for i in range(args.batch_size):
        pred = [vocab[x] for x in triples[i,:,1].tolist()]
        relationships = []
        for j in range(len(pred)):
            relationships.append([triples[i,j,0].item(), pred[j], triples[i,j,2].item()])
        # print(relationships)
        scene_json.append(
            {
                "objects": [json_[str(x)] for x in label[i,:,0].tolist()],
                "relationships": relationships
            }
        )
    # print(scene_json)
    # create_status_json('./scenegraph/figure_{}_{}.json'.format(epoch + 1, idx + 1), scene_json)
    return scene_json

def get_dataset(dataset, img_size):
    if dataset == "coco":
        data = CocoSceneGraphDataset(image_dir='./datasets/coco/train2017/',
                                     instances_json='./datasets/coco/annotations/instances_train2017.json',
                                     stuff_json='./datasets/coco/annotations/stuff_train2017.json',
                                     stuff_only=True, image_size=(img_size, img_size), left_right_flip=True)
    elif dataset == 'vg':
        data = VgSceneGraphDataset(vocab_json='./data/tmp/vocab.json', h5_path='./data/tmp/preprocess_vg/train.h5',
                                   image_dir='./datasets/vg/images/',
                                   image_size=(img_size, img_size), max_objects=30, left_right_flip=True)
    return data


def main(args):
    # parameters
    img_size = args.img_size
    z_dim = 128
    lamb_obj = 1.0
    lamb_app = 1.0
    lamb_img = 0.1
    pred_classes = 7
    num_classes = 184 if args.dataset == 'coco' else 179
    num_obj = 8 if args.dataset == 'coco' else 31

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
    train_data = get_dataset(args.dataset, img_size)

    dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        drop_last=True, shuffle=True, num_workers=8)

    # Load model
    device = torch.device('cuda')
    netG = ResnetGenerator128_hvita(num_classes=num_classes, output_dim=3).to(device)
    netD = HVITADiscriminator128().to(device)
    netOD = HVITADiscriminator64(num_classes=num_classes).to(device)

    if len(glob.glob(args.model_path+'G*')) is 0:
        netG.to(device)
    else:
        state_dict = torch.load(max(glob.iglob(args.model_path+'G*'), key=os.path.getmtime))
        print('[*] load_{}'.format(max(glob.iglob(args.model_path+'G*'), key=os.path.getmtime)))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`nvidia
            new_state_dict[name] = v

        model_dict = netG.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        netG.load_state_dict(model_dict)
        netG.to(device)
    
    if len(glob.glob(args.model_path+'D*')) is 0:
        netD.to(device)
    else:
        state_dict = torch.load(max(glob.iglob(args.model_path+'D*'), key=os.path.getmtime))
        print('[*] load_{}'.format(max(glob.iglob(args.model_path+'D*'), key=os.path.getmtime)))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`nvidia
            new_state_dict[name] = v

        model_dict = netD.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        netD.load_state_dict(model_dict)
        netD.to(device)
    
    parallel = True
    if parallel:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        netOD = nn.DataParallel(netOD)

    g_lr, d_lr, od_lr = args.g_lr, args.d_lr, args.d_lr
    gen_parameters = []
    for key, value in dict(netG.named_parameters()).items():
        if value.requires_grad:
            if 'mapping' in key:
                gen_parameters += [{'params': [value], 'lr': g_lr * 0.1}]
            else:
                gen_parameters += [{'params': [value], 'lr': g_lr}]

    g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))

    dis_parameters = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            dis_parameters += [{'params': [value], 'lr': d_lr}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))

    o_dis_parameters = []
    for key, value in dict(netOD.named_parameters()).items():
        if value.requires_grad:
            o_dis_parameters += [{'params': [value], 'lr': d_lr}]
    od_optimizer = torch.optim.Adam(o_dis_parameters, betas=(0, 0.999))

    # make dirs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.makedirs(os.path.join(args.out_path, 'model/'))
    writer = SummaryWriter(os.path.join(args.out_path, 'log'))

    logger = setup_logger("HVITA", args.out_path, 0)
    logger.info(netG)
    logger.info(netD)
    logger.info(netOD)

    start_time = time.time()
    l1_loss = nn.DataParallel(nn.L1Loss())

    total_idx = len(dataloader.dataset) / args.batch_size
    print('[*] total_idx: ', total_idx)
    zeros = torch.zeros([1], dtype=torch.int)
    ones = torch.ones([1], dtype=torch.int)
    o_254 = 126 * ones
    o_255 = 127 * ones

    for epoch in range(args.start_epoch, args.total_epoch):
        netG.train()
        netD.train()
        netOD.train()

        for idx, data in enumerate(tqdm(dataloader)):
            real_images, label, bbox, triples = data
            real_images, label, bbox, triples = real_images.to(device), label.long().to(device).unsqueeze(-1), bbox.float(), triples.cuda()
            randomly_selected = torch.randint(3, (1,))  # torch.randint(bbox.size(1), (1,))
            selected_bbox = torch.unsqueeze(bbox[:,randomly_selected.item(), :], 1)
            mask = bil.bbox2_mask(selected_bbox, real_images, is_train=True)  # 1 for mask, 0 for non-mask
            mask = mask.cuda()
            masked_images = real_images * (1.-mask) # + 0.5 * mask
            b, h, w = real_images.size(0), real_images.size(2), real_images.size(3)
            x1, y1, x2, y2 = (w * selected_bbox[:,0,0]).int(), (h * selected_bbox[:,0,1]).int(), (w * (selected_bbox[:,0,2]+selected_bbox[:,0,0])).int(), (h * (selected_bbox[:,0,3]+selected_bbox[:,0,1])).int()  # [B, 1]
            x1, y1, x2, y2 = torch.min(torch.max(zeros, x1), o_254), torch.min(torch.max(zeros, y1), o_254), torch.min(torch.max(x1+1, x2), o_255), torch.min(torch.max(y1+1, y2), o_255)

            for i in range(real_images.size(0)):
                if i == 0:
                    obj_images = F.interpolate(torch.unsqueeze(real_images[i,:,y1[i]:y2[i], x1[i]:x2[i]], 0), (int(h/2), int(w/2)))
                else:
                    obj_images = torch.cat((obj_images, F.interpolate(torch.unsqueeze(real_images[i,:,y1[i]:y2[i], x1[i]:x2[i]], 0), (int(h/2), int(w/2)))), 0)
            
            # update D network
            netD.zero_grad()
            d_out_real = netD(real_images)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            z = torch.randn(real_images.size(0), num_obj, z_dim).to(device)
            content = {'image_contents': masked_images, 'mask': mask, 'label': label[:,randomly_selected.item()].squeeze(dim=-1), 'bbox': selected_bbox, 'triples': triples, 'z': z, 'batch_randomly_selected': randomly_selected.expand(b, -1)}
            fake_images_dict = netG(content)
            # fake_images = fake_images_dict['image_contents'] * mask + (1.-mask) * masked_images
            fake_images = fake_images_dict['image_contents']
            fake_obj_images = fake_images_dict['object']
            d_out_fake = netD(fake_images.detach())
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # update OD network
            netOD.zero_grad()
            d_out_o_real = netOD(obj_images, label[:,randomly_selected.item()].squeeze(dim=-1))
            d_loss_o_real = torch.nn.ReLU()(1.0 - d_out_o_real).mean()
            d_out_o_fake = netOD(fake_obj_images.detach(), label[:,randomly_selected.item()].squeeze(dim=-1))
            d_loss_o_fake = torch.nn.ReLU()(1.0 + d_out_o_fake).mean()

            od_loss = d_loss_o_real + d_loss_o_fake
            od_loss.backward()
            od_optimizer.step()

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                g_out_fake = netD(fake_images)
                g_loss_fake = - g_out_fake.mean()

                g_out_o_fake = netOD(fake_obj_images, label[:,randomly_selected.item()].squeeze(dim=-1))
                g_loss_o_fake = - g_out_o_fake.mean()

                pixel_loss = l1_loss(fake_images, real_images).mean()
                o_pixel_loss = l1_loss(obj_images, fake_obj_images).mean()

                g_loss = g_loss_fake + 0.1 * pixel_loss + 5. * g_loss_o_fake + 0.1 * o_pixel_loss
                g_loss.backward()
                g_optimizer.step()

            if idx % (total_idx/4 + 1) == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                logger.info("Time Elapsed: [{}]".format(elapsed))
                writer.add_image("real images", make_grid(real_images.cpu().data * 0.5 + 0.5, nrow=int(args.batch_size/4)), epoch * len(dataloader) + idx + 1)
                writer.add_image("fake images", make_grid(fake_images.cpu().data * 0.5 + 0.5, nrow=int(args.batch_size/4)), epoch * len(dataloader) + idx + 1)
                writer.add_image("masked images", make_grid(masked_images.cpu().data * 0.5 + 0.5, nrow=int(args.batch_size/4)), epoch * len(dataloader) + idx + 1)
                writer.add_image("masks", make_grid(mask.cpu().data * 0.5 + 0.5, nrow=int(args.batch_size/4)), epoch * len(dataloader) + idx + 1)

                writer.add_scalars("D_loss_real", {"real": d_loss_real.item(),
                                                    'real_obj': d_loss_o_real.item(),
                                                       "loss": d_loss.item()}, epoch * len(dataloader) + idx + 1)
                writer.add_scalars("D_loss_fake", {"fake": d_loss_fake.item(),
                                                    'obj': d_loss_o_fake.item(),}, epoch * len(dataloader) + idx + 1)
                writer.add_scalars("G_loss", {"img": g_loss_fake.item(),
                                                "obj": g_loss_o_fake.item(),
                                                "pixel": pixel_loss.item(),
                                                "loss": g_loss.item()}, epoch * len(dataloader) + idx + 1)
                # writer.add_text('t/json', str(scene_graphs), epoch*len(dataloader) + idx + 1)

        # save model
        # if (epoch + 1) % 5 == 0:
        torch.save(netG.state_dict(), os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch + 1)))
        torch.save(netD.state_dict(), os.path.join(args.out_path, 'model/', 'D_%d.pth' % (epoch + 1)))
        torch.save(netOD.state_dict(), os.path.join(args.out_path, 'model/', 'OD_%d.pth' % (epoch + 1)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size of training data. Default: 16')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='number of total training epoch')
    parser.add_argument('--total_epoch', type=int, default=200,
                        help='number of total training epoch')
    parser.add_argument('--d_lr', type=float, default=0.001,
                        help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.001,
                        help='learning rate for generator')
    parser.add_argument('--out_path', type=str, default='./hvita_outputs/tmp/our_d/',
                        help='path to output files')
    parser.add_argument('--model_path', type=str, default='./hvita_outputs/model/',
                        help='path to output files')
    parser.add_argument('--img_size', type=str, default=128,
                        help='generated image size')
    args = parser.parse_args()
    main(args)

# python train_hvita_inpaint.py --dataset coco --out_path hvita_outputs/ --model_path ./hvita_outputs/coco/128/model/
# python train_hvita_inpaint.py --dataset vg --out_path hvita_outputs/ --model_path ./hvita_outputs/vg/128/model/