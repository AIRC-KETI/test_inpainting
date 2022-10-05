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
from data.vg import *
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

def crop_resize_objects(images, y1, y2, x1, x2):
    for i in range(images.size(0)):
        for j in range(y1.size(1)):
            if j == 0:
                in_obj_images = torch.unsqueeze(F.interpolate(torch.unsqueeze(images[i,:,y1[i,j]:y2[i,j], x1[i,j]:x2[i,j]], 0), (32, 32)), 0)
            else:
                in_obj_images = torch.cat((in_obj_images, torch.unsqueeze(F.interpolate(torch.unsqueeze(images[i,:,y1[i,j]:y2[i,j], x1[i,j]:x2[i,j]], 0), (32, 32)), 0)), 1)
        if i == 0:
            obj_images = in_obj_images
        else:
            obj_images = torch.cat((obj_images, in_obj_images), 0)
    return obj_images

def sgsm_loss(obj_embeddings, spatial, avg, cos):
    local_cos, global_cos = local_global(obj_embeddings, spatial, avg, cos)
    # print('[*] local global: {}, {}'.format(local_cos.mean(), global_cos.mean()))
    for i in range(obj_embeddings.size(0)-1):
        roll_obj_embeddings = torch.roll(obj_embeddings, 1+i, 0)
        if i == 0:
            roll_local_cos, roll_global_cos = local_global(roll_obj_embeddings, spatial, avg, cos)
        else:
            temp_roll_local_cos, temp_roll_global_cos = local_global(roll_obj_embeddings, spatial, avg, cos)
            roll_local_cos = torch.cat((roll_local_cos, temp_roll_local_cos), 0)
            roll_global_cos = torch.cat((roll_global_cos, temp_roll_global_cos), 0)
    # print('[**] local global: {}, {}'.format(roll_local_cos.mean(), roll_global_cos.mean()))
    local_cos = local_cos.sum()
    roll_local_cos = roll_local_cos.sum()
    global_cos = global_cos.sum()
    roll_global_cos = roll_global_cos.sum()
    sgsm_l = -torch.log(torch.maximum(local_cos, torch.ones_like(local_cos)))
    sgsm_l = sgsm_l + torch.log(torch.maximum(roll_local_cos, torch.ones_like(roll_local_cos)))
    sgsm_g = -torch.log(torch.maximum(global_cos, torch.ones_like(global_cos)))
    sgsm_g = sgsm_g + torch.log(torch.maximum(roll_global_cos, torch.ones_like(roll_global_cos)))
    sgsm = (sgsm_l + sgsm_g)
    return sgsm

def local_global(obj_embeddings, spatial, avg, cos):
    gamma1  = 5.
    gamma2 = 5.
    gamma3 = 10.
    gv = torch.bmm(obj_embeddings, spatial)
    s_ij = gamma1 * torch.exp(gv) / (torch.exp(gv).sum(dim=-1, keepdim=True)+1.e-6)  # [b, o, 17*17]
    exp_s_ij = torch.exp(s_ij)
    gj = torch.bmm(exp_s_ij, spatial.permute(0, 2, 1))  # [b, o, 256]
    gj_ = gj / (exp_s_ij.sum(dim=-1, keepdim=True)+1.e-6)  # [b, o, 256]
    gj__ = torch.exp(gamma2 *cos(obj_embeddings, gj_)).sum(dim=-1, keepdim=True)
    local_cos = torch.log(torch.maximum(gj__, torch.ones_like(gj__))) * (1/gamma2)  # [b]
    global_cos = cos(avg, obj_embeddings.mean(dim=1))  # [b]
    local_cos = torch.exp(gamma3 * local_cos)
    global_cos = torch.exp(gamma3 * global_cos)
    # print('[***] gv s_ij gj gj_ local : {}, {}, {}, {}, {}'.format(gv[0], s_ij[0], gj[0], gj_[0], local_cos[0]))
    return local_cos, global_cos

def main(args):
    # parameters
    img_size = args.img_size
    z_dim = 128
    lamb_obj = 1.0
    lamb_app = 1.0
    lamb_img = 0.1
    pred_classes = 7 if args.dataset == 'coco' else 46
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
        drop_last=True, shuffle=True, num_workers=12, prefetch_factor=8)

    # Load model
    device = torch.device('cuda')
    netG = OCGANGenerator(num_classes=num_classes, output_dim=3, pred_classes=pred_classes).to(device)
    netD = Discriminator().to(device)
    # netOD = ObjectDiscriminator(num_classes=num_classes).to(device)

    if len(glob.glob(args.model_path+'G_*')) == 0:
        netG.to(device)
    else:
        state_dict = torch.load(max(glob.iglob(args.model_path+'G_*'), key=os.path.getmtime))
        print('[*] load_{}'.format(max(glob.iglob(args.model_path+'G_*'), key=os.path.getmtime)))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`nvidia
            new_state_dict[name] = v

        model_dict = netG.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        netG.load_state_dict(model_dict)
        netG.to(device)
    
    if len(glob.glob(args.model_path+'D*')) == 0:
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
    '''
    if len(glob.glob(args.model_path+'OD*')) == 0:
        netOD.to(device)
    else:
        state_dict = torch.load(max(glob.iglob(args.model_path+'OD*'), key=os.path.getmtime))
        print('[*] load_{}'.format(max(glob.iglob(args.model_path+'OD*'), key=os.path.getmtime)))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`nvidia
            new_state_dict[name] = v

        model_dict = netOD.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        netOD.load_state_dict(model_dict)
        netOD.to(device)
    '''
    parallel = True
    if parallel:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        # netOD = nn.DataParallel(netOD)

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
    '''
    o_dis_parameters = []
    for key, value in dict(netOD.named_parameters()).items():
        if value.requires_grad:
            o_dis_parameters += [{'params': [value], 'lr': d_lr}]
    od_optimizer = torch.optim.Adam(o_dis_parameters, betas=(0, 0.999))
    '''
    # make dirs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.makedirs(os.path.join(args.out_path, 'model/'))
    writer = SummaryWriter(os.path.join(args.out_path, 'log'))

    logger = setup_logger("OCGAN", args.out_path, 0)
    logger.info(netG)
    logger.info(netD)
    # logger.info(netOD)

    start_time = time.time()
    l1_loss = nn.DataParallel(nn.L1Loss())
    ce_loss = nn.DataParallel(nn.CrossEntropyLoss())
    vgg_loss = nn.DataParallel(MyVGGLoss())
    inceptionv3 = nn.DataParallel(Inceptionv3SpatialFeature())
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    total_idx = len(dataloader.dataset) / args.batch_size
    print('[*] total_idx: ', total_idx)
    zeros = torch.zeros([1], dtype=torch.int)
    ones = torch.ones([1], dtype=torch.int)
    o_254 = 127 * ones
    o_255 = 128 * ones

    for epoch in range(args.start_epoch, args.total_epoch):
        netG.train()
        #netOD.train()
        netD.train()
        with torch.autograd.detect_anomaly():
            for idx, data in enumerate(tqdm(dataloader)):
                real_images, label, bbox, triples = data
                real_images, label, bbox, triples = real_images.to(device), label.long().to(device), bbox.float(), triples.cuda()
                randomly_selected = torch.randint(30, (1,))  # torch.randint(bbox.size(1), (1,))
                selected_bbox = torch.unsqueeze(bbox[:,randomly_selected.item(), :], 1)
                mask = bil.bbox2_mask(selected_bbox, real_images, is_train=True)  # 1 for mask, 0 for non-mask
                mask = mask.cuda()
                masked_images = real_images * (1.-mask) # + 0.5 * mask
                b, o, h, w = real_images.size(0), label.size(1), real_images.size(2), real_images.size(3)
                x1, y1, x2, y2 = (w * bbox[:,:,0]).int(), (h * bbox[:,:,1]).int(), (w * (bbox[:,:,2]+bbox[:,:,0])).int(), (h * (bbox[:,:,3]+bbox[:,:,1])).int()  # [B, o, 1]
                x1, y1, x2, y2 = torch.min(torch.max(zeros, x1), o_254), torch.min(torch.max(zeros, y1), o_254), torch.min(torch.max(x1+1, x2), o_255), torch.min(torch.max(y1+1, y2), o_255)  # [B, o, 1]
                sel_y1, sel_y2, sel_x1, sel_x2 = y1[:, randomly_selected.item()].unsqueeze(-1), y2[:, randomly_selected.item()].unsqueeze(-1), x1[:, randomly_selected.item()].unsqueeze(-1), x2[:, randomly_selected.item()].unsqueeze(-1)
                # obj_images = crop_resize_objects(real_images, sel_y1, sel_y2, sel_x1, sel_x2).squeeze()
                # [B, o, 3, 32, 32]

                # update D network
                netD.zero_grad()
                d_out_real0, d_out_real1 = netD(real_images)
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real0).mean() + torch.nn.ReLU()(1.0 - d_out_real1).mean()

                z = torch.randn(real_images.size(0), label.size(1), z_dim).to(device)
                spatial, avg, _ = inceptionv3(F.interpolate(real_images, (299, 299)))
                content = {'image_contents': masked_images, 'mask': mask, 'label': label, 'bbox': bbox, 'triples': triples, 'z': z, 'batch_randomly_selected': randomly_selected.unsqueeze(0).expand(b, -1), 'spatial': spatial, 'avg': avg}
                fake_images_dict = netG(content)
                fake_images = fake_images_dict['image_contents'] * mask + (1.-mask) * masked_images
                # fake_obj_images = crop_resize_objects(fake_images, sel_y1, sel_y2, sel_x1, sel_x2).squeeze()
                d_out_fake0, d_out_fake1 = netD(fake_images.detach())
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake0).mean() + torch.nn.ReLU()(1.0 + d_out_fake1).mean()

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                '''
                netOD.zero_grad()
                od_loss_real, odloss_real_cls = netOD(obj_images, label, triples, randomly_selected.unsqueeze(0).expand(b, -1))
                d_loss_o_real = torch.nn.ReLU()(1.0 - od_loss_real).mean()
                od_loss_fake, odloss_fake_cls = netOD(fake_obj_images.detach(), label, triples, randomly_selected.unsqueeze(0).expand(b, -1))
                d_loss_o_fake = torch.nn.ReLU()(1.0 + od_loss_fake).mean()

                od_cls_real = ce_loss(odloss_real_cls, label[:,randomly_selected.item()]).mean()
                od_loss = d_loss_o_real + d_loss_o_fake + od_cls_real
                od_loss.backward()
                od_optimizer.step()
                '''
                # update G network
                netG.zero_grad()
                g_out_fake0, g_out_fake1 = netD(fake_images)
                g_loss_fake = - g_out_fake0.mean() - g_out_fake1.mean()

                # g_out_o_fake, odloss_fake_cls = netOD(fake_obj_images.view(b, 3, 32, 32), label, triples, randomly_selected.unsqueeze(0).expand(b, -1))
                # g_loss_o_fake = - g_out_o_fake.mean()

                pixel_loss = l1_loss(fake_images, real_images).mean()  # torch.nan_to_num(l1_loss(fake_images, real_images).mean())
                # o_pixel_loss = l1_loss(obj_images, fake_obj_images).mean()
                perc_loss, gram_loss = vgg_loss(real_images, fake_images)  # torch.nan_to_num(perceptual_loss(real_images, fake_images).mean())
                perc_loss, gram_loss = perc_loss.mean(), gram_loss.mean()
                tv_loss = total_variation_loss(fake_images).mean()
                # od_cls_fake = ce_loss(odloss_fake_cls, label[:,randomly_selected.item()]).mean()
                sgsm = sgsm_loss(fake_images_dict['obj_embeddings'], fake_images_dict['spatial'], fake_images_dict['avg'], cos)
                # print(fake_images_dict['obj_embeddings'][0])
                g_loss = g_loss_fake + 1. * perc_loss + 1. * sgsm + 1. * pixel_loss + 1. * gram_loss + 1. * tv_loss  # + g_loss_o_fake + od_cls_fake
                g_loss.backward()
                g_optimizer.step()

                if idx % (int(total_idx/9)) == 1:
                # if idx % 1 == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    logger.info("Epoch: [{}], Time Elapsed: [{}]".format(epoch, elapsed))
                    logger.info("dloss, real, fake: {}, {}, {}".format(d_loss.item(), d_loss_real.item(), d_loss_fake.item()))
                    logger.info("gloss, fake, perc, sgsm: {}, {}, {}, {}".format(g_loss.item(), g_loss_fake.item(), perc_loss.item(), sgsm.item(),))
                    # logger.info("odloss, real, fake, fake, real_cls, fake_cls: {}, {}, {}, {}, {}, {}".format(od_loss.item(), d_loss_o_real.item(), d_loss_o_fake.item(), g_loss_o_fake.item(), od_cls_real.item(), od_cls_fake.item(),))
                    writer.add_image("real images", make_grid(real_images.cpu().data * 0.5 + 0.5, nrow=int(args.batch_size/4)), epoch * len(dataloader) + idx + 1)
                    '''
                    writer.add_image("fake obj images", make_grid(fake_obj_images.view(b*o,3, 32, 32).cpu().data * 0.5 + 0.5, nrow=int(args.batch_size/4)), epoch * len(dataloader) + idx + 1)
                    '''
                    writer.add_image("fake images", make_grid(fake_images.cpu().data * 0.5 + 0.5, nrow=int(args.batch_size/4)), epoch * len(dataloader) + idx + 1)
                    writer.add_image("masked images", make_grid(masked_images.cpu().data * 0.5 + 0.5, nrow=int(args.batch_size/4)), epoch * len(dataloader) + idx + 1)
                    writer.add_image("masks", make_grid(mask.cpu().data * 0.5 + 0.5, nrow=int(args.batch_size/4)), epoch * len(dataloader) + idx + 1)

                    writer.add_scalars("D_loss", {"real": d_loss_real.item(),
                                        "fake": d_loss_fake.item(),
                                        "loss": d_loss.item(),
                                        }, epoch * len(dataloader) + idx + 1)
                    '''
                    writer.add_scalars("OD_loss", {"real": d_loss_o_real.item(),
                                                    'fake': d_loss_o_fake.item(),
                                                    'loss': od_loss.item(),}, epoch * len(dataloader) + idx + 1)
                    '''
                    writer.add_scalars("G_loss", {"fake": g_loss_fake.item(),
                                                    # "o_fake": g_loss_o_fake.item(),
                                                    "perc": perc_loss.item(),
                                                    "sgsm": sgsm.item(),
                                                    "loss": g_loss.item()}, epoch * len(dataloader) + idx + 1)

                    # writer.add_text('t/json', str(scene_graphs), epoch*len(dataloader) + idx + 1)

        # save model
        if (epoch) % 2 == 0:
            torch.save(netG.state_dict(), os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch + 1)))
            torch.save(netD.state_dict(), os.path.join(args.out_path, 'model/', 'D_%d.pth' % (epoch + 1)))
            # torch.save(netOD.state_dict(), os.path.join(args.out_path, 'model/', 'OD_%d.pth' % (epoch + 1)))



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

# python train_hvita_inpaint_triple.py --dataset coco --out_path september/hvita_triple/ --batch_size 128 --model_path september/hvita_triple/vg/128/model/ --model_path september/hvita_triple/vg/128/model/ --start_epoch 200 --total_epoch 400
# python train_hvita_inpaint.py --dataset vg --out_path hvita_outputs/ --model_path ./hvita_outputs/vg/128/model/