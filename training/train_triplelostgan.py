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
from model.generator_my import *
from model.discriminator_my import *
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
import stylegan2
import os
import random

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
        data = VgSceneGraphDataset(vocab_json='./datasets/vg/vocab.json', h5_path='./datasets/vg/train.h5',
                                   image_dir='./datasets/vg/images/',
                                   image_size=(img_size, img_size), max_objects=30, left_right_flip=True)
    return data


def main(args):
    # parameters
    img_size = args.img_size
    z_dim = 128
    lamb_obj = 1.0
    lamb_app = 1.0
    lamb_img = 1.0
    lamb_pred = 1.0
    pred_classes = 7 if args.dataset == 'coco' else 46
    # pred_classes = 7 if args.dataset == 'coco' else 7
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
        drop_last=True, shuffle=True, num_workers=num_workers)

    # Load model
    device = torch.device('cuda')
    netG = TripleGenerator(num_classes=num_classes, pred_classes=pred_classes, output_dim=3).to(device)
    netD = CombineDiscriminator128_app(num_classes=num_classes, input_dim=4, pred_classes=pred_classes).to(device)

    if len(glob.glob(args.model_path+'G*')) > 0:
        current_g = max(glob.iglob(args.model_path+'G*'), key=os.path.getmtime)
        state_dict = torch.load(current_g)
        print('[*] load_{}'.format(current_g))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`nvidia
            new_state_dict[name] = v

        model_dict = netG.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        netG.load_state_dict(model_dict)
    else:
        print('no load')
    netG.to(device)

    if len(glob.glob(args.model_path+'D*')) > 0:
        current_d = max(glob.iglob(args.model_path+'D*'), key=os.path.getmtime)
        state_dict = torch.load(current_d)
        print('[*] load_{}'.format(current_d))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`nvidia
            new_state_dict[name] = v

        model_dict = netD.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        netD.load_state_dict(model_dict)
    else:
        print('no load')

    netD.to(device)
    parallel = True
    if parallel:
        netG = DataParallelWithCallback(netG)
        netD = nn.DataParallel(netD)

    # set lr
    g_lr, d_lr = args.g_lr, args.d_lr
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

    # make dirs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.makedirs(os.path.join(args.out_path, 'model/'))
    writer = SummaryWriter(os.path.join(args.out_path, 'log'))

    logger = setup_logger("STALostGAN", args.out_path, 0)
    logger.info(netG)
    logger.info(netD)

    start_time = time.time()
    vgg_loss = VGGLoss()
    vgg_loss = nn.DataParallel(vgg_loss)
    l1_loss = nn.DataParallel(nn.L1Loss())
    '''
    json_file = open('/media/LostGANs/id_name.json', 'r', encoding='utf-8')
    json_ = json.load(json_file)
    json_file.close()
    vocab = [
            '__in_image__',
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]
    '''
    for epoch in range(args.start_epoch, args.total_epoch):
        netG.train()
        netD.train()

        for idx, data in enumerate(tqdm(dataloader)):
            real_images, label, bbox, triples = data
            real_images, label, bbox, triples = real_images.to(device), label.long().to(device).unsqueeze(-1), bbox.float(), triples.cuda()
            s, p, o = triples.chunk(3, dim=-1)  # [B,# of triples, 1]
            s, p, o = [x.squeeze(-1) for x in [s, p, o]]  # [B, # of triples]
            randomly_selected = torch.randint(bbox.size(1), (1,))
            selected_bbox = torch.unsqueeze(bbox[:,randomly_selected.item(), :], 1)
            mask = bil.bbox2_mask(selected_bbox, real_images)  # 1 for mask, 0 for non-mask
            mask = mask.cuda()
            masked_images = real_images * (1.-mask)
            con_masked_images = torch.cat((masked_images, mask), 1)
            b, obj = triples.size(0), triples.size(1)
            # update D network
            netD.zero_grad()
            real_images, label = real_images.to(device), label.long().to(device)
            d_out_real, d_out_robj, d_out_robj_app, d_out_rpred = netD(torch.cat((real_images, mask), 1), bbox, label, triples)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
            d_loss_robj_app = torch.nn.ReLU()(1.0 - d_out_robj_app).mean()
            d_loss_rpred = torch.nn.ReLU()(1.0 - d_out_rpred).mean()

            z = torch.randn(real_images.size(0), num_obj, z_dim).to(device)
            f_bbox = bbox
            f_con_masked_images = con_masked_images
            is_inpaint = True
            if random.random() >=0.5:
                # rand = 0.5 * torch.rand(b, obj, 4)
                # f_bbox = torch.cat((rand[:,:,0:2], 1.-rand[:,:,2:-1]-rand[:,:,0:2]),-1)
                f_bbox = torch.Tensor([[[0.,0.,1.,1.,]]]).expand(b, obj, -1)
            if random.random() >= 0.5:
                is_inpaint=False
                f_con_masked_images = torch.cat((torch.zeros(b, 3, 128, 128), torch.ones(b, 1, 128, 128)), 1)

            fake_images = netG(z, f_bbox, y=label.squeeze(dim=-1), triples=triples, masked_images=f_con_masked_images)
            if is_inpaint:
                fake_images = fake_images * mask + real_images * (1.-mask)
            d_out_fake, d_out_fobj, d_out_fobj_app, d_out_fpred = netD(torch.cat((fake_images.detach(), mask), 1), bbox, label, triples)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()
            d_loss_fobj_app = torch.nn.ReLU()(1.0 + d_out_fobj_app).mean()
            d_loss_fpred = torch.nn.ReLU()(1.0 + d_out_fpred).mean()
            d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake) + lamb_app * (d_loss_robj_app + d_loss_fobj_app) + lamb_pred * (d_loss_rpred + d_loss_fpred)
            d_loss.backward()
            d_optimizer.step()

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                g_out_fake, g_out_obj, g_out_obj_app, g_out_pred = netD(torch.cat((fake_images, mask), 1), bbox, label, triples)
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()
                g_loss_obj_app = - g_out_obj_app.mean()
                g_loss_pred = - g_out_pred.mean()

                pixel_loss = l1_loss(fake_images, real_images).mean()
                feat_loss = vgg_loss(fake_images, real_images).mean()

                g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img + pixel_loss + feat_loss + lamb_app * g_loss_obj_app + lamb_pred * g_loss_pred
                g_loss.backward()
                g_optimizer.step()

            if idx % args.writer_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                logger.info("Time Elapsed: [{}]".format(elapsed))
                logger.info("Step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_out_fake: {:.4f} ".format(epoch + 1,
                                                                                                              idx + 1,
                                                                                                              d_loss_real.item(),
                                                                                                              d_loss_fake.item(),
                                                                                                              g_loss_fake.item()))
                logger.info("             d_obj_real: {:.4f}, d_obj_fake: {:.4f}, g_obj_fake: {:.4f} ".format(
                    d_loss_robj.item(),
                    d_loss_fobj.item(),
                    g_loss_obj.item()))
                logger.info("             d_obj_real_app: {:.4f}, d_obj_fake_app: {:.4f}, g_obj_fake_app: {:.4f} ".format(
                    d_loss_robj_app.item(),
                    d_loss_fobj_app.item(),
                    g_loss_obj_app.item()))
                logger.info("             d_pred_real: {:.4f}, d_pred_fake: {:.4f}, g_pred: {:.4f} ".format(
                    d_loss_rpred.item(),
                    d_loss_fpred.item(),
                    g_loss_pred.item()))
                logger.info("             pixel_loss: {:.4f}, feat_loss: {:.4f}".format(pixel_loss.item(), feat_loss.item()))

                writer.add_image("real images", make_grid(real_images.cpu().data * 0.5 + 0.5, nrow=4), epoch * len(dataloader) + idx + 1)
                writer.add_image("fake images", make_grid(fake_images.cpu().data * 0.5 + 0.5, nrow=4), epoch * len(dataloader) + idx + 1)
                writer.add_image("masked images", make_grid(masked_images.cpu().data * 0.5 + 0.5, nrow=4), epoch * len(dataloader) + idx + 1)
                writer.add_image("masks", make_grid(mask.cpu().data * 0.5 + 0.5, nrow=4), epoch * len(dataloader) + idx + 1)

                writer.add_scalars("D_loss_real", {"real": d_loss_real.item(),
                                                       "robj": d_loss_robj.item(),
                                                       "robj_app": d_loss_robj_app.item(),
                                                       "r_pred": d_loss_rpred.item(),
                                                       "loss": d_loss.item()}, epoch*len(dataloader) + idx + 1)
                writer.add_scalars("D_loss_fake", {"fake": d_loss_fake.item(),
                                                    "fobj": d_loss_fobj.item(),
                                                    "fobj_app": d_loss_fobj_app.item(),
                                                    "f_pred": d_loss_rpred.item(),
                                                    }, epoch*len(dataloader) + idx + 1)
                writer.add_scalars("G_loss", {"fake": g_loss_fake.item(),
                                                "obj": g_loss_obj.item(),
                                                "obj_app": g_loss_obj_app.item(),
                                                "pred": g_loss_pred.item(),
                                                "loss": g_loss.item()}, epoch*len(dataloader) + idx + 1)
                # writer.add_text('t/json', str(scene_graphs), epoch*len(dataloader) + idx + 1)

        # save model
        # if (epoch + 1) % 5 == 0:
        torch.save(netG.state_dict(), os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch + 1)))
        torch.save(netD.state_dict(), os.path.join(args.out_path, 'model/', 'D_%d.pth' % (epoch + 1)))



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
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for generator')
    parser.add_argument('--out_path', type=str, default='./outputs/tmp/our_d/',
                        help='path to output files')
    parser.add_argument('--model_path', type=str, default='./my_outputs/model/',
                        help='path to output files')
    parser.add_argument('--img_size', type=str, default=128,
                        help='generated image size')
    parser.add_argument('--writer_step', type=int, default=500,
                        help='generated image size')
    args = parser.parse_args()
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main(args)

# python train_triplelostgan.py --dataset coco --out_path ./triplelostgan_v2  --model_path ./triplelostgan_v1/coco/128/model/  --start_epoch 98
# python train_triplelostgan.py --dataset vg --out_path triplelostgan_v1
