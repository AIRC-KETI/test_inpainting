from tkinter.messagebox import NO
from xml.etree.ElementPath import prepare_descendant
import torch
import torch.nn as nn
import torch.nn.functional as F
#from .roi_layers import ROIAlign, ROIPool
from torchvision.ops import RoIAlign, RoIPool
from utils.util import *
from utils.bilinear import *


class ResnetDiscriminator128_app_triples(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64, pred_classes=46):
        super(ResnetDiscriminator128_app_triples, self).__init__()
        self.num_classes = num_classes
        self.pred_classes = pred_classes
        self.block1 = OptimizedBlock(input_dim, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block6 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.roi_align_s = RoIAlign((8, 8), 1.0 / 4.0, int(0))
        self.roi_align_l = RoIAlign((8, 8), 1.0 / 8.0, int(0))
        # object discriminator
        self.block_obj3 = ResBlock(ch * 2, ch * 4, downsample=False)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))
        # apperance discriminator
        self.app_conv = ResBlock(ch * 8, ch * 8, downsample=False)
        self.l_y_app = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 8))
        self.app = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        # pred discriminator
        self.pred_conv = ResBlock(ch * 8, ch * 8, downsample=False)
        self.l_pred = nn.utils.spectral_norm(nn.Embedding(pred_classes, ch * 16))

    def forward(self, x, y=None, bbox=None, triples=None):
        b = bbox.size(0)
        x = self.block1(x)  # 64x64
        x1 = self.block2(x)  # 32x32
        x2 = self.block3(x1)  # 16x16
        x = self.block4(x2)  # 8x8
        x = self.block5(x)  # 4x4        
        x = self.block6(x)  # 2x2
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l7(x)

        # obj path
        # seperate different path
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 64) * ((bbox[:, 4] - bbox[:, 2]) < 64)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]

        y_l, y_s = y[~s_idx], y[s_idx]

        obj_feat_s = self.block_obj3(x1)
        obj_feat_s = self.block_obj4(obj_feat_s)

        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj4(x2)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)
        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        # apperance
        app_feat = self.app_conv(obj_feat)
        app_feat = self.activation(app_feat)

        s1, s2, s3, s4 = app_feat.size()
        app_feat = app_feat.view(s1, s2, s3 * s4)
        app_gram = torch.bmm(app_feat, app_feat.permute(0, 2, 1)) / s2

        app_y = self.l_y_app(y).unsqueeze(1).expand(s1, s2, s2)
        app_all = torch.cat([app_gram, app_y], dim=-1)
        out_app = self.app(app_all).sum(1) / s2

        # pred
        s, p, o = triples.chunk(3, dim=-1)  # [B,# of triples, 1]
        s, p, o = [x.squeeze(-1) for x in [s, p, o]]  # [B, # of triples]
        s = s.view(-1)
        o = o.view(-1)
        pred_feat = torch.sum(self.activation(self.pred_conv(obj_feat)), dim=(2, 3))
        pred = self.l_pred(p.view(-1))
        s_feat = torch.gather(pred_feat, -1, torch.unsqueeze(s, -1).expand(-1, pred_feat.size(-1)))
        o_feat = torch.gather(pred_feat, -1, torch.unsqueeze(o, -1).expand(-1, pred_feat.size(-1)))
        out_pred = torch.cat((s_feat, o_feat), -1)
        out_pred = torch.sum(out_pred * pred, dim=1, keepdim=True)

        # original one for single instance
        obj_feat = self.block_obj5(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        
        out_obj = self.l_obj(obj_feat)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)
        return out_im, out_obj, out_app, out_pred


class CombineDiscriminator128_app(nn.Module):
    def __init__(self, num_classes=81, input_dim=3, pred_classes=46):
        super(CombineDiscriminator128_app, self).__init__()
        self.obD = ResnetDiscriminator128_app_triples(num_classes=num_classes, input_dim=input_dim, pred_classes=pred_classes)
    
    def forward(self, images, bbox, label, triples=None, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()  # [0, 1, 2, ..., B-1] with [B, obj, 1]

        bbox = bbox.cuda()
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]  # [w -> x_end]
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]  # [h -> y_end]
        bbox = bbox * images.size(2)  # [per -> pix]
        bbox = torch.cat((idx, bbox.float()), dim=2)  # [B, obj, 5]
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        d_out_img, d_out_obj, out_app, out_pred = self.obD(images, label, bbox, triples)
        return d_out_img, d_out_obj, out_app, out_pred


class OptimizedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(OptimizedBlock, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()
        self.downsample = downsample

    def forward(self, in_feat):
        x = in_feat
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x + self.shortcut(in_feat)

    def shortcut(self, x):
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return self.c_sc(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.activation = nn.ReLU()
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)

    def residual(self, in_feat):
        x = in_feat
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, in_feat):
        return self.residual(in_feat) + self.shortcut(in_feat)

def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv