import torch
import torch.nn as nn
import torch.nn.functional as F
#from .roi_layers import ROIAlign, ROIPool
from torchvision.ops import RoIAlign, RoIPool
from utils.util import *
from utils.bilinear import *


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


class ResnetDiscriminator128(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator128, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(3, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block6 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.roi_align_s = RoIAlign((8, 8), 1.0 / 4.0, int(0))
        self.roi_align_l = RoIAlign((8, 8), 1.0 / 8.0, int(0))

        self.block_obj3 = ResBlock(ch * 2, ch * 4, downsample=False)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))

    def forward(self, x, y=None, bbox=None):
        b = bbox.size(0)
        # 128x128
        x = self.block1(x)
        # 64x64
        x1 = self.block2(x)
        # 32x32
        x2 = self.block3(x1)
        # 16x16
        x = self.block4(x2)
        # 8x8
        x = self.block5(x)
        # 4x4
        x = self.block6(x)
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
        # print(obj_feat_s.shape)
        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj4(x2)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        obj_feat = self.block_obj5(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        out_obj = self.l_obj(obj_feat)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        return out_im, out_obj


class HVITADiscriminator128(nn.Module):
    def __init__(self, input_dim=3, ch=64):
        super(HVITADiscriminator128, self).__init__()
        self.block1 = OptimizedBlock(3, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block6 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

    def forward(self, x, y=None):
        b = x.size(0)
        x = self.block1(x)
        x1 = self.block2(x)
        x2 = self.block3(x1)
        x = self.block4(x2)
        x = self.block5(x)
        x = self.block6(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l7(x)
        return out_im


class HVITADiscriminator64(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(HVITADiscriminator64, self).__init__()
        self.block1 = OptimizedBlock(3, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16 + 20, 1))
        self.activation = nn.ReLU()
        self.label_embedding = nn.Embedding(num_classes, 20)

    def forward(self, x, y=None):
        y = self.label_embedding(y)
        b = x.size(0)
        x = self.block1(x)
        x1 = self.block2(x)
        x2 = self.block3(x1)
        x = self.block4(x2)
        x = self.block5(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        x = torch.cat((x, y), dim=-1)
        out_im = self.l7(x)
        return out_im


class HVITADiscriminator64_triple(nn.Module):
    def __init__(self, num_classes=179, input_dim=3, pred_classes=46, ch=64):
        super(HVITADiscriminator64_triple, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, 180)
        self.pred_embedding = nn.Embedding(pred_classes, 180)
        self.fc = nn.utils.spectral_norm(nn.Linear(11340, 2048))

        self.block1 = OptimizedBlock(3, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16 + 2048, 1))
        self.activation = nn.ReLU()

    def forward(self, x, y=None, triples=None, randomly_selected=None):
        b, obj = y.size(0), y.size(1)
        s, p, o = triples.chunk(3, dim=-1)  # [B,# of triples, 1]
        s, p, o = [x.squeeze(-1) for x in [s, p, o]]  # [B, # of triples]
        s_emb = torch.gather(y, -1, s)
        sel_p = p.unsqueeze(1).expand(-1, obj, -1) * torch.eq(s_emb.unsqueeze(1).expand(-1, obj, -1), y.unsqueeze(-1).expand(-1, -1, s.size(-1))).long()  # [B, num_o, num_t]
        sel_o = o.unsqueeze(1).expand(-1, obj, -1) * torch.eq(s_emb.unsqueeze(1).expand(-1, obj, -1), y.unsqueeze(-1).expand(-1, -1, s.size(-1))).long()  # [B, num_o, num_t]
        obj_triple = torch.cat((y.unsqueeze(-1), sel_o), -1)  # [B, num_o, # of triples + 1]
        label_embedding = self.label_embedding(obj_triple)  # [B, num_o, # of triples + 1, 180]
        pred_embedding = self.pred_embedding(sel_p)  # [B, num_o, # of triples, 180]
        obj_triple_embedding = torch.cat((label_embedding, pred_embedding), 2).view(b, obj, -1)  # [B, num_o, (2 * (# of triples) + 1) * 180]
        obj_triple_embedding = obj_triple_embedding[:,randomly_selected[0]].squeeze()
        # print(obj_triple_embedding.shape)
        # obj_triple_embedding = obj_triple_embedding.view(b * obj, -1)  # [B, num_o, (2 * (# of triples) + 1) * 180 + feat]
        # object generator
        y = self.fc(obj_triple_embedding)
        x = self.block1(x)
        x1 = self.block2(x)
        x2 = self.block3(x1)
        x = self.block4(x2)
        x = self.block5(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        x = torch.cat((x, y), dim=-1)
        out_im = self.l7(x)
        return out_im


class ResnetDiscriminator128_segment(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator128_segment, self).__init__()
        self.ch = ch
        self.conv_1 = conv2d(3+8+1, self.ch, stride=2)  # 8 for coco
        self.conv_2 = conv2d(self.ch, 2 * self.ch, stride=2)
        self.conv_3 = conv2d(2 * self.ch, 4 * self.ch, stride=2)
        self.conv_4 = conv2d(4 * self.ch, 8 * self.ch, stride=2)
        self.conv_5 = conv2d(8 * self.ch, 1, stride=1)
    def forward(self, x, y=None, bbox=None):  # kernel_size=3, stride=1, pad=1, spectral_norm=True):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        
        return x


class ResnetDiscriminator128_edge(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator128_edge, self).__init__()
        self.ch = ch
        self.conv_1 = conv2d(3+1+1, self.ch, stride=2)
        self.conv_2 = conv2d(self.ch, 2 * self.ch, stride=2)
        self.conv_3 = conv2d(2 * self.ch, 4 * self.ch, stride=2)
        self.conv_4 = conv2d(4 * self.ch, 8 * self.ch, stride=2)
        self.conv_5 = conv2d(8 * self.ch, 1, stride=1)
    def forward(self, x, y=None, bbox=None):  # kernel_size=3, stride=1, pad=1, spectral_norm=True):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        
        return x


class ResnetDiscriminator128_triple(nn.Module):
    def __init__(self, num_classes=81, pred_classes=46, input_dim=3, ch=64):
        super(ResnetDiscriminator128_triple, self).__init__()
        self.ch = ch
        self.conv_1 = conv2d(3+1, self.ch, kernel_size=5, pad=2, stride=2)
        self.conv_2 = conv2d(self.ch, 2 * self.ch, kernel_size=5, pad=2, stride=2)
        self.conv_3 = conv2d(2 * self.ch, 4 * self.ch, kernel_size=5, pad=2, stride=2)
        self.conv_4 = conv2d(4 * self.ch, 8 * self.ch, kernel_size=5, pad=2, stride=2)
        self.conv_5 = conv2d(8 * self.ch, 8 * self.ch, kernel_size=5, pad=2, stride=2)
        self.conv_6 = conv2d(17252, 8 * self.ch, kernel_size=5, pad=2, stride=2)
        self.activation = nn.ReLU()
        self.label_embedding = nn.Embedding(num_classes, 180)
        self.pred_embedding = nn.Embedding(pred_classes, 180)

    def forward(self, x, y=None, triples=None):  # kernel_size=3, stride=1, pad=1, spectral_norm=True):
        features = []
        b = x.size(0)
        x = self.conv_1(x)
        features.append(x)
        x = self.conv_2(self.activation(x))
        features.append(x)
        x = self.conv_3(self.activation(x))
        features.append(x)
        x = self.conv_4(self.activation(x))
        features.append(x)
        x = self.conv_5(self.activation(x))
        features.append(x)

        s, p, o = triples.chunk(3, dim=-1)  # [B,# of triples, 1]
        s, p, o = [x.squeeze(-1) for x in [s, p, o]]  # [B, # of triples]
        for i in range(len(s)):
            if i is 0:
                small_s = [[y[i][x].tolist() for x in s[i].tolist()]]
                small_o = [[y[i][x].tolist() for x in o[i].tolist()]]
                # print(len(small_s), y[0].shape, s[i].shape)
            else:
                small_s.append([y[i][x].tolist() for x in s[i].tolist()])
                small_o.append([y[i][x].tolist() for x in s[i].tolist()])
        
        s, o = torch.tensor(small_s), torch.tensor(small_o)  # [32, 8]
        so = torch.cat((s, o), dim=0).cuda()
        so = torch.squeeze(so)
        label_embedding = self.label_embedding(so)  # [2B, # of triples, 180]
        pred_embedding = self.pred_embedding(p)  # [B, # of triples, 180]
        s_label_embedding, o_label_embedding = label_embedding.chunk(2, dim=0)  # [B, # of triples, 180]
        # print(s_label_embedding.size(), o_label_embedding.size(), pred_embedding.size())
        label_embedding = torch.cat((s_label_embedding, o_label_embedding, pred_embedding), -1)
        label_embedding = label_embedding.view(b, -1, 1, 1)
        # print(label_embedding.size())
        label_embedding = label_embedding.repeat(1, 1, 4, 4)
        # print(x.size(), label_embedding.size())
        z = torch.cat((x, label_embedding), 1)
        z = self.conv_6(self.activation(z))
        features.append(z)
        hh, ww = z.size(2), z.size(3)
        x = F.interpolate(x, size=(hh, ww), mode='bilinear')
        z = x  + z
        features.append(z)
        return z, features


class ResnetDiscriminator128_app(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator128_app, self).__init__()
        self.num_classes = num_classes

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

        self.block_obj3 = ResBlock(ch * 2, ch * 4, downsample=False)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))
        # apperance discriminator
        self.app_conv = ResBlock(ch * 8, ch * 8, downsample=False)
        self.l_y_app = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 8))
        self.app = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))

    def forward(self, x, y=None, bbox=None):
        features = []
        b = bbox.size(0)
        # 128x128
        x = self.block1(x)
        features.append(x)
        # 64x64
        x1 = self.block2(x)
        features.append(x)
        # 32x32
        x2 = self.block3(x1)
        features.append(x2)
        # 16x16
        x = self.block4(x2)
        features.append(x)
        # 8x8
        x = self.block5(x)
        features.append(x)
        # 4x4
        x = self.block6(x)
        features.append(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l7(x)
        features.append(out_im)
        # obj path
        # seperate different path
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 64) * ((bbox[:, 4] - bbox[:, 2]) < 64)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]

        y_l, y_s = y[~s_idx], y[s_idx]

        obj_feat_s = self.block_obj3(x1)
        features.append(obj_feat_s)
        obj_feat_s = self.block_obj4(obj_feat_s)
        features.append(obj_feat_s)

        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)
        features.append(obj_feat_s)

        obj_feat_l = self.block_obj4(x2)
        features.append(obj_feat_l)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)
        features.append(obj_feat_l)
        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        # apperance
        app_feat = self.app_conv(obj_feat)
        features.append(app_feat)
        app_feat = self.activation(app_feat)
        features.append(app_feat)

        s1, s2, s3, s4 = app_feat.size()
        app_feat = app_feat.view(s1, s2, s3 * s4)
        app_gram = torch.bmm(app_feat, app_feat.permute(0, 2, 1)) / s2

        app_y = self.l_y_app(y).unsqueeze(1).expand(s1, s2, s2)
        features.append(app_y)
        app_all = torch.cat([app_gram, app_y], dim=-1)
        out_app = self.app(app_all).sum(1) / s2
        features.append(out_app)

        # original one for single instance
        obj_feat = self.block_obj5(obj_feat)
        features.append(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        # print(obj_feat.shape)
        out_obj = self.l_obj(obj_feat)
        features.append(out_obj)

        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)
        features.append(out_obj)
        return out_im, out_obj, out_app



class ResnetDiscriminator128_app_triples(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator128_app, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(3, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block6 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.roi_align_s = RoIAlign((8, 8), 1.0 / 4.0, int(0))
        self.roi_align_l = RoIAlign((8, 8), 1.0 / 8.0, int(0))

        self.block_obj3 = ResBlock(ch * 2, ch * 4, downsample=False)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))
        # apperance discriminator
        self.app_conv = ResBlock(ch * 8, ch * 8, downsample=False)
        self.l_y_app = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 8))
        self.app = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))

    def forward(self, x, y=None, bbox=None, triples=None):
        b = bbox.size(0)
        # 128x128
        x = self.block1(x)
        # 64x64
        x1 = self.block2(x)
        # 32x32
        x2 = self.block3(x1)
        # 16x16
        x = self.block4(x2)
        # 8x8
        x = self.block5(x)
        # 4x4
        x = self.block6(x)
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

        # original one for single instance
        obj_feat = self.block_obj5(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        # print(obj_feat.shape)
        out_obj = self.l_obj(obj_feat)

        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        return out_im, out_obj, out_app


class ResnetDiscriminator64(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator64, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(input_dim, ch, downsample=False)
        self.block2 = ResBlock(ch, ch * 2, downsample=False)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_im = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        # object path
        self.roi_align = ROIAlign((8, 8), 1.0 / 2.0, 0)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 8, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 8))

        self.init_parameter()

    def forward(self, x, y=None, bbox=None):
        b = bbox.size(0)
        # 64x64
        x = self.block1(x)
        # 64x64
        x = self.block2(x)
        # 32x32
        x1 = self.block3(x)
        # 16x16
        x = self.block4(x1)
        # 8x8
        x = self.block5(x)
        x = self.activation(x)
        x = torch.mean(x, dim=(2, 3))
        out_im = self.l_im(x)

        # obj path
        obj_feat = self.roi_align(x1, bbox)
        obj_feat = self.block_obj4(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        out_obj = self.l_obj(obj_feat)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        return out_im, out_obj

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResnetDiscriminator256(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator256, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(3, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 8, downsample=True)
        self.block6 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block7 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l8 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.roi_align_s = ROIAlign((8, 8), 1.0 / 8.0, int(0))
        self.roi_align_l = ROIAlign((8, 8), 1.0 / 16.0, int(0))

        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 8, downsample=False)
        self.block_obj6 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))

    def forward(self, x, y=None, bbox=None):
        b = bbox.size(0)
        # 256x256
        x = self.block1(x)
        # 128x128
        x = self.block2(x)
        # 64x64
        x1 = self.block3(x)
        # 32x32
        x2 = self.block4(x1)
        # 16x16
        x = self.block5(x2)
        # 8x8
        x = self.block6(x)
        # 4x4
        x = self.block7(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l8(x)

        # obj path
        # seperate different path
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 128) * ((bbox[:, 4] - bbox[:, 2]) < 128)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]
        y_l, y_s = y[~s_idx], y[s_idx]

        obj_feat_s = self.block_obj4(x1)
        obj_feat_s = self.block_obj5(obj_feat_s)
        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj5(x2)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        obj_feat = self.block_obj6(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        out_obj = self.l_obj(obj_feat)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        return out_im, out_obj


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


class CombineDiscriminator256(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator256, self).__init__()
        self.obD = ResnetDiscriminator256(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj = self.obD(images, label, bbox)
        return d_out_img, d_out_obj


class CombineDiscriminator128(nn.Module):
    def __init__(self, num_classes=81, input_dim=3):
        super(CombineDiscriminator128, self).__init__()
        self.obD = ResnetDiscriminator128(num_classes=num_classes, input_dim=input_dim)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        #idx = idx.cuda()
        # print(bbox)
        bbox = bbox.cuda()
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj = self.obD(images, label, bbox)
        return d_out_img, d_out_obj


class CombineDiscriminator128_app(nn.Module):
    def __init__(self, num_classes=81, input_dim=3):
        super(CombineDiscriminator128_app, self).__init__()
        self.obD = ResnetDiscriminator128_app(num_classes=num_classes, input_dim=input_dim)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()

        bbox = bbox.cuda()
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj, out_app = self.obD(images, label, bbox)
        return d_out_img, d_out_obj, out_app


class CombineDiscriminator128_app_inpaint(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator128_app_inpaint, self).__init__()
        self.obD = ResnetDiscriminator128_app(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()

        bbox = bbox.cuda()
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj, out_app = self.obD(images, label, bbox)
        return d_out_img, d_out_obj, out_app


class CombineDiscriminator64(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator64, self).__init__()
        self.obD = ResnetDiscriminator64(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj = self.obD(images, label, bbox)
        return d_out_img, d_out_obj
