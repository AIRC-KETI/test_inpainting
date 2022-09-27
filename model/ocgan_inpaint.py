import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_module import *
from .mask_regression import *
from .sync_batchnorm import SynchronizedBatchNorm2d
import copy
import torchvision
from .simsg.graph import GraphTripleConv
from utils.util import *

BatchNorm = SynchronizedBatchNorm2d


class Generator(nn.Module):
    def __init__(self, ch=16, z_dim=128, num_classes=10, pred_classes=7, output_dim=3, emb_dim=128, num_t=31):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.pred_classes = pred_classes
        self.z_dim = z_dim
        self.obj_embedding = nn.Embedding(num_classes, emb_dim)
        self.pred_embedding = nn.Embedding(pred_classes, emb_dim)
        
        self.spatial_projection = conv2d(768, 256, kernel_size=1, pad=0)
        self.avg_projection = nn.utils.spectral_norm(nn.Linear(2048, 256), eps=1e-4)
        self.scene_graph_encoder = SceneGraphEncoder()
        self.mask_regress = MaskRegressNet(obj_feat=256, map_size=128)

        self.conv = conv2d(1024, 1024)
        self.res1 = ResBlock(1024+4, 1024, num_w=159, upsample=True, predict_mask=False)  # 4->8
        self.res2 = ResBlock(1024+4, 1024, num_w=159, upsample=False, predict_mask=False)  # 8
        self.res3 = ResBlock(1024+4, 512, num_w=159, upsample=True, predict_mask=False)  # 16
        self.res4 = ResBlock(512+4, 256, num_w=159, upsample=True, predict_mask=False)  # 32
        self.res5 = ResBlock(256+4, 128, num_w=159, upsample=True, predict_mask=False)  # 64
        self.res6 = ResBlock(128+4, 64, num_w=159, upsample=True, predict_mask=False)  # 128
        self.final = nn.Sequential(BatchNorm(64),
                                   nn.ReLU(),
                                   conv2d(64, output_dim, 3, 1, 1),
                                   nn.Tanh())

        self.init_parameter()
    
    def forward(self, content):
        z = content['z']
        bbox = content['bbox']
        y = content['label']
        triples = content['triples']
        masked_images = content['image_contents']
        mask = content['mask']
        spatial, avg = content['spatial'], content['avg']
        b, obj = y.size(0), y.size(1)
        # ===================================SGSM===================================
        y_emb = self.obj_embedding(y)  # [b, o, 128]
        s, p, o = triples.chunk(3, dim=-1)  # [B,# of triples, 1]
        s, p, o = [x.squeeze(-1) for x in [s, p, o]]  # [B, # of triples]
        p = self.pred_embedding(p)  # [b, obj, emb_dim]
        obj_embeddings = self.scene_graph_encoder(y_emb, p, triples)  # [b, o, 256]
        spatial = self.spatial_projection(spatial).view(b, 256, -1)  # [b, 256, 17*17]
        avg = self.avg_projection(avg.squeeze())  # [b, 256]
        # ===================================Conditioning===================================
        z_obj = torch.randn(b, obj, self.z_dim).cuda()
        bmask = self.mask_regress(torch.cat((y_emb, z_obj), -1), bbox)  # [b, obj, h, w]
        bbox_boundary_ = bbox_boundary(z, bbox, 128, 128)  # [b, obj, h, w]
        condition = (bmask + bbox_boundary_)/2.
        one_hot = F.one_hot(torch.arange(0, y.size(1))).expand(b, obj, -1).cuda()
        w = torch.cat((y_emb, one_hot), -1).view(b*obj, -1)  # [128 + 31]
        # ===================================Generator===================================
        z = torch.randn([y.size(0), 1024, 4, 4]).cuda()
        masked_images = torch.cat((masked_images, mask), 1)
        x = self.conv(z)

        hh, ww = x.size(2), x.size(3)
        temp_masked_images = F.interpolate(masked_images, (hh, ww))
        x = torch.cat((x, temp_masked_images), 1)
        x, _ = self.res1(x, w, condition)

        hh, ww = x.size(2), x.size(3)
        temp_masked_images = F.interpolate(masked_images, (hh, ww))
        x = torch.cat((x, temp_masked_images), 1)
        x, _ = self.res2(x, w, condition)

        hh, ww = x.size(2), x.size(3)
        temp_masked_images = F.interpolate(masked_images, (hh, ww))
        x = torch.cat((x, temp_masked_images), 1)
        x, _ = self.res3(x, w, condition)

        hh, ww = x.size(2), x.size(3)
        temp_masked_images = F.interpolate(masked_images, (hh, ww))
        x = torch.cat((x, temp_masked_images), 1)
        x, _ = self.res4(x, w, condition)

        hh, ww = x.size(2), x.size(3)
        temp_masked_images = F.interpolate(masked_images, (hh, ww))
        x = torch.cat((x, temp_masked_images), 1)
        x, _ = self.res5(x, w, condition)

        hh, ww = x.size(2), x.size(3)
        temp_masked_images = F.interpolate(masked_images, (hh, ww))
        x = torch.cat((x, temp_masked_images), 1)
        x, _ = self.res6(x, w, condition)
        x = self.final(x)
        return {'image_contents': x, 'obj_embeddings': obj_embeddings, 'spatial': spatial, 'avg': avg}

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class Discriminator(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(Discriminator, self).__init__()
        self.sub_discriminator1 = SubDiscriminator1()
        self.sub_discriminator2 = SubDiscriminator2()
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.sub_discriminator1(x)
        x2 = self.pool(x)
        x2 = self.sub_discriminator2(x2)
        return x1, x2

class SubDiscriminator1(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(SubDiscriminator1, self).__init__()
        self.activation = nn.LeakyReLU()
        self.ch = [3, 64, 128, 256, 512, 512, 512]
        self.conv1 = conv2d(self.ch[0], self.ch[1], kernel_size=4, stride=2, pad=1, spectral_norm=True)
        self.conv2 = conv2d(self.ch[1], self.ch[2], kernel_size=4, stride=2, pad=1, spectral_norm=True)
        self.conv3 = conv2d(self.ch[2], self.ch[3], kernel_size=4, stride=2, pad=1, spectral_norm=True)
        self.conv4 = conv2d(self.ch[3], self.ch[4], kernel_size=4, stride=2, pad=1, spectral_norm=True)
        self.conv5 = conv2d(self.ch[4], self.ch[5], kernel_size=4, stride=2, pad=1, spectral_norm=True)
        self.conv6 = conv2d(self.ch[5], self.ch[6], kernel_size=2, stride=2, pad=0, spectral_norm=True)

    def forward(self, x, y=None, triples=None, randomly_selected=None):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.conv6(x)
        return x  # torch.nan_to_num(x)

class SubDiscriminator2(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(SubDiscriminator2, self).__init__()
        self.activation = nn.LeakyReLU()
        self.ch = [3, 64, 128, 256, 512, 512, 512]
        self.conv1 = conv2d(self.ch[0], self.ch[1], kernel_size=4, stride=2, pad=1, spectral_norm=True)
        self.conv2 = conv2d(self.ch[1], self.ch[2], kernel_size=4, stride=2, pad=1, spectral_norm=True)
        self.conv3 = conv2d(self.ch[2], self.ch[3], kernel_size=4, stride=2, pad=1, spectral_norm=True)
        self.conv4 = conv2d(self.ch[3], self.ch[4], kernel_size=4, stride=2, pad=1, spectral_norm=True)
        self.conv5 = conv2d(self.ch[4], self.ch[5], kernel_size=4, stride=2, pad=1, spectral_norm=True)
        self.conv6 = conv2d(self.ch[5], self.ch[6], kernel_size=2, stride=2, pad=0, spectral_norm=True)

    def forward(self, x, y=None, triples=None, randomly_selected=None):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.conv6(x)
        return x  # torch.nan_to_num(x)

class SceneGraphEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super(SceneGraphEncoder, self).__init__()
        self.gcn1 = GraphTripleConv(128, 128, output_dim=256)
        self.gcn2 = GraphTripleConv(256, 256, output_dim=256)
        self.gcn3 = GraphTripleConv(256, 256, output_dim=256)
        self.gcn4 = GraphTripleConv(256, 256, output_dim=256)
        self.gcn5 = GraphTripleConv(256, 256, output_dim=256)
        self.gcn6 = GraphTripleConv(256, 256, output_dim=256)

    def forward(self, obj_vecs, pred_vecs, triples):
        b, obj = obj_vecs.size(0), obj_vecs.size(1)
        obj_vecs = obj_vecs.view(b*obj, -1)
        pred_vecs = pred_vecs.view(b*obj, -1)
        count = torch.arange(0, b*obj, obj, dtype=torch.long).unsqueeze(-1).unsqueeze(-1).cuda()
        triples = (triples+count).view(b*obj, -1)
        obj_vecs, pred_vecs = self.gcn1(obj_vecs, pred_vecs, triples)
        # obj_vecs, pred_vecs = torch.nan_to_num(obj_vecs), torch.nan_to_num(pred_vecs)
        obj_vecs, pred_vecs = self.gcn2(obj_vecs, pred_vecs, triples)
        # obj_vecs, pred_vecs = torch.nan_to_num(obj_vecs), torch.nan_to_num(pred_vecs)
        obj_vecs, pred_vecs = self.gcn3(obj_vecs, pred_vecs, triples)
        # obj_vecs, pred_vecs = torch.nan_to_num(obj_vecs), torch.nan_to_num(pred_vecs)
        obj_vecs, pred_vecs = self.gcn4(obj_vecs, pred_vecs, triples)
        # obj_vecs, pred_vecs = torch.nan_to_num(obj_vecs), torch.nan_to_num(pred_vecs)
        obj_vecs, pred_vecs = self.gcn5(obj_vecs, pred_vecs, triples)
        # obj_vecs, pred_vecs = torch.nan_to_num(obj_vecs), torch.nan_to_num(pred_vecs)
        obj_vecs, pred_vecs = self.gcn6(obj_vecs, pred_vecs, triples)
        return obj_vecs.view(b,obj,-1) # torch.nan_to_num(obj_vecs).view(b,obj,-1)

class ObjectDiscriminator(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ObjectDiscriminator, self).__init__()
        self.activation = nn.LeakyReLU()
        self.ch = [3, 64, 128, 256, 512, 512]
        self.conv1 = OptimizedBlock(3, 64, downsample=False)
        self.conv2 = DisResBlock(self.ch[1], self.ch[2], downsample=True)
        self.conv3 = DisResBlock(self.ch[2], self.ch[3], downsample=True)
        self.conv4 = DisResBlock(self.ch[3], self.ch[4], downsample=True)
        self.conv5 = DisResBlock(self.ch[4], self.ch[5], downsample=True)
        self.linear = nn.Linear(512, 512)
        self.cls = nn.Linear(512, num_classes)
    def forward(self, x, y=None, triples=None, randomly_selected=None):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = x.sum(dim=(-1, -2))
        cls = x
        x = self.linear(x)
        cls = self.cls(cls)
        return x, cls

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, upsample=False, stride=1, num_w=128, predict_mask=True, psp_module=False):
        super(ResBlock, self).__init__()
        self.upsample = upsample
        self.h_ch = h_ch if h_ch else out_ch
        self.conv1 = conv2d(in_ch, self.h_ch, ksize, stride=stride, pad=pad)
        self.conv2 = conv2d(self.h_ch, out_ch, ksize, stride=1, pad=pad)
        self.b1 = SpatialAdaptiveSynBatchNorm2d(in_ch, num_w=num_w, batchnorm_func=BatchNorm)
        self.b2 = SpatialAdaptiveSynBatchNorm2d(self.h_ch, num_w=num_w, batchnorm_func=BatchNorm)
        self.learnable_sc = in_ch != out_ch or upsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()

        self.predict_mask = predict_mask
        if self.predict_mask:
            if psp_module:
                self.conv_mask = nn.Sequential(PSPModule(out_ch, 100),
                                               nn.Conv2d(100, 184, kernel_size=1))
            else:
                self.conv_mask = nn.Sequential(nn.Conv2d(out_ch, 100, 3, 1, 1),
                                               BatchNorm(100),
                                               nn.ReLU(),
                                               nn.Conv2d(100, 184, 1, 1, 0, bias=True))

    def residual(self, in_feat, w, bbox):
        x = in_feat
        x = self.b1(x, w, bbox)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.b2(x, w, bbox)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.c_sc(x)
        return x

    def forward(self, in_feat, w, bbox):
        out_feat = self.residual(in_feat, w, bbox) + self.shortcut(in_feat)
        if self.predict_mask:
            mask = self.conv_mask(out_feat)
        else:
            mask = None
        return out_feat, mask

class DisResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(DisResBlock, self).__init__()
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

class GatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, stride=1):
        super(GatedConv, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, stride=stride, pad=pad)
        self.conv2 = conv2d(in_ch, out_ch, ksize, stride=stride, pad=pad)
        self.sig = nn.Sigmoid()
        self.elu = nn.ELU()

    def forward(self, in_feat):
        out_feat = self.sig(self.conv1(in_feat)) * self.elu(self.conv2(in_feat))
        return out_feat

def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv

def batched_index_select(input, dim, index):
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def bbox_mask(x, bbox, H, W):
    bbox = bbox.to(x.device)
    b, o, _ = bbox.size()
    N = b * o

    bbox_1 = bbox.float().view(-1, 4)  # [b*obj,4]
    x0, y0 = bbox_1[:, 0], bbox_1[:, 1]
    ww, hh = bbox_1[:, 2], bbox_1[:, 3]

    x0 = x0.contiguous().view(N, 1).expand(N, H)
    ww = ww.contiguous().view(N, 1).expand(N, H)
    y0 = y0.contiguous().view(N, 1).expand(N, W)
    hh = hh.contiguous().view(N, 1).expand(N, W)

    X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).to(x.device)  # cuda(device=x.device)
    Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).to(x.device)  # cuda(device=x.device)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
    Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)

    out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1)
    return out_mask.view(b, o, H, W)

def bbox_boundary(x, bbox, H, W):
    bbox = bbox.to(x.device)
    b, o, _ = bbox.size()
    N = b * o
    h_eps, w_eps = 1/H, 1/W
    bbox_1 = bbox.float().view(-1, 4)  # [b*obj,4]
    x0, y0 = bbox_1[:, 0], bbox_1[:, 1]
    ww, hh = bbox_1[:, 2], bbox_1[:, 3]

    x0 = x0.contiguous().view(N, 1).expand(N, H)
    ww = ww.contiguous().view(N, 1).expand(N, H)
    y0 = y0.contiguous().view(N, 1).expand(N, W)
    hh = hh.contiguous().view(N, 1).expand(N, W)

    X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).to(x.device)  # cuda(device=x.device)
    Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).to(x.device)  # cuda(device=x.device)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    X_out_mask = ((X < -w_eps) + (X > 1+w_eps) + (X > w_eps) * (X < 1-w_eps)).view(N, 1, W).expand(N, H, W)
    Y_out_mask = ((Y < -h_eps) + (Y > 1+h_eps) + (Y > h_eps) * (Y < 1-h_eps)).view(N, H, 1).expand(N, H, W)

    out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1)
    return out_mask.view(b, o, H, W)

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn, nn.ReLU())

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

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
