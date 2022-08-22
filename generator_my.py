import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_module import *
from .mask_regression import *
from .sync_batchnorm import SynchronizedBatchNorm2d
import copy
import torchvision
from .simsg.graph import BatchGraphTripleConv

BatchNorm = SynchronizedBatchNorm2d


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TripleGenerator(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, pred_classes=7, output_dim=3):
        super(TripleGenerator, self).__init__()
        self.num_classes = num_classes
        self.pred_classes = pred_classes

        self.label_embedding = nn.Embedding(num_classes, 180)
        self.pred_embedding = nn.Embedding(pred_classes, 180)

        num_w = 128 + 180 + 180 + 180
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))

        self.enc1 = GatedConv(4, ch*1, stride=2)  # 64x64
        self.enc2 = GatedConv(ch * 1, ch*2, stride=2)  # 32x32
        self.enc3 = GatedConv(ch * 2, ch*4, stride=2)  # 16x16
        self.enc4 = GatedConv(ch * 4, ch*8, stride=2)  # 8x8
        self.enc5 = GatedConv(ch * 8, ch*16, stride=2)  # 4x4

        self.res1 = ResBlock(ch * 32, ch * 16, upsample=True, num_w=num_w)  # ch * (32+16)
        self.res2 = ResBlock(ch * 24, ch * 8, upsample=True, num_w=num_w)  # ch * (16+8)
        self.res3 = ResBlock(ch * 12, ch * 4, upsample=True, num_w=num_w)  # ch * (8+4)
        self.res4 = ResBlock(ch * 6, ch * 2, upsample=True, num_w=num_w, psp_module=True)  # ch * (4+2)
        self.res5 = ResBlock(ch * 3, ch * 1, upsample=True, num_w=num_w, predict_mask=False)  # ch * (2+1)
        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        mapping.append(nn.Linear(num_w,num_w))
        mapping.append(nn.Linear(num_w,num_w))

        self.alpha1 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, num_classes, 1))

        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv3(num_w)
        self.init_parameter()

    def forward(self, z, bbox=None, z_im=None, y=None, triples=None, masked_images=None):
        enc_1 = m = self.enc1(masked_images)
        enc_2 = m = self.enc2(m)
        enc_3 = m = self.enc3(m)
        enc_4 = m = self.enc4(m)
        enc_5 = m = self.enc5(m)

        b, obj = z.size(0), z.size(1)
        label_embedding = self.label_embedding(y)
        z = z.view(b * obj, -1)

        s, p, o = triples.chunk(3, dim=-1)  # [B,# of triples, 1]
        s, p, o = [x.squeeze(-1) for x in [s, p, o]]  # [B, # of triples]
        s_emb = torch.gather(y, -1, s)
        o_emb = torch.gather(y, -1, o)

        label_embedding = self.label_embedding(torch.cat((s_emb, o_emb), 0))  # [2B, # of triples, 180]
        pred_embedding = self.pred_embedding(p)  # [B, # of triples, 180]
        s_label_embedding, o_label_embedding = label_embedding.chunk(2, dim=0)  # [B, # of triples, 180]
        total_embedding = torch.cat((s_label_embedding, pred_embedding, o_label_embedding), -1).view(b * obj, -1)
        w = torch.cat((z, total_embedding), dim=-1)  # [s,p,o] with order
        s_bbox = torch.gather(bbox, 1, s.view(b, obj, -1).expand(b, obj, 4))  # order bbox by s
        o_bbox = torch.gather(bbox, 1, o.view(b, obj, -1).expand(b, obj, 4))  # order bbox by o
        s_mask, o_mask = self.mask_regress(w, s_bbox, o_bbox)  # generated masks with order
        bmask = 0.5 * (s_mask + o_mask)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        bbox_mask_ = bbox_mask(z, bbox, 64, 64)

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        hh, ww = m.size(2), m.size(3)
        x = torch.cat((x, enc_5), dim=1)
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        x = torch.cat((x, enc_4), dim=1)
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x = torch.cat((x, enc_3), dim=1)
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x = torch.cat((x, enc_2), dim=1)
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x = torch.cat((x, enc_1), dim=1)
        x, _ = self.res5(x, w, stage_bbox)

        # to RGB
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class TripleGenerator_v2(nn.Module):
    def __init__(self, ch=16, z_dim=128, num_classes=10, pred_classes=7, output_dim=3, emb_dim=180, num_t=31):
        super(TripleGenerator_v2, self).__init__()
        self.num_classes = num_classes
        self.pred_classes = pred_classes

        self.label_embedding = nn.Embedding(num_classes, emb_dim)
        self.pred_embedding = nn.Embedding(pred_classes, emb_dim)

        num_w = 512
        self.z_dim = z_dim
        self.proj = nn.utils.spectral_norm(nn.Linear(z_dim + emb_dim*(2*num_t+1)+512*2*2, num_w))
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 32 * ch))

        self.enc1 = GatedConv(4, ch*1, stride=2)  # 8x128x128
        self.enc2 = GatedConv(ch * 1, ch * 2, stride=2)  # 16x64x64
        self.enc3 = GatedConv(ch * 2, ch * 4, stride=2)  # 32x32x32
        self.enc4 = GatedConv(ch * 4, ch * 8, stride=2)  # 64x16x16
        self.enc5 = GatedConv(ch * 8, ch * 16, stride=2)  # 128x8x8
        self.enc6 = GatedConv(ch * 16, ch * 32, stride=2)  # 256x4x4
        self.enc7 = GatedConv(ch * 32, ch * 32, stride=2)  # 512x2x2

        self.res1 = ResBlock(ch * 64, ch * 32, upsample=True, num_w=num_w)  # ch * (32+16)
        self.res2 = ResBlock(ch * 48, ch * 16, upsample=True, num_w=num_w)  # ch * (16+8)
        self.res3 = ResBlock(ch * 24, ch * 8, upsample=True, num_w=num_w)  # ch * (8+4)
        self.res4 = ResBlock(ch * 12, ch * 4, upsample=True, num_w=num_w)  # ch * (4+2)
        self.res5 = ResBlock(ch * 6, ch * 2, upsample=True, num_w=num_w)  # ch * (2+1)
        self.res6 = ResBlock(ch * 3, ch * 1, upsample=True, num_w=num_w, predict_mask=False)  # ch * (2+1)

        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        self.alpha1 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha5 = nn.Parameter(torch.zeros(1, num_classes, 1))

        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()
    
    def forward(self, content):
        z = None
        bbox = content['bbox']
        y = content['label']
        triples = content['triples']
        masked_images = content['image_contents']
        mask = content['mask']
        # ===================================Encoder===================================
        # image, mask = torch.split(masked_images, [3, 1], dim=1)
        masked_images = torch.cat((masked_images, mask), 1)
        enc_1 = m = self.enc1(masked_images)
        enc_2 = m = self.enc2(m)
        enc_3 = m = self.enc3(m)
        enc_4 = m = self.enc4(m)
        enc_5 = m = self.enc5(m)
        enc_6 = m = self.enc6(m)
        m = self.enc7(m)

        # ===================================Embedding===================================
        b, obj = y.size(0), y.size(1)
        if z is None:
            z = torch.randn(b, obj, self.z_dim).cuda()

        z = z.view(b * obj, -1)
        s, p, o = triples.chunk(3, dim=-1)  # [B,# of triples, 1]
        s, p, o = [x.squeeze(-1) for x in [s, p, o]]  # [B, # of triples]
        s_emb = torch.gather(y, -1, s)
        sel_p = p.unsqueeze(1).expand(-1, obj, -1) * torch.eq(s_emb.unsqueeze(1).expand(-1, obj, -1), y.unsqueeze(-1).expand(-1, -1, s.size(-1))).long()  # [B, num_o, num_t]
        sel_o = o.unsqueeze(1).expand(-1, obj, -1) * torch.eq(s_emb.unsqueeze(1).expand(-1, obj, -1), y.unsqueeze(-1).expand(-1, -1, s.size(-1))).long()  # [B, num_o, num_t]
        obj_triple = torch.cat((y.unsqueeze(-1), sel_o), -1)  # [B, num_o, # of triples + 1]
        label_embedding = self.label_embedding(obj_triple)  # [B, num_o, # of triples + 1, 180]
        pred_embedding = self.pred_embedding(sel_p)  # [B, num_o, # of triples, 180]      
        obj_triple_embedding = torch.cat((label_embedding, pred_embedding), 2).view(b, obj, -1)  # [B, num_o, (2 * (# of triples) + 1) * 180]
        obj_triple_embedding = torch.cat((obj_triple_embedding, m.view(b, -1).unsqueeze(1).expand(-1, obj, -1)), -1).view(b * obj, -1)  # [B, num_o, (2 * (# of triples) + 1) * 180 + feat]
        w = torch.cat((z, obj_triple_embedding), dim=-1)  # 
        w = self.proj(w)
        bmask = self.mask_regress(w, bbox)
        bbox_mask_ = bbox_mask(z, bbox, 64, 64)

        # ===================================Decoder===================================
        # print(m.shape, self.fc)
        hh, ww = enc_6.size(2), enc_6.size(3)
        x = F.interpolate(m, size=(hh, ww), mode='bilinear')
        # 8x8
        x = torch.cat((x, enc_6), dim=1)
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        x = torch.cat((x, enc_5), dim=1)
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x = torch.cat((x, enc_4), dim=1)
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x = torch.cat((x, enc_3), dim=1)
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x = torch.cat((x, enc_2), dim=1)
        x, stage_mask = self.res5(x, w, stage_bbox)

        # 256x256
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha5 = torch.gather(self.sigmoid(self.alpha5).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha5) + seman_bbox * alpha5
        x = torch.cat((x, enc_1), dim=1)
        x, _ = self.res6(x, w, stage_bbox)

        # to RGB
        x = self.final(x)
        return {'image_contents': x, 'stage_bbox': stage_bbox}

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class TripleGenerator_v0812(nn.Module):
    def __init__(self, ch=16, z_dim=128, num_classes=10, pred_classes=7, output_dim=3, emb_dim=180, num_t=8):
        super(TripleGenerator_v0812, self).__init__()
        self.num_classes = num_classes
        self.pred_classes = pred_classes

        self.label_embedding = nn.Embedding(num_classes, emb_dim)
        self.pred_embedding = nn.Embedding(pred_classes, emb_dim)

        num_w = 512
        self.proj = nn.utils.spectral_norm(nn.Linear(z_dim + emb_dim*(2*num_t+1)+ch*32*2*2, num_w))
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 32 * ch))

        self.enc1 = GatedConv(4, ch*1, stride=2)  # 8x128x128
        self.enc2 = GatedConv(ch * 1, ch * 2, stride=2)  # 16x64x64
        self.enc3 = GatedConv(ch * 2, ch * 4, stride=2)  # 32x32x32
        self.enc4 = GatedConv(ch * 4, ch * 8, stride=2)  # 64x16x16
        self.enc5 = GatedConv(ch * 8, ch * 16, stride=2)  # 128x8x8
        self.enc6 = GatedConv(ch * 16, ch * 32, stride=2)  # 256x4x4
        self.enc7 = GatedConv(ch * 32, ch * 32, stride=2)  # 512x2x2

        self.res1 = ResBlock(ch * 32, ch * 16, upsample=True, num_w=num_w)  # ch * (32+16)  enc6
        self.res2 = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_w)  # ch * (16+8)  enc5
        self.res3 = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_w)  # ch * (8+4)  enc4
        self.res4 = ResBlock(ch * 4, ch * 2, upsample=True, num_w=num_w)  # ch * (4+2)  enc3
        self.res5 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w)  # ch * (2+1)  enc2
        self.res6 = ResBlock(ch * 1, ch * 1, upsample=True, num_w=num_w, predict_mask=False)  # ch * (2+1)    enc1

        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        self.alpha1 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha5 = nn.Parameter(torch.zeros(1, num_classes, 1))

        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        bbox_network = list()
        bbox_network.append(nn.Linear(512+4, 128))
        bbox_network.append(nn.ReLU())
        bbox_network.append(nn.Linear(128, 64))
        bbox_network.append(nn.ReLU())
        bbox_network.append(nn.Linear(64, 4))
        self.bbox_network = nn.Sequential(*bbox_network)
        self.init_parameter()
    
    def forward(self, z, bbox=None, z_im=None, y=None, triples=None, masked_images=None):
        # ===================================Encoder===================================
        image, mask = torch.split(masked_images, [3, 1], dim=1)
        enc_1 = m = self.enc1(masked_images)
        enc_2 = m = self.enc2(m)
        enc_3 = m = self.enc3(m)
        enc_4 = m = self.enc4(m)
        enc_5 = m = self.enc5(m)
        enc_6 = m = self.enc6(m)
        m = self.enc7(m)

        # ===================================Embedding===================================
        b, obj = y.size(0), y.size(1)
        if z is None:
            z = torch.randn(b, obj, triples.size(1))
        z = z.view(b * obj, -1)
        s, p, o = triples.chunk(3, dim=-1)  # [B,# of triples, 1]
        s, p, o = [x.squeeze(-1) for x in [s, p, o]]  # [B, # of triples]
        s_emb = torch.gather(y, -1, s)
        sel_p = p.unsqueeze(1).expand(-1, obj, -1) * torch.eq(s_emb.unsqueeze(1).expand(-1, obj, -1), y.unsqueeze(-1).expand(-1, -1, s.size(-1))).long()  # [B, num_o, num_t]
        sel_o = o.unsqueeze(1).expand(-1, obj, -1) * torch.eq(s_emb.unsqueeze(1).expand(-1, obj, -1), y.unsqueeze(-1).expand(-1, -1, s.size(-1))).long()  # [B, num_o, num_t]
        obj_triple = torch.cat((y.unsqueeze(-1), sel_o), -1)  # [B, num_o, # of triples + 1]
        label_embedding = self.label_embedding(obj_triple)  # [B, num_o, # of triples + 1, 180]
        pred_embedding = self.pred_embedding(sel_p)  # [B, num_o, # of triples, 180]      
        obj_triple_embedding = torch.cat((label_embedding, pred_embedding), 2).view(b, obj, -1)  # [B, num_o, (2 * (# of triples) + 1) * 180]
        obj_triple_embedding = torch.cat((obj_triple_embedding, m.view(b, -1).unsqueeze(1).expand(-1, obj, -1)), -1).view(b * obj, -1)  # [B, num_o, (2 * (# of triples) + 1) * 180 + feat]
        w = torch.cat((z, obj_triple_embedding), dim=-1)  # 
        w = self.proj(w)

        # ===================================Mask_Regressor===================================
        bbox = self.bbox_network(torch.cat((w, bbox.view(b * obj, -1)), -1)).view(b, obj, -1)
        eps = 0.004
        bbox_x0y0 = torch.clamp(bbox[:,:,0:2], eps, 1.-eps)
        bbox_wh = torch.clamp(bbox[:,:,2:]-bbox_x0y0, eps, 1.-eps)
        bbox = torch.cat((bbox_x0y0, bbox_wh), -1)
        bmask = self.mask_regress(w, bbox)
        bbox_mask_ = bbox_mask(z, bbox, 64, 64)

        # ===================================Decoder===================================
        # print(m.shape, self.fc)
        hh, ww = enc_6.size(2), enc_6.size(3)
        x = F.interpolate(m, size=(hh, ww), mode='bilinear')
        # 8x8
        # x = torch.cat((x, enc_6), dim=1)
        temp_mask = F.interpolate(mask, size=(hh, ww), mode='bilinear')
        x = x * temp_mask + enc_6 * (1.-temp_mask)
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        # x = torch.cat((x, enc_5), dim=1)
        temp_mask = F.interpolate(mask, size=(hh, ww), mode='bilinear')
        x = x * temp_mask + enc_5 * (1.-temp_mask)
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        # x = torch.cat((x, enc_4), dim=1)
        temp_mask = F.interpolate(mask, size=(hh, ww), mode='bilinear')
        x = x * temp_mask + enc_4 * (1.-temp_mask)
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        # x = torch.cat((x, enc_3), dim=1)
        temp_mask = F.interpolate(mask, size=(hh, ww), mode='bilinear')
        x = x * temp_mask + enc_3 * (1.-temp_mask)
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        # x = torch.cat((x, enc_2), dim=1)
        temp_mask = F.interpolate(mask, size=(hh, ww), mode='bilinear')
        x = x * temp_mask + enc_2 * (1.-temp_mask)
        x, stage_mask = self.res5(x, w, stage_bbox)

        # 256x256
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, obj, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha5 = torch.gather(self.sigmoid(self.alpha5).expand(b, -1, -1), dim=1, index=y.view(b, obj, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha5) + seman_bbox * alpha5
        # x = torch.cat((x, enc_1), dim=1)
        temp_mask = F.interpolate(mask, size=(hh, ww), mode='bilinear')
        x = x * temp_mask + enc_1 * (1.-temp_mask)
        x, _ = self.res6(x, w, stage_bbox)

        # to RGB
        x = self.final(x)
        return x, stage_bbox, bbox

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResnetGenerator256(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator256, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128 + 180
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))

        self.res1 = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_w)
        self.res2 = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch * 8, ch * 8, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_w)
        self.res5 = ResBlock(ch * 4, ch * 2, upsample=True, num_w=num_w)
        self.res6 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False)
        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha5 = nn.Parameter(torch.zeros(1, 184, 1))
        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()

    def forward(self, z, bbox, z_im=None, y=None, include_mask_loss=False):
        b, o = z.size(0), z.size(1)

        label_embedding = self.label_embedding(y)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1))

        # preprocess bbox
        bmask = self.mask_regress(w, bbox)  # [B, num_o*2, m, m]

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        bbox_mask_ = bbox_mask(z, bbox, 128, 128)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)
        w = self.mapping(latent_vector.view(b * o, -1))

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        # label mask
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x, stage_mask = self.res5(x, w, stage_bbox)

        # 256x256
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha5 = torch.gather(self.sigmoid(self.alpha5).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha5) + seman_bbox * alpha5
        x, _ = self.res6(x, w, stage_bbox)
        # to RGB
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


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
