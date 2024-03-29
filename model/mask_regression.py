import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.bilinear import *
from .sync_batchnorm import SynchronizedBatchNorm2d
import pickle
import numpy as np
from torch.nn import Parameter


class MaskRegressNet(nn.Module):
    def __init__(self, obj_feat=128, mask_size=16, map_size=64):
        super(MaskRegressNet, self).__init__()
        self.mask_size = mask_size
        self.map_size = map_size

        self.fc = nn.utils.spectral_norm(nn.Linear(obj_feat, 128 * 4 * 4))
        conv1 = list()
        conv1.append(nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)))
        conv1.append(SynchronizedBatchNorm2d(128))
        conv1.append(nn.ReLU())
        self.conv1 = nn.Sequential(*conv1)

        conv2 = list()
        conv2.append(nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)))
        conv2.append(SynchronizedBatchNorm2d(128))
        conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*conv2)

        conv3 = list()
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)))
        conv3.append(SynchronizedBatchNorm2d(128))
        conv3.append(nn.ReLU())
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(128, 1, 1, 1)))
        conv3.append(nn.Sigmoid())
        self.conv3 = nn.Sequential(*conv3)

    def forward(self, obj_feat, bbox):
        """
        :param obj_feat: (b*num_o, feat_dim)
        :param bbox: (b, num_o, 4)
        :return: bbmap: (b, num_o, map_size, map_size)
        """
        b, num_o, _ = bbox.size()
        obj_feat = obj_feat.view(b * num_o, -1)
        x = self.fc(obj_feat)
        x = self.conv1(x.view(b * num_o, 128, 4, 4))
        x = F.interpolate(x, size=8, mode='bilinear')
        x = self.conv2(x)
        x = F.interpolate(x, size=16, mode='bilinear')
        x = self.conv3(x)
        x = x.view(b, num_o, 16, 16)

        bbmap = masks_to_layout(bbox, x, self.map_size).view(b, num_o, self.map_size, self.map_size)
        return bbmap


class MaskRegressNetv2(nn.Module):
    def __init__(self, obj_feat=128, mask_size=16, map_size=64):
        super(MaskRegressNetv2, self).__init__()
        self.mask_size = mask_size
        self.map_size = map_size

        self.fc = nn.utils.spectral_norm(nn.Linear(obj_feat, 256 * 4 * 4))
        conv1 = list()
        conv1.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv1.append(nn.InstanceNorm2d(256))
        conv1.append(nn.ReLU())
        self.conv1 = nn.Sequential(*conv1)

        conv2 = list()
        conv2.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv2.append(nn.InstanceNorm2d(256))
        conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*conv2)

        conv3 = list()
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv3.append(nn.InstanceNorm2d(256))
        conv3.append(nn.ReLU())
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(256, 1, 1, 1)))
        conv3.append(nn.Sigmoid())
        self.conv3 = nn.Sequential(*conv3)

    def forward(self, obj_feat, bbox):
        """
        :param obj_feat: (b*num_o, feat_dim)
        :param bbox: (b, num_o, 4)
        :return: bbmap: (b, num_o, map_size, map_size)
        """
        b, num_o, _ = bbox.size()
        obj_feat = obj_feat.view(b * num_o, -1)

        x = self.fc(obj_feat)
        # print(x.size())  # x.view has problem  # 992 4096
        # print(x.view(b * num_o, 256, 4, 4))
        x = self.conv1(x.view(b * num_o, 256, 4, 4))
        x = F.interpolate(x, size=8, mode='bilinear')
        x = self.conv2(x)
        x = F.interpolate(x, size=16, mode='bilinear')
        x = self.conv3(x)
        x = x.view(b, num_o, 16, 16)

        bbmap = masks_to_layout(bbox, x, self.map_size).view(b, num_o, self.map_size, self.map_size)
        return bbmap


class MaskRegressNetv3(nn.Module):
    def __init__(self, obj_feat=128, mask_size=16, map_size=64, num_classes=180):
        super(MaskRegressNetv3, self).__init__()
        self.mask_size = mask_size
        self.map_size = map_size

        self.fc = nn.utils.spectral_norm(nn.Linear(obj_feat, 256 * 4 * 4))
        conv1 = list()
        conv1.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv1.append(nn.InstanceNorm2d(256))
        conv1.append(nn.ReLU())
        self.conv1 = nn.Sequential(*conv1)

        conv2 = list()
        conv2.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv2.append(nn.InstanceNorm2d(256))
        conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*conv2)

        conv3 = list()
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv3.append(nn.InstanceNorm2d(256))
        conv3.append(nn.ReLU())
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(256, 2, 1, 1)))
        conv3.append(nn.Sigmoid())
        self.conv3 = nn.Sequential(*conv3)

    def forward(self, obj_feat, bbox, o_bbox):
        """
        :param obj_feat: (b*num_o, feat_dim)
        :param bbox: (b, num_o, 4)
        :return: bbmap: (b, num_o, map_size, map_size)
        """
        b, num_o, _ = bbox.size()
        obj_feat = obj_feat.view(b * num_o, -1)

        x = self.fc(obj_feat)  # x.view has problem  # 992 4096
        x = self.conv1(x.view(b * num_o, 256, 4, 4))
        x = F.interpolate(x, size=8, mode='bilinear')
        x = self.conv2(x)
        x = F.interpolate(x, size=16, mode='bilinear')
        x = self.conv3(x)
        x = x.view(b, num_o * 2, 16, 16)
        s, o = x.chunk(2, dim=1)
        s_bbmap = masks_to_layout(bbox, s, self.map_size).view(b, num_o, self.map_size, self.map_size)
        o_bbmap = masks_to_layout(o_bbox, o, self.map_size).view(b, num_o, self.map_size, self.map_size)

        return s_bbmap, o_bbmap

class MaskRegressNet16(nn.Module):
    def __init__(self, obj_feat=128, mask_size=16, map_size=64):
        super(MaskRegressNet16, self).__init__()
        self.mask_size = mask_size
        self.map_size = map_size

        self.fc = nn.utils.spectral_norm(nn.Linear(obj_feat, 256 * 4 * 4))
        conv1 = list()
        conv1.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv1.append(nn.InstanceNorm2d(256))
        conv1.append(nn.ReLU())
        self.conv1 = nn.Sequential(*conv1)

        conv2 = list()
        conv2.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv2.append(nn.InstanceNorm2d(256))
        conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*conv2)

        conv3 = list()
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv3.append(nn.InstanceNorm2d(256))
        conv3.append(nn.ReLU())
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(256, 1, 1, 1)))
        conv3.append(nn.Sigmoid())
        self.conv3 = nn.Sequential(*conv3)

    def forward(self, obj_feat, bbox):
        """
        :param obj_feat: (b*num_o, feat_dim)
        :param bbox: (b, num_o, 4)
        :return: bbmap: (b, num_o, map_size, map_size)
        """
        b, num_o, _ = bbox.size()
        obj_feat = obj_feat.view(b * num_o, -1)
        x = self.fc(obj_feat)
        x = self.conv1(x.view(b * num_o, 256, 4, 4))
        x = F.interpolate(x, size=8, mode='bilinear')
        x = self.conv2(x)
        x = F.interpolate(x, size=16, mode='bilinear')
        x = self.conv3(x)
        x = x.view(b, num_o, self.mask_size, self.mask_size)

        bbmap = masks_to_layout(bbox, x, self.map_size).view(b, num_o, self.map_size, self.map_size)
        return bbmap, x