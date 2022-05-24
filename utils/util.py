import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import sys
from typing import Tuple

def crop_resize(image, bbox, imsize=64, cropsize=28, label=None):
    """"
    :param image: (b, 3, h, w)
    :param bbox: (b, o, 4)
    :param imsize: input image size
    :param cropsize: image size after crop
    :param label:
    :return: crop_images: (b*o, 3, h, w)
    """
    crop_images = list()
    b, o, _ = bbox.size()
    if label is not None:
        rlabel = list()
    for idx in range(b):
        for odx in range(o):
            if torch.min(bbox[idx, odx]) < 0:
                continue
            crop_image = image[idx:idx+1, :, int(imsize*bbox[idx, odx, 1]):int(imsize*(bbox[idx, odx, 1]+bbox[idx, odx, 3])),
                               int(imsize*bbox[idx, odx, 0]):int(imsize*(bbox[idx, odx, 0]+bbox[idx, odx, 2]))]
            crop_image = F.interpolate(crop_image, size=(cropsize, cropsize), mode='bilinear')
            crop_images.append(crop_image)
            if label is not None:
                rlabel.append(label[idx, odx, :].unsqueeze(0))
    # print(rlabel)
    if label is not None:
        #if len(rlabel) % 2 == 1:
        #    return torch.cat(crop_images[:-1], dim=0), torch.cat(rlabel[:-1], dim=0)
        return torch.cat(crop_images, dim=0), torch.cat(rlabel, dim=0)
    return torch.cat(crop_images, dim=0)


def truncted_random(num_o=8, thres=1.0):
    z = np.ones((1, num_o, 128)) * 100
    for i in range(num_o):
        for j in range(128):
            while z[0, i, j] > thres or z[0, i, j] < - thres:
                z[0, i, j] = np.random.normal()
    return z


# VGG Features matching
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss



class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.eps = 1e-6
        self.tau = 7e-2
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, x, y):
        batch_size = x.size(0)
        log = 0.
        for i in range(batch_size):
            pin_x = x[i].expand(batch_size, -1)
            esp_cos = torch.exp(self.cos(pin_x, y)/self.tau)
            pos = esp_cos[i]
            log = log + torch.log(pos/(torch.sum(esp_cos + self.eps, dim=0) + self.eps)) * (-1./(2. * batch_size))
        
        for i in range(batch_size):
            pin_y = y[i].expand(batch_size, -1)
            esp_cos = torch.exp(self.cos(pin_y, x)/self.tau)
            pos = esp_cos[i]
            log = log + torch.log(pos/(torch.sum(esp_cos + self.eps, dim=0) + self.eps)) * (-1./(2. * batch_size))
        
        return log


class VggContrastiveLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(VggContrastiveLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).cuda()
        self.crit = ContrastiveLoss()
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        return self.crit(x_vgg, y_vgg)


class Inceptionv3OnlyFeature(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Inceptionv3OnlyFeature, self).__init__()
        # Get a resnet50 backbone
        self.net = models.inception.inception_v3(pretrained=True)
        self.net.eval()
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x = 2.*x - 1.
        # print(torch.min(x), torch.max(x))
        # N x 3 x 299 x 299
        x = self.net.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.net.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.net.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.net.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.net.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.net.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.net.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.net.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.net.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.net.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.net.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.net.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.net.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.net.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.net.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.net.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.net.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.net.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = feat = self.net.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.net.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.net.fc(x)
        # N x 1000 (num_classes)
        return feat, x


def compute_metric(x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        r"""
        Fits multivariate Gaussians: :math:`X \sim \mathcal{N}(\mu_x, \sigma_x)` and
        :math:`Y \sim \mathcal{N}(\mu_y, \sigma_y)` to image stacks.
        Then computes FID as :math:`d^2 = ||\mu_x - \mu_y||^2 + Tr(\sigma_x + \sigma_y - 2\sqrt{\sigma_x \sigma_y})`.

        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D)`

        Returns:
            The Frechet Distance.
        """
        # GPU -> CPU
        mu_x, sigma_x = _compute_statistics(x_features.detach().to(dtype=torch.float64))
        mu_y, sigma_y = _compute_statistics(y_features.detach().to(dtype=torch.float64))

        score = _compute_fid(mu_x, sigma_x, mu_y, sigma_y)

        return score

def _approximation_error(matrix: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error


def _sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Square root of matrix using Newton-Schulz Iterative method
    Source: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method
    Returns:
        Square root of matrix
        Error
    """
    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)
    Z = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1, device=matrix.device, dtype=matrix.dtype)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.tensor([0.], device=error.device, dtype=error.dtype), atol=1e-5):
            break

    return s_matrix, error


def _compute_fid(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor,
                 eps=1e-6) -> torch.Tensor:
    r"""
    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant. used if sigma_1 @ sigma_2 matrix is singular

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2
    covmean, _ = _sqrtm_newton_schulz(sigma1.mm(sigma2))

    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        print(f'FID calculation produces singular product; adding {eps} to diagonal of cov estimates')
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean, _ = _sqrtm_newton_schulz((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


def _cov(m: torch.Tensor, rowvar: bool = True) -> torch.Tensor:
    r"""Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() < 2:
        m = m.view(1, -1)

    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def _compute_statistics(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculates the statistics used by FID
    Args:
        samples:  Low-dimension representation of image set.
            Shape (N_samples, dims) and dtype: np.float32 in range 0 - 1
    Returns:
        mu: mean over all activations from the encoder.
        sigma: covariance matrix over all activations from the encoder.
    """
    mu = torch.mean(samples, dim=0)
    sigma = _cov(samples, rowvar=False)
    return mu, sigma

def inception_score(features: torch.Tensor, num_splits: int = 10):
    r"""Compute Inception Score for a list of image features.
    Expects raw logits from Inception-V3 as input.

    Args:
        features (torch.Tensor): Low-dimension representation of image set. Shape (N_samples, encoder_dim).
        num_splits: Number of parts to divide features. Inception Score is computed for them separately and
            results are then averaged.

    Returns:
        score

        variance

    References:
        "A Note on the Inception Score"
        https://arxiv.org/pdf/1801.01973.pdf

    """
    assert len(features.shape) == 2, \
        f"Features must have shape (N_samples, encoder_dim), got {features.shape}"
    N = features.size(0)

    # Convert logits to probabilities
    probas = F.softmax(features, dim=-1)

    # In the paper the score is computed for 10 splits of the dataset and then averaged.
    partial_scores = []
    for i in range(num_splits):
        subset = probas[i * (N // num_splits): (i + 1) * (N // num_splits), :]

        # Compute KL divergence
        p_y = torch.mean(subset, dim=0)
        scores = []
        for k in range(subset.shape[0]):
            p_yx = subset[k, :]
            scores.append(F.kl_div(p_y.log(), p_yx, reduction='sum'))

        # Compute exponential of the mean of the KL-divergence for each split
        partial_scores.append(torch.tensor(scores).mean().exp())

    partial_scores = torch.tensor(partial_scores)
    return torch.mean(partial_scores).to(features), torch.std(partial_scores).to(features)
