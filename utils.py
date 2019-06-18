from functools import reduce
import os
import sys
import subprocess
import time

import GPUtil
import imageio
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt


def get_num_params_from_model(model):
    return round(sum([reduce(lambda x, y: x * y, tensor.shape) for tensor in model.parameters()]) / 1e6, 2)


def random_weight_init(model, method='kaiming'):
    init_func = {'kaiming': nn.init.kaiming_normal_, 'normal': nn.init.normal_}

    for layer in model.modules():
        classname = layer.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                layer.weight.data = init_func[method](layer.weight.data)
            except AttributeError:
                pass
        elif classname.find('BatchNorm') != -1:
            layer.weight.data.normal_(1.0, 0.02)
            layer.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            layer.weight.data = init_func[method](layer.weight.data)
            

class StableBCELoss(nn.modules.Module):

       def __init__(self):
             super().__init__()

       def forward(self, input, target):
             neg_abs = - input.abs()
             loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
             return loss.mean()


def cross_entropy(output, target, pos_ratio=0.01):
    pos_weight = 1 / 0.01
    neg_weight = 1
    
    pos_loss = -(target * torch.log(torch.clamp(output, min=1e-8, max=1 - 1e-8))).mean()
    neg_loss = -pos_ratio * ((1 - target) * torch.log(torch.clamp(1 - output, min=1e-8, max=1 - 1e-8))).mean()
    
    pos_loss, neg_loss = torch.clamp(pos_loss, min=1e-8, max=10), torch.clamp(neg_loss, min=1e-8, max=10)
    loss = pos_loss + neg_loss
    
    return loss


def monitor_gpu_usage(gpu_ids): 
    pid = os.getpid()

    p = subprocess.Popen(['python', 'gpu_monitor.py', str(pid), gpu_ids], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # something long running
    
    return p 


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


