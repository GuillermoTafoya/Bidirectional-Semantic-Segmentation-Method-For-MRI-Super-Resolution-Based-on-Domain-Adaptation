import torch
import torch.nn as nn
import numpy as np
import os

from .vgg import *

# Created by @simonamador, 12/12/2023
# Inspired by Liang & Wang (doi: 10.1080/01431161.2023.2169593)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

class Perceptual(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x_vgg[f'relu{prefix[i]}_1'], y_vgg[f'relu{prefix[i]}_1'])
        return content_loss

l2_loss = nn.MSELoss()
per_loss = Perceptual()
ce_loss = nn.CrossEntropyLoss()

def r_loss(r_d_It, It, r_Is, u_Is, s_r_d_Is = None, As = None):

    l_mse = l2_loss(r_d_It, It)
    l_per = per_loss(r_Is, u_Is)
    loss = l_mse + l_per

    # ------- New perception -------
    if As is not None:
        loss += per_loss(s_r_d_Is, As)

    return loss

def pdd_loss(pdd_r_Is, pdd_It):
    return torch.log(pdd_It) + torch.log(1 - pdd_r_Is)

def s_loss(s_Is, u_As, s_d_r_Is, F_S, F_R, Tssl = None, At = None, c_As = None):
    l1 = ce_loss(s_Is, u_As)

    # ------- Composite Source -------
    if c_As is None:
        l2 = ce_loss(s_d_r_Is, u_As)
    else:
        l2 = ce_loss(s_d_r_Is, c_As)

    loss = l1 + l2

    # ------- FA Loss -------
    fa_loss = 0
    for i in range(F_S.shape[1]):
        for j in range(F_S.shape[2]):
            C_S = (F_S[:,i,:,:]/torch.abs(F_S[:,i,:,:])).transpose() * (F_S[:,:,j,:]/torch.abs(F_S[:,:,j,:]))
            C_R = (F_R[:,i,:,:]/torch.abs(F_R[:,i,:,:])).transpose() * (F_R[:,:,j,:]/torch.abs(F_R[:,:,j,:]))
            fa_loss = torch.abs(C_S, C_R)

    fa_loss /= F_S.shape[1]**2 * F_S.shape[2]**2
    loss += fa_loss
    
    # ------ Tssl -------
    if Tssl is not None:
        loss+= ce_loss(Tssl, At)

    return loss

def odd_loss(odd_d_It, odd_Is, z):
    loss = (1-z)*torch.log(odd_d_It) + z*torch.log(odd_Is)
    return - loss.sum()
