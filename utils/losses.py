import torch
import torch.nn as nn
import numpy as np
import lpips
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

per_loss = lpips.LPIPS(pretrained = True, net = 'alex', eval_mode = True, lpips = True).to(device)
l2_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()

def r_loss(r_d_It, It, r_Is, u_Is):
    if not torch.is_tensor(r_d_It):
        r_d_It = torch.from_numpy(np.expand_dims(r_d_It, axis=0)).type(torch.float).to(device)
        It = torch.from_numpy(np.expand_dims(It, axis=0)).type(torch.float).to(device)
        r_Is = torch.from_numpy(np.expand_dims(r_Is, axis=0)).type(torch.float).to(device)
        u_Is = torch.from_numpy(np.expand_dims(u_Is, axis=0)).type(torch.float).to(device)

    l_mse = l2_loss(r_d_It, It)
    l_per = per_loss(r_Is, u_Is)

    return l_mse + l_per

pdd_loss = 0

def s_loss(s_Is, u_As, s_d_r_Is):
    if not torch.is_tensor(s_Is):
        s_Is = torch.from_numpy(np.expand_dims(s_Is, axis=0)).type(torch.float).to(device)
        u_As = torch.from_numpy(np.expand_dims(u_As, axis=0)).type(torch.float).to(device)
        s_d_r_Is = torch.from_numpy(np.expand_dims(s_d_r_Is, axis=0)).type(torch.float).to(device)

    l1 = ce_loss(s_Is, u_As)
    l2 = ce_loss(s_d_r_Is, u_As)

    return l1 + l2

odd_loss = 0

fa_loss = 0

s_loss_b = 0