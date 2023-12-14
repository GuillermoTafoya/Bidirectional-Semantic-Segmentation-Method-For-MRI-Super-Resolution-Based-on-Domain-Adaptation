import numpy as np
import torch
from scipy import ndimage

def block_mean(X, device, fact=2):
    low_res = []

    for batch in im:
        im = X[batch][0].cpu().detach().numpy().squeeze()

        sx, sy = im.shape
        X, Y = np.ogrid[0:sx, 0:sy]

        regions = sy//fact * (X//fact) + Y//fact
        res = ndimage.mean(im, labels=regions, index=np.arange(regions.max() + 1))
        res.shape = (sx//fact, sy//fact)
        
        res = torch.from_numpy(res).to(device)
        low_res.append(res)
    return torch.cat(low_res, 0)