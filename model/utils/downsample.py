import numpy as np
import torch
from scipy import ndimage

def block_mean(X, fact=2):
    low_res = []

    for batch in X:
        im = batch[0].cpu().detach().numpy().squeeze()

        sx, sy = im.shape

        cropped_sx, cropped_sy = (sx // fact) * fact, (sy // fact) * fact
        cropped_im = im[:cropped_sx, :cropped_sy]

        X, Y = np.ogrid[0:cropped_sx, 0:cropped_sy]

        regions = cropped_sy // fact * (X // fact) + Y // fact
        res = ndimage.mean(cropped_im, labels=regions, index=np.arange(regions.max() + 1))
        res.shape = (cropped_sx // fact, cropped_sy // fact)
        
        res = torch.from_numpy(res)#.to(device)
        low_res.append(res)
    return torch.cat([x.unsqueeze(0) for x in low_res], 0)