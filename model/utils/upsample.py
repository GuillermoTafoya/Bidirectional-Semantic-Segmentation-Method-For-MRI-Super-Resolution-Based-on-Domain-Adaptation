import torch.nn.functional as F
import torch
from .notices import *

def upsample(X, ratio=2):
    high_res = []
    
    #prCyan(f'input {X.shape=}')
    for batch in range(X.shape[0]):
        # Assuming X is of shape [batch_size, channels, height, width]
        im = X[batch, 0]  # Extract the first channel of the image

        h, w = im.shape  # Original height and width
        
        ht, wt = h * ratio, w * ratio  # Target height and width

        im = im[None, None, :]  # Reshape for interpolation

        # Upsample the image
        n_im = F.interpolate(im, size=[ht, wt], mode='bilinear', align_corners=True)

        #prOrange(f'single image upsampling: {n_im.shape=}')

        high_res.append(n_im)

    #prCyan(f'output {torch.cat(high_res, 0).shape=}')

    # Concatenate along the batch dimension
    return torch.cat(high_res, 0)
        

    
    """
    h = 1/ratio
    fhigh_res = []
    for batch in X:
        prCyan(f'{batch.shape=}')
        high_res = []
        for ch in batch:
            prMagenta(f'{ch.shape=}')
            im = ch
            h, w = im.shape[0], im.shape[1]
            ht, wt = h*ratio, w*ratio

            im = im[None, None, :]

            n_im = F.interpolate(im, size=[ht, wt], mode='bilinear', align_corners=True)

            
            high_res.append(n_im)
        high_res = torch.cat(high_res,0)
        fhigh_res.append(high_res)
    return torch.cat(fhigh_res,0)"""


    # def bicubic(x, a):
    #         if (abs(x) >= 0) and (abs(x) <= 1):
    #             return (a+2)*(abs(x)**3)-(a+3)*(abs(x)**2)+1
    #         elif (abs(x) > 1) and (abs(x) <= 2):
    #             return a*(abs(x)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(x)-4*a
         
    # high_res = []
       
    # for batch in range(X.shape(0)):
    #     im = X[batch][0].cpu().detach().numpy().squeeze()
    #     h = 1/ratio
    #     n_x, n_y = im.shape[0], im.shape[1]

    #     n_im = np.zeros((n_x*2,n_y*2))

    #     pad_im = n_im
    #     n_im[im[0]-im[0]/2:im[0]+im[0]/2, im[0]-im[0]/2:im[0]+im[0]/2] = im
        
    #     for j in range(im.shape[0]):
    #         for i in range(im.shape[1]):
    #             x, y = i * h + 2, j * h + 2
    
    #             x1 = 1 + x - math.floor(x) 
    #             x2 = x - math.floor(x) 
    #             x3 = math.floor(x) + 1 - x 
    #             x4 = math.floor(x) + 2 - x 

    #             y1 = 1 + y - math.floor(y) 
    #             y2 = y - math.floor(y) 
    #             y3 = math.floor(y) + 1 - y 
    #             y4 = math.floor(y) + 2 - y 
                    
    #             # Considering all nearby 16 values 
    #             mat_l = np.matrix([[bicubic(x1, a), bicubic(x2, a), bicubic(x3, a), bicubic(x4, a)]]) 
    #             mat_m = np.matrix([[n_im[int(y-y1), int(x-x1)], 
    #                                 n_im[int(y-y2), int(x-x1)], 
    #                                 n_im[int(y+y3), int(x-x1)], 
    #                                 n_im[int(y+y4), int(x-x1)]], 
    #                                 [n_im[int(y-y1), int(x-x2)], 
    #                                 n_im[int(y-y2), int(x-x2)], 
    #                                 n_im[int(y+y3), int(x-x2)], 
    #                                 n_im[int(y+y4), int(x-x2)]], 
    #                                 [n_im[int(y-y1), int(x+x3)], 
    #                                 n_im[int(y-y2), int(x+x3)], 
    #                                 n_im[int(y+y3), int(x+x3)], 
    #                                 n_im[int(y+y4), int(x+x3)]], 
    #                                 [n_im[int(y-y1), int(x+x4)], 
    #                                 n_im[int(y-y2), int(x+x4)], 
    #                                 n_im[int(y+y3), int(x+x4)], 
    #                                 n_im[int(y+y4), int(x+x4)]]]) 
    #             mat_r = np.matrix( 
    #                 [[bicubic(y1, a)], [bicubic(y2, a)], [bicubic(y3, a)], [bicubic(y4, a)]]) 
                    
    #             # Here the dot function is used to get  
    #             # the dot product of 2 matrices 
    #             n_im[j, i] = np.dot(np.dot(mat_l, mat_m), mat_r)
    #     n_im = torch.from_numpy(n_im).to(device)
    #     high_res.append(n_im)
    # return torch.cat(high_res,0)