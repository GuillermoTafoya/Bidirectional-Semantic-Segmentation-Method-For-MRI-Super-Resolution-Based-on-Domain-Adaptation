import torch.nn as nn

from unet.config import config
from unet.unet_network import Unet_network

parser = config()
args = parser.parse_args()

class Module_e(nn.Module):
    def __init__(self,
                 axi,
                 isize = [160, 160], 
                 iaxis = [7,7,4], 
                 loss = 'dice_loss', 
                 coef = 'dice_coef', 
                 style = 'basic', 
                 ite = 3, 
                 depth = 4, 
                 dim = 32, 
                 init = 'he_normal', 
                 acti = 'elu', 
                 lr = 1e-4
                 ):
        super(Module_e, self).__init__()
    parser = config()

    def forward(self, x):
        
        return logits
    
test_dic, _ =make_dic(img_list, img_list, isize, 'axi',max_shape=max_shape)