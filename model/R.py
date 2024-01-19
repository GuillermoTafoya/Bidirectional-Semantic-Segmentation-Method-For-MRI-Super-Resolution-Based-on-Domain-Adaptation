import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math


# https://github.com/yjn870/ESPCN-pytorch
# https://arxiv.org/abs/1609.05158
class ESPCN(nn.Module):
    def __init__(self, scale_factor, num_channels=1,output_channels = None):
        o_c = num_channels if output_channels is None else output_channels
        super(ESPCN, self).__init__()
        self.first_part = nn.Sequential(
            # num_channels 
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=5//2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=3//2),
            nn.Tanh(),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(32, o_c * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        features = x
        x = self.last_part(x)
        return (x,features)

# Given 
class Decoder_r(nn.Module):
    def __init__(self):
        super(Decoder_r, self).__init__() 
        # TODO assert channels
        self.firstUpscaling = ESPCN(2,64)
        self.secondUpscaling = ESPCN(2,64)
        # TODO assert if necessary
        #self.bilinear_interpolation_layer = F.interpolate(size=(320,320))

    def forward(self, x):
        dict_feats = dict()
        x,f = self.firstUpscaling(x)
        dict_feats['step1'] = f
        x,f = self.secondUpscaling(x)
        dict_feats['step2'] = f
        # TODO assert if necessary
        #x = F.interpolate(x,size=(320,320),mode='bilinear')
        return (x,dict_feats)
    

# PatchGAN
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Pdd(nn.Module):
    # initializers
    def __init__(self, d=64):
        """
        Five convolutional layers with 4 x 4 convolutional kernel size, 
        where the channel number is {64, 128, 256, 512, 1}, respectively
        """
        super(Pdd, self).__init__()

        self.step1 = nn.Sequential(
            nn.ConvTranspose2d(1, 64, kernel_size=4),
            nn.LeakyReLU(.02)
        )
        self.step2 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4),
            nn.LeakyReLU(.02)
        )
        self.step3 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=4),
            nn.LeakyReLU(.02)
        )
        self.step4 = nn.Sequential(
            nn.ConvTranspose2d(256, 512, kernel_size=4),
            nn.LeakyReLU(.02)
        )
        self.step5 = nn.ConvTranspose2d(512, 1, kernel_size=4)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = self.step4(x)
        x = self.step5(x)
        return x
    
if __name__ == '__main__':
    from torchinfo import summary
    rmodel = Decoder_r()
    pddmodel = Pdd()
    summary(rmodel, input_size=(16, 80, 80), device='cpu')
    summary(pddmodel, (1, 320, 320), device='cpu')