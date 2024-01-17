import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Given 
class Decoder_r(nn.Module):
    def __init__(self):
        super(Decoder_r, self).__init__()  
        self.pixelShuffle1 = nn.PixelShuffle(2)
        self.pixelShuffle2 = nn.PixelShuffle(2)
        # TODO assert if necessary
        #self.bilinear_interpolation_layer = F.interpolate(size=(320,320))

    def forward(self, x):
        dict_feats = dict()
        x = self.pixelShuffle1(x)
        dict_feats['step1'] = x
        x = self.pixelShuffle2(x)
        dict_feats['step2'] = x
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