import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Given 
class r_thingy(nn.Module):
    def __init__(self):
        super(r_thingy, self).__init__()  
        self.pixelShuffle1 = nn.PixelShuffle(2)
        self.pixelShuffle2 = nn.PixelShuffle(2)
        #self.bilinear_interpolation_layer = F.interpolate(size=(320,320))

    def forward(self, x):
        dict_feats = dict()
        x = self.pixelShuffle1(x)
        dict_feats['step1'] = x
        x = self.pixelShuffle2(x)
        dict_feats['step2'] = x
        #x = F.interpolate(x,size=(320,320),mode='bilinear')
        return (x,dict_feats)
    

# PatchGAN
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class pdd(nn.Module):
    # initializers
    def __init__(self, d=64):
        """
        five convolutional layers with 4 x 4 convolutional kernel size, 
        where the channel number is {64, 128, 256, 512, 1}, respectively
        """
        super(pdd, self).__init__()
        """
        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)
        """
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

    """
    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x
    """
    def forward(self, x):
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = self.step4(x)
        x = self.step5(x)
        return x
    
if __name__ == '__main__':
    #from torchsummary import summary
    from torchinfo import summary
    #model = Net().to(device)
    rmodel = r_thingy()
    pddmodel = pdd()
    summary(rmodel, input_size=(16, 80, 80), device='cpu')
    summary(pddmodel, (1, 320, 320), device='cpu')