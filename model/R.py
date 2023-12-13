import torch
import torch.nn as nn
import torch.nn.functional as F

torch.nn.PixelShuffle

# Given 
class r_thingy(nn.Module):
    def __init__(self):
        self.pixelShuffle1 = nn.PixelShuffle(2)
        self.pixelShuffle2 = nn.PixelShuffle(2)
        self.bilinear_interpolation_layer = nn.UpsamplingBilinear2d(size=(320,320))

    def forward(self, dict_feats, x):
        x = self.pixelShuffle1(x)
        dict_feats['step1'] = x
        x = self.pixelShuffle2(x)
        dict_feats['step2'] = x
        x = self.bilinear_interpolation_layer(x)
        return (x,dict_feats)
    

# PatchGAN
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class pdd(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(pdd, self).__init__()
        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x
    
if __name__ == '__main__':
    from torchsummary import summary
    rmodel = r_thingy()
    pddmodel = pdd()
    summary(rmodel, (1, 80, 80), device='cpu')
    summary(pddmodel, (1, 320, 320), device='cpu')