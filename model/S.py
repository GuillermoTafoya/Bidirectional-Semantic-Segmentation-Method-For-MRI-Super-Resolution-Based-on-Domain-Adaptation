import torch
import torch.nn as nn

# Created by @simonamador, 12/12/2023
# Inspired by Liang & Wang (doi: 10.1080/01431161.2023.2169593)

# Decoder of the S Module. Input MRI feature map (80x80) and module R features for each label, 
# output image labels (320x320). 

class Decoder_s(nn.Module):
    def __init__(self):
        super(Decoder_s, self).__init__()
        self.step1 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=0, stride=2, output_padding=1)
        self.step2 = nn.ConvTranspose2d(1, 3, kernel_size=3, padding=0, stride=2, output_padding=1)
        self.step3 = nn.ConvTranspose2d(3, 6, kernel_size=9, padding=0, stride=1)
        self.activation = nn.Softmax()

    def forward(self, R_feats, x):
        x = self.step1(x)
        ft = x
        x = x + R_feats['step1']
        x = self.step2(x)
        x = x + R_feats['step2']
        x = self.step3(x)
        s = self.activation(x)
        return s, ft

# Output-Space Domain Discriminator (ODD) of the Module S. Encode generated labels to obtain ODD Loss.
class Odd(nn.Module):
    def __init__(self,):
        super(Odd, self).__init__()
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

    def forward(self, x):
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = self.step4(x)
        x = self.step5(x)
        return x
    
if __name__ == '__main__':
    from torchsummary import summary
    smodel = Decoder_s()
    oddmodel = Odd()
    summary(smodel, (1, 80, 80), device='cpu')
    summary(oddmodel, (1, 320, 320), device='cpu')
