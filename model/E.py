""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn

from config_parser import config
from unet.unet_network import unet_network as Unet_network

parser = config()
args = parser.parse_args()

class Module_e(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Module_e, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        model = Unet_network([*args.isize,1], args.iaxis[0], loss=args.loss, metrics=args.coef, style=args.style, 
                                  ite=args.ite, depth=args.depth, dim=args.dim, init=args.init, acti=args.acti, lr=args.lr).build()
        self.model = model.load_weights(args.axi)

        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2)
        
    def forward(self, x):
        _ = self.model(x)
        feature_map = self.model.deepest_feature_map
        reduced_feature_map = self.conv(feature_map)
        return reduced_feature_map
    
