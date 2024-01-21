import tensorflow as tf
from torch import from_numpy as t
import torch.nn as nn

from unet.config import config
from unet.unet_network import Unet_network
from unet.utils import *

parser = config()
args = parser.parse_args()

class Module_e(nn.Module):
    def __init__(self,
                 gpu,
                 view = 'axi',
                 isize = [160, 160], 
                 iaxis = [7,7,4], 
                 coef = 'dice_coef'
                 ):
        super(Module_e, self).__init__()
        view_path = './model/unet/weights/fold0'+view+'.h5'
        set_gpu(args.gpu)

        model = Unet_network([*isize,1], iaxis[0], metrics=coef).build()

        model.load_weights((view_path))
        deepest_layer_output = model.get_layer('activation_27').output
        self.model = tf.keras.Model(inputs=model.input, outputs=deepest_layer_output)

    def forward(self, x):
        deepest_layer_features = self.model.predict(x, batch=1)
        return (t(deepest_layer_features))