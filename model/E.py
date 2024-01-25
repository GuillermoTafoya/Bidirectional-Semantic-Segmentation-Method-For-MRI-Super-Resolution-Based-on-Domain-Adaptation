import tensorflow as tf
from torch import from_numpy as t
import torch.nn as nn

from .unet.unet_network import Unet_network
from .unet.utils import *

class Module_e(nn.Module):
    def __init__(self,
                 gpu,
                 batch,
                 view = 'axi',
                 isize = [192, 192], 
                 iaxis = [7,7,4], 
                 coef = 'dice_coef'
                 ):
        super(Module_e, self).__init__()
        self.batch = batch
        view_path = './model/unet/weights/fold0'+view+'.h5'
        set_gpu('0' if gpu == '2' else '2')

        model = Unet_network([*isize,1], iaxis[0], metrics=coef).build()

        model.load_weights((view_path))
        deepest_layer_output = model.get_layer('activation_27').output
        self.model = tf.keras.Model(inputs=model.input, outputs=deepest_layer_output)

    def forward(self, x):
        x = x.cpu().numpy()
        deepest_layer_features = self.model.predict(x, batch_size=self.batch)
        deepest_layer_features = np.transpose(deepest_layer_features, (0, 3, 1, 2))
        return t(deepest_layer_features)