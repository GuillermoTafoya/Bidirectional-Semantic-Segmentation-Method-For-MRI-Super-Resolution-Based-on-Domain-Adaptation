""" Full assembly of the parts to form the complete network """

import torch.nn as nn


from .S import *
from .R import *
from .E import *

from .utils import downsample

class JUAN(nn.Module): # ROBERTO -> roBERTo
    def __init__(self, gpu, batch):
        super(JUAN, self).__init__()
        self.module_e = Module_e(gpu, batch)
        self.decoder_r = Decoder_r()
        self.pdd = Pdd()
        self.decoder_s = Decoder_s()
        self.odd = Odd()

    def forward(self, data_dic):
        # data_dic = {"Is": low_res images,
        #             "It": high_res images}

        # ----- MODULE E -----
        # TBD
        Is_features = self.module_e(data_dic['Is'])
        It_features = self.module_e(downsample(data_dic['It']))

        ## ----- MODULE R -----
        r_Is, layers_Is = self.decoder_r(Is_features)
        r_d_It, layers_It = self.decoder_r(It_features)

        Cs_features = self.module_e(downsample(r_Is))
        a, layers_Cs = self.module_s(Cs_features)

        ## ------ MODULE S -----
        S_pred = self.decoder_s(layers_Is, Is_features)
        T_pred = self.decoder_s(layers_It, It_features)
        Cs_pred = self.decoder_s(layers_Cs, Cs_features)

        return {"r_Is": r_Is, "r_d_It":r_d_It, "S_pred": S_pred, "T_pred": T_pred,
                "Cs_pred": Cs_pred}