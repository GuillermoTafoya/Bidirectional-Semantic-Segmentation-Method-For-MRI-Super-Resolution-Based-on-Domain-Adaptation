import torch.nn as nn

import E
import S
import R
from utils import downsample, upsample, max_treshold

class JUAN(nn.Module):
    def __init__(self):
        super(JUAN, self).__init__()
        self.module_e = E.Module_e
        self.decoder_r = R.Decoder_r()
        self.pdd = R.Pdd()
        self.decoder_s = S.Decoder_s()
        self.odd = S.Odd()

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