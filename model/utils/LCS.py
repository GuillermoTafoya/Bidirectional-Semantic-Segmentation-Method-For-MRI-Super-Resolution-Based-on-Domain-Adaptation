import torch
from .notices import *
from .gpu import get_gpu_memory

import numpy as np

def LCS(As, s_Is, threshold=0.05, lambda_v=0.9):
    c_As_list = []
    i = -1
    prCyan(i:=i+1)
    prCyan("-"*20)
    for batch_As, batch_s_Is in zip(As, s_Is):
        corrected_As = torch.zeros_like(batch_As)
        j=-1
        for x in range(batch_s_Is.shape[1]):  
            j += 1; prDarkBlue(f'Completed {j} slices == {round((j*100)/383, 2)}%') if (j + 1) % (383 // 10) == 0 else None
            for y in range(batch_s_Is.shape[2]): 
                class_probs = batch_s_Is[:, x, y]
                max_prob, max_class = torch.max(class_probs, dim=0)

                # Adjusting for zero-based indexing if necessary
                original_label = batch_As[0, x, y].long() - 1
                if (j + 1) % (383 // 10) == 0:
                    prWhite(f'{batch_As[0, x, y].long()=}')

                # Ensure original_label is within the valid index range
                if original_label >= 0 and original_label < class_probs.shape[0]:
                    original_prob = class_probs[original_label]
                    
                    if (max_prob > lambda_v) and ((max_prob - original_prob) > threshold):
                        corrected_As[0, x, y] = max_class + 1  # Adjust back if labels are 1-indexed
                    else:
                        corrected_As[0, x, y] = original_label + 1
                else:
                    # Handle invalid label index, e.g., by keeping the original label
                    corrected_As[0, x, y] = batch_As[0, x, y]

        c_As_list.append(corrected_As)
        prGreen(f'{corrected_As.shape=}')

    return torch.stack(c_As_list, dim=0)

"""
def LCS(As, s_Is, threshold = 0.05, lambda_v = 0.9):
    c_As_list=[]
    i = -1
    prWhite(get_gpu_memory())
    prRed(f'{s_Is.shape=}')
    prRed(f'{As.shape=}')
    for batch_As, batch_s_Is in zip(As, s_Is):
        
        prCyan(i:=i+1)
        prCyan("-"*20)

        
        prRed(f'{batch_s_Is.shape=}')
        prRed(f'{batch_As.shape=}')

        c_As = []
        j=-1
        for x in range(batch_s_Is.shape[1]):
            prDarkBlue(j:=j+1)
            As_prob_line = []
            max_probs, max_classes = torch.max(batch_s_Is[x,:], dim=-1)
            for y in range(batch_s_Is.shape[2]):
                #prMagenta(f'{x=} {y=}')
                As_id_px = batch_As[0, x, y]-1
                
                if As_id_px == -1:
                    As_id_px += 1
                As_prob_line.append(batch_s_Is[int(As_id_px),x,y].unsqueeze(0))
            As_prob_line = torch.cat(As_prob_line,0)
            condition = (max_probs - As_prob_line) > threshold
            combined_condition = torch.logical_and(condition, max_probs > lambda_v)

            c_As_line = torch.where(combined_condition, max_classes+1, torch.zeros(size=max_probs.shape, device=combined_condition.device))
            prGreen(f'{c_As_line.shape=}')
            c_As.append(c_As_line)
        c_As_line = torch.cat(c_As_line, 0)
        As_mask = batch_As
        As_mask[As_mask>0] = 1
        c_As_list.append(torch.cat(c_As*As_mask,0))

    return torch.cat(c_As_list, 0)
"""