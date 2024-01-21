import torch

def LCS(As, s_Is, threshold = 0.05, lambda_v = 0.9):
    c_As_list=[]
    for batch_As, batch_s_Is in zip(As, s_Is):
        max_probs, max_classes = torch.max(batch_s_Is, dim=-1)
        
        As_prob = batch_As[max_classes]-1
        if As_prob == -1:
            As_prob += 1
        As_mask = As_prob
        As_mask[As_mask>0] = 1

        condition = (max_probs - batch_s_Is[max_classes[0], max_classes[1], As_prob]) > threshold
        
        c_As = torch.where(condition and max_probs > lambda_v, max_classes+1, torch.zeros(size=max_probs.shape))
        c_As_list.append(c_As*As_mask)

    return torch.cat(c_As_list, 0)