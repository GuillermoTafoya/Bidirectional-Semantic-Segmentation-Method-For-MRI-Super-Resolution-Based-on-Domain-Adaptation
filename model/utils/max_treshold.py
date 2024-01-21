import torch

# Inspired by https://arxiv.org/pdf/2012.04828.pdf

"""
def maximum_probability_thresholding(logits, threshold=0.9):
    
    probabilities = F.softmax(logits, dim=1)
    max_probs, pseudo_labels = torch.max(probabilities, 1)
    mask = max_probs > threshold

    return pseudo_labels, mask
"""


def max_threshold(predictions, lambda_v = 0.9):
    pseudo_labels=[]
    for batch in predictions:
        max_probs, max_classes = torch.max(batch, dim=-1)
        pseudo_label = torch.where(max_probs > lambda_v, max_classes+1, torch.zeros(size=max_probs.shape))
        pseudo_labels.append(pseudo_label)

    return torch.cat(pseudo_labels, 0)

