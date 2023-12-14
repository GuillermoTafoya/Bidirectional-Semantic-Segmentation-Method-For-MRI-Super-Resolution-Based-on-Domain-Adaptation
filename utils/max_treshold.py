import torch

# Inspired by https://arxiv.org/pdf/2012.04828.pdf

"""
def maximum_probability_thresholding(logits, threshold=0.9):
    
    probabilities = F.softmax(logits, dim=1)
    max_probs, pseudo_labels = torch.max(probabilities, 1)
    mask = max_probs > threshold

    return pseudo_labels, mask
"""


def maximum_probability_thresholding_torch(predictions, lambda_values):

    max_probs, max_classes = torch.max(predictions, dim=1)
    pseudo_labels = torch.where(max_probs > lambda_values[max_classes], max_classes, torch.tensor(-1).to(predictions.device))
    
    return pseudo_labels

