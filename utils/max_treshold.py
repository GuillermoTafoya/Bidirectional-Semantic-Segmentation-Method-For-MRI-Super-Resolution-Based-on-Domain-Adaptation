import torch

# Inspired by https://arxiv.org/pdf/2012.04828.pdf

def maximum_probability_thresholding_torch(predictions, lambda_values, p_value):
    """
    Apply maximum probability thresholding using PyTorch to assign pseudo labels.
    
    :param predictions: PyTorch tensor of shape (n_samples, n_classes) containing predicted probabilities.
    :param lambda_values: PyTorch tensor of shape (n_classes,) containing thresholds for each class.
    :param p_value: float, portion of the most confident predictions to consider for setting thresholds.
    
    :return: PyTorch tensor of pseudo labels.
    """
    # Adjusting lambda_values based on p_value if needed
    # This step is context-specific and depends on how you calculate or update lambda_values

    max_probs, max_classes = torch.max(predictions, dim=1)

    # Comparing max probabilities with lambda threshold for each class
    pseudo_labels = torch.where(max_probs > lambda_values[max_classes], max_classes, torch.tensor(-1).to(predictions.device))

    return pseudo_labels

