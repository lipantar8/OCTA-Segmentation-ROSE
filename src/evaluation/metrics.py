import numpy as np
import torch


from sklearn.metrics import confusion_matrix, roc_auc_score
from skimage.morphology import skeletonize

def calculate_metrics(pred, target, threshold=0.5):
    """
    Compute Dice coefficient from prediction logits and ground truth masks.

    Args:
        pred: Model output logits.
        target: Ground truth binary masks.
        threshold: Threshold used for binarization.

    Returns:
        Dice score.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum()

    dice = (2.0 * intersection + 1e-7) / (
        pred.sum() + target.sum() + 1e-7
    )

    return dice.item()