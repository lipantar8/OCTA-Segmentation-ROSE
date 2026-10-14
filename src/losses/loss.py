import torch.nn as nn
import segmentation_models_pytorch as smp


dice_loss_fn = smp.losses.DiceLoss(
    mode="binary",
    from_logits=True
)

bce_loss_fn = nn.BCEWithLogitsLoss()


def hybrid_loss(output, target):
    """
    Hybrid Dice-BCE loss.
    """
    return (
        0.7 * dice_loss_fn(output, target)
        +
        0.3 * bce_loss_fn(output, target)
    )