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



def get_category_metrics(model, loader, df_val, device):
    """
    Compute Dice score separately for different image categories.

    Args:
        model: Trained segmentation model.
        loader: Validation DataLoader.
        df_val: Validation dataframe containing image categories.
        device: Computation device.

    Returns:
        Dictionary containing Dice scores for each category.
    """

    model.eval()

    category_scores = {
        "SVC": [],
        "SVC_DVC": []
    }

    val_types = df_val["class"].tolist()

    idx = 0


    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)


            outputs = model(images)

            preds = (
                torch.sigmoid(outputs) > 0.5
            ).float()


            for j in range(images.size(0)):

                if idx >= len(val_types):
                    break


                pred = preds[j]
                mask = masks[j]


                intersection = (
                    pred * mask
                ).sum()

                dice = (
                    2.0 * intersection
                ) / (
                    pred.sum()
                    + mask.sum()
                    + 1e-7
                )


                image_type = str(
                    val_types[idx]
                ).strip()


                if image_type == "ROSE-1_Full":

                    category_scores["SVC_DVC"].append(
                        dice.item()
                    )


                elif image_type == "ROSE-1_SVC":

                    category_scores["SVC"].append(
                        dice.item()
                    )


                idx += 1


    return category_scores