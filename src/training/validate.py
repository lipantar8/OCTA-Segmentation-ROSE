import torch

from src.evaluation.metrics import calculate_metrics

def validate(model, loader, loss_fn, device):

    model.eval()

    val_loss = 0
    val_dice = 0


    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)


            outputs = model(images)


            loss = loss_fn(outputs, masks)

            val_loss += loss.item()


            dice_batch = calculate_metrics(
                outputs.detach(),
                masks
            )

            val_dice += dice_batch


    return (
        val_loss / len(loader),
        val_dice / len(loader)
    )