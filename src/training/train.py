import torch
from tqdm import tqdm

from src.evaluation.metrics import calculate_metrics


def train_one_epoch(model, loader, optimizer, loss_fn, device):

    model.train()

    scaler = torch.amp.GradScaler("cuda")

    total_loss = 0
    total_dice = 0

    optimizer.zero_grad()

    loop = tqdm(loader)

    for images, masks in loop:

        images = images.to(device)
        masks = masks.to(device)

        with torch.amp.autocast("cuda"):

            outputs = model(images)

            loss = loss_fn(outputs, masks)


        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()


        dice_batch = calculate_metrics(
            outputs.detach(),
            masks
        )

        total_loss += loss.item()
        total_dice += dice_batch


        loop.set_postfix(
            loss=loss.item(),
            dice=f"{dice_batch:.4f}"
        )


    return (
        total_loss / len(loader),
        total_dice / len(loader)
    )