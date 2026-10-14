import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils.config import (
    DEVICE,
    BATCH_SIZE
)

from src.models.attention_unet import get_attention_unet

from src.datasets.segmentation_dataset import (
    SegmentationDataset,
    train_transform,
    val_transform
)

from src.datasets.dataset_loading import load_data_to_lists

from src.training.train import train_one_epoch
from src.training.validate import validate

from src.losses.loss import hybrid_loss


# ============================
# Hyperparameters
# ============================

EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 15


# ============================
# Load FIVES dataset
# ============================

DATA_IMAGE_DIR = (
    "data/FIVES/train/Original"
)

DATA_MASK_DIR = (
    "data/FIVES/train/Ground truth"
)


data = load_data_to_lists(
    DATA_IMAGE_DIR,
    DATA_MASK_DIR,
    "FIVES"
)


image_paths = data["image_paths"]
mask_paths = data["mask_paths"]


print(
    f"Loaded FIVES samples: {len(image_paths)}"
)


# ============================
# Train / validation split
# ============================

train_img, val_img, train_mask, val_mask = train_test_split(
    image_paths,
    mask_paths,
    test_size=0.2,
    random_state=42
)


train_dataset = SegmentationDataset(
    train_img,
    train_mask,
    train_transform
)


val_dataset = SegmentationDataset(
    val_img,
    val_mask,
    val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ============================
# Model
# ============================

model = get_attention_unet().to(DEVICE)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)


# ============================
# Training loop
# ============================

best_val_dice = 0.0
patience_counter = 0


for epoch in range(EPOCHS):

    train_loss, train_dice = train_one_epoch(
        model,
        train_loader,
        optimizer,
        hybrid_loss,
        DEVICE
    )


    val_loss, val_dice = validate(
        model,
        val_loader,
        hybrid_loss,
        DEVICE
    )


    scheduler.step(val_loss)


    print(
        f"\nEpoch {epoch+1}/{EPOCHS}"
    )

    print(
        f"Train loss: {train_loss:.4f} | "
        f"Train Dice: {train_dice:.4f}"
    )

    print(
        f"Val loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )


    if val_dice > best_val_dice:

        best_val_dice = val_dice
        patience_counter = 0


        torch.save(
            model.state_dict(),
            "checkpoints/best_model_FIVES.pth"
        )


        print(
            "Best model saved"
        )


    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"{patience_counter}/{PATIENCE}"
        )


    if patience_counter >= PATIENCE:

        print(
            "Early stopping"
        )

        break


print(
    "FIVES training completed"
)