import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils.config import (
    DEVICE,
    BATCH_SIZE
)

from src.models.attention_unet import get_attention_unet

from src.datasets.octa500_dataset import OCTA500Dataset
from src.datasets.dataset_loading import load_octa500_to_df

from src.datasets.segmentation_dataset import (
    train_transform,
    val_transform
)

from src.training.train import train_one_epoch
from src.training.validate import validate

from src.losses.loss import hybrid_loss


# =====================================
# Hyperparameters
# =====================================

EPOCHS = 1000
LEARNING_RATE = 5e-5
PATIENCE = 20


# =====================================
# Dataset OCTA500
# =====================================

df_octa500 = load_octa500_to_df(
    "data/OCTA500/images",
    "data/OCTA500/large_vessel",
    "data/OCTA500/capillary"
)


print(
    f"OCTA500 samples: {len(df_octa500)}"
)


# =====================================
# Train / validation split
# =====================================

train_df, val_df = train_test_split(
    df_octa500,
    test_size=0.15,
    random_state=42
)


train_dataset = OCTA500Dataset(
    image_paths=train_df["image_path"].tolist(),
    large_vessel_paths=train_df["lv_path"].tolist(),
    capillary_paths=train_df["cap_path"].tolist(),
    transform=train_transform
)


val_dataset = OCTA500Dataset(
    image_paths=val_df["image_path"].tolist(),
    large_vessel_paths=val_df["lv_path"].tolist(),
    capillary_paths=val_df["cap_path"].tolist(),
    transform=val_transform
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


# =====================================
# Model initialization from FIVES
# =====================================

model = get_attention_unet().to(DEVICE)


FIVES_CHECKPOINT = (
    "checkpoints/best_model_FIVES.pth"
)


model.load_state_dict(
    torch.load(
        FIVES_CHECKPOINT,
        map_location=DEVICE
    )
)


print(
    "Loaded FIVES pretrained weights"
)


# =====================================
# Optimizer and scheduler
# =====================================

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


# =====================================
# Training loop
# =====================================

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
        f"\nOCTA500 Epoch {epoch+1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Dice: {train_dice:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )


    if val_dice > best_val_dice:

        best_val_dice = val_dice
        patience_counter = 0


        torch.save(
            model.state_dict(),
            "results/best_model_OCTA500.pth"
        )


        print(
            "Best OCTA500 model saved"
        )


    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"{patience_counter}/{PATIENCE}"
        )


    if patience_counter >= PATIENCE:

        print(
            "Early stopping triggered"
        )

        break


print(
    "OCTA500 training completed"
)
