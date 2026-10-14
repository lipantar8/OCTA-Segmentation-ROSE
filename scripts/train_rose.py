import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import yaml
import torch
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


from src.utils.config import DEVICE, BATCH_SIZE

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



# =====================================
# Hyperparameters
# =====================================

EPOCHS = 1000
LEARNING_RATE = 5e-6

PATIENCE = 30

K_FOLDS = 5



# =====================================
# Load dataset paths
# =====================================

with open(
    ROOT / "configs" / "paths.yaml",
    "r"
) as f:
    paths = yaml.safe_load(f)


rose_paths = paths["datasets"]["ROSE"]



# =====================================
# Load ROSE dataset
# =====================================

image_paths = []
mask_paths = []
classes = []



def add_rose_dataset(
    image_dir,
    mask_dir,
    class_name
):

    data = load_data_to_lists(
        image_dir,
        mask_dir,
        class_name
    )

    image_paths.extend(
        data["image_paths"]
    )

    mask_paths.extend(
        data["mask_paths"]
    )

    classes.extend(
        data["classes"]
    )



# Superficial vascular complex

add_rose_dataset(
    rose_paths["SVC"]["images"],
    rose_paths["SVC"]["masks"],
    "ROSE-1_SVC"
)



# Full thickness (SVC + DVC)

add_rose_dataset(
    rose_paths["SVC_DVC"]["images"],
    rose_paths["SVC_DVC"]["masks"],
    "ROSE-1_Full"
)



df_rose = pd.DataFrame(
    {
        "image_path": image_paths,
        "mask_path": mask_paths,
        "class": classes
    }
)


print("\nDataset distribution:")
print(
    df_rose["class"].value_counts()
)


print(
    f"\nTotal ROSE samples: {len(df_rose)}"
)



if len(df_rose) == 0:
    raise RuntimeError(
        "ROSE dataset is empty. Check paths.yaml"
    )



# =====================================
# Stratified K-Fold Cross Validation
# =====================================

skf = StratifiedKFold(
    n_splits=K_FOLDS,
    shuffle=True,
    random_state=42
)



# =====================================
# Training
# =====================================

for fold, (train_idx, val_idx) in enumerate(
    skf.split(
        df_rose.index,
        df_rose["class"]
    )
):

    print("\n" + "=" * 60)
    print(
        f"FOLD {fold + 1}/{K_FOLDS}"
    )
    print("=" * 60)



    # -------------------------------
    # Initialize model
    # -------------------------------

    model = get_attention_unet().to(DEVICE)


    checkpoint_path = (
        ROOT /
        "checkpoints" /
        "best_model_OCTA500.pth"
    )


    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )


    print(
        "Loaded OCTA500 pretrained weights"
    )



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



    # -------------------------------
    # Dataset split
    # -------------------------------

    df_train = df_rose.iloc[train_idx]

    df_val = df_rose.iloc[val_idx]



    train_dataset = SegmentationDataset(
        df_train["image_path"].tolist(),
        df_train["mask_path"].tolist(),
        train_transform
    )


    val_dataset = SegmentationDataset(
        df_val["image_path"].tolist(),
        df_val["mask_path"].tolist(),
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



    best_val_dice = 0.0

    patience_counter = 0



    # -------------------------------
    # Training loop
    # -------------------------------

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


        scheduler.step(
            val_loss
        )


        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )



        if val_dice > best_val_dice:

            best_val_dice = val_dice

            patience_counter = 0


            save_path = (
                ROOT /
                "checkpoints" /
                f"best_model_ROSE_fold{fold+1}.pth"
            )


            torch.save(
                model.state_dict(),
                save_path
            )


            print(
                "Best model saved"
            )


        else:

            patience_counter += 1

            print(
                f"No improvement: "
                f"{patience_counter}/{PATIENCE}"
            )



        if patience_counter >= PATIENCE:

            print(
                "Early stopping triggered"
            )

            break



    del model

    torch.cuda.empty_cache()



print(
    "\nROSE training completed successfully"
)