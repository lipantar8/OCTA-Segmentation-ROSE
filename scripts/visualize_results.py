import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import yaml
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


from src.utils.config import DEVICE, BATCH_SIZE

from src.models.attention_unet import get_attention_unet

from src.datasets.segmentation_dataset import (
    SegmentationDataset,
    val_transform
)

from src.datasets.dataset_loading import load_data_to_lists

from src.evaluation.topology_metrics import (
    apply_postprocessing_advanced,
    apply_topological_postprocessing
)



# =====================================
# Configuration
# =====================================

FOLD_TO_VISUALIZE = 1
NUM_IMAGES = 4



# =====================================
# Load paths
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



add_rose_dataset(
    rose_paths["SVC"]["images"],
    rose_paths["SVC"]["masks"],
    "ROSE-1_SVC"
)


add_rose_dataset(
    rose_paths["SVC_DVC"]["images"],
    rose_paths["SVC_DVC"]["masks"],
    "ROSE-1_Full"
)



import pandas as pd


df_rose = pd.DataFrame(
    {
        "image_path": image_paths,
        "mask_path": mask_paths,
        "class": classes
    }
)



# =====================================
# Create validation split
# =====================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


folds = list(
    skf.split(
        df_rose.index,
        df_rose["class"]
    )
)



_, val_idx = folds[
    FOLD_TO_VISUALIZE - 1
]


df_val = df_rose.iloc[val_idx]



dataset = SegmentationDataset(
    df_val["image_path"].tolist(),
    df_val["mask_path"].tolist(),
    val_transform
)



loader = DataLoader(
    dataset,
    batch_size=NUM_IMAGES,
    shuffle=True
)



# =====================================
# Load model
# =====================================

model = get_attention_unet().to(DEVICE)


model.load_state_dict(
    torch.load(
        ROOT /
        "checkpoints" /
        f"best_model_ROSE_fold{FOLD_TO_VISUALIZE}.pth",
        map_location=DEVICE
    )
)


model.eval()



# =====================================
# Prediction
# =====================================

images, masks = next(iter(loader))


images_device = images.to(DEVICE)


with torch.no_grad():

    outputs = model(
        images_device
    )


predictions = (
    torch.sigmoid(outputs) > 0.5
).cpu().numpy()



images = images.numpy()
masks = masks.numpy()



# =====================================
# Save figures
# =====================================

figure_dir = (
    ROOT /
    "results" /
    "figures"
)

figure_dir.mkdir(
    exist_ok=True
)



for i in range(
    predictions.shape[0]
):

    image = images[i].transpose(
        1,
        2
    )

    gt = masks[i,0]

    raw_pred = predictions[i,0]


    pp_pred = apply_postprocessing_advanced(
        raw_pred,
        morph_op="closing",
        kernel_size=3,
        smooth="median",
        min_area=30
    )


    raw_skeleton = apply_topological_postprocessing(
        raw_pred,
        morph_op=None,
        smooth=None,
        min_area=0
    )


    pp_skeleton = apply_topological_postprocessing(
        raw_pred,
        morph_op="closing",
        smooth="median",
        min_area=30
    )



    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16,8)
    )


    fig.suptitle(
        f"Fold {FOLD_TO_VISUALIZE} - Sample {i+1}"
    )


    axes[0,0].imshow(
        image,
        cmap="gray"
    )
    axes[0,0].set_title(
        "Input"
    )


    axes[0,1].imshow(
        gt,
        cmap="gray"
    )
    axes[0,1].set_title(
        "Ground Truth"
    )


    axes[0,2].imshow(
        raw_pred,
        cmap="gray"
    )
    axes[0,2].set_title(
        "Raw Prediction"
    )


    axes[0,3].imshow(
        pp_pred,
        cmap="gray"
    )
    axes[0,3].set_title(
        "Post-processing"
    )


    axes[1,0].imshow(
        image,
        cmap="gray"
    )
    axes[1,0].set_title(
        "Input"
    )


    axes[1,1].imshow(
        gt,
        cmap="gray"
    )
    axes[1,1].set_title(
        "GT Skeleton"
    )


    axes[1,2].imshow(
        raw_skeleton,
        cmap="gray"
    )
    axes[1,2].set_title(
        "Raw Skeleton"
    )


    axes[1,3].imshow(
        pp_skeleton,
        cmap="gray"
    )
    axes[1,3].set_title(
        "PP Skeleton"
    )



    for ax in axes.ravel():
        ax.axis("off")


    plt.tight_layout()


    plt.savefig(
        figure_dir /
        f"fold{FOLD_TO_VISUALIZE}_sample{i+1}.png",
        dpi=300
    )


    plt.close()



print(
    "Visualization completed successfully"
)