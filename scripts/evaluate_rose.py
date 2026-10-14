import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import yaml
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


from src.utils.config import DEVICE, BATCH_SIZE

from src.models.attention_unet import get_attention_unet

from src.datasets.segmentation_dataset import (
    SegmentationDataset,
    val_transform
)

from src.datasets.dataset_loading import load_data_to_lists

from src.evaluation.metrics import calculate_metrics

from src.evaluation.topology_metrics import (
    get_pp_configs,
    calculate_cldice,
    apply_postprocessing_advanced
)



# =====================================
# Configuration
# =====================================

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



df_rose = pd.DataFrame(
    {
        "image_path": image_paths,
        "mask_path": mask_paths,
        "class": classes
    }
)



print(
    df_rose["class"].value_counts()
)



# =====================================
# Cross validation
# =====================================

skf = StratifiedKFold(
    n_splits=K_FOLDS,
    shuffle=True,
    random_state=42
)



volume_results = []

topology_results = []

configs = get_pp_configs()



# =====================================
# Evaluation loop
# =====================================

for fold, (_, val_idx) in enumerate(
    skf.split(
        df_rose.index,
        df_rose["class"]
    )
):

    print("\n" + "="*60)
    print(
        f"Evaluating fold {fold+1}/{K_FOLDS}"
    )
    print("="*60)



    model = get_attention_unet().to(DEVICE)



    model.load_state_dict(
        torch.load(
            ROOT /
            "checkpoints" /
            f"best_model_ROSE_fold{fold+1}.pth",
            map_location=DEVICE
        )
    )


    model.eval()



    df_val = df_rose.iloc[val_idx]



    dataset = SegmentationDataset(
        df_val["image_path"].tolist(),
        df_val["mask_path"].tolist(),
        val_transform
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )



    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)


            outputs = model(images)


            predictions = (
                torch.sigmoid(outputs) > 0.5
            ).float().cpu().numpy()


            masks = masks.numpy()



            for i in range(
                predictions.shape[0]
            ):

                pred = predictions[i,0]

                gt = masks[i,0]



                raw_dice = (
                    2*np.sum(pred*gt)+1e-7
                ) / (
                    np.sum(pred)
                    +
                    np.sum(gt)
                    +
                    1e-7
                )



                raw_cldice = calculate_cldice(
                    pred,
                    gt
                )



                volume_results.append(
                    {
                        "fold": fold+1,
                        "config": "Raw",
                        "dice": raw_dice
                    }
                )


                topology_results.append(
                    {
                        "fold": fold+1,
                        "config": "Raw",
                        "clDice": raw_cldice
                    }
                )



                for cfg in configs:


                    pp_pred = apply_postprocessing_advanced(
                        pred,
                        morph_op=cfg["morph_op"],
                        kernel_size=cfg["kernel_size"],
                        smooth=cfg["smooth"],
                        min_area=cfg["min_area"]
                    )



                    dice = (
                        2*np.sum(pp_pred*gt)+1e-7
                    ) / (
                        np.sum(pp_pred)
                        +
                        np.sum(gt)
                        +
                        1e-7
                    )


                    cldice = calculate_cldice(
                        pp_pred,
                        gt
                    )



                    volume_results.append(
                        {
                            "fold": fold+1,
                            "config": cfg["name"],
                            "dice": dice
                        }
                    )


                    topology_results.append(
                        {
                            "fold": fold+1,
                            "config": cfg["name"],
                            "clDice": cldice
                        }
                    )



    del model

    torch.cuda.empty_cache()



# =====================================
# Save results
# =====================================

results_dir = ROOT / "results"

results_dir.mkdir(
    exist_ok=True
)



pd.DataFrame(
    volume_results
).to_csv(
    results_dir /
    "evaluation_results.csv",
    index=False
)



pd.DataFrame(
    topology_results
).to_csv(
    results_dir /
    "topology_results.csv",
    index=False
)



print(
    "\nEvaluation completed successfully"
)