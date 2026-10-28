from pathlib import Path
import json

import pandas as pd

from src.datasets.dataset_loading import load_data_to_lists
from src.evaluation.grid_search import run_grid_search
from src.utils.config import DEVICE


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "ROSE" / "ROSE-1"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "final_models_rose"
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"


# ============================================================
# LOAD DATA
# ============================================================

def main():
    print("=" * 80)
    print("OCTA ROSE - MODEL EVALUATION")
    print("=" * 80)

    print(f"Device: {DEVICE}")
    print(f"Dataset directory: {DATASET_ROOT}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Results directory: {RESULTS_DIR}")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {CHECKPOINT_DIR}\n"
            "Expected final_model_ROSE_fold1.pth through "
            "final_model_ROSE_fold5.pth."
        )

    print("\nLoading ROSE dataset...")

    # SVC
    svc_data = load_data_to_lists(
        DATASET_ROOT / "SVC" / "test" / "img",
        DATASET_ROOT / "SVC" / "test" / "gt",
        "SVC",
    )

    # SVC_DVC
    svc_dvc_data = load_data_to_lists(
        DATASET_ROOT / "SVC_DVC" / "test" / "img",
        DATASET_ROOT / "SVC_DVC" / "test" / "gt",
        "SVC_DVC",
    )

    # Combine the two datasets into one DataFrame
    df_rose = pd.DataFrame({
        "image_path": (
            svc_data["image_paths"]
            + svc_dvc_data["image_paths"]
        ),
        "mask_path": (
            svc_data["mask_paths"]
            + svc_dvc_data["mask_paths"]
        ),
        "class": (
            svc_data["classes"]
            + svc_dvc_data["classes"]
        ),
    })

    if df_rose.empty:
        raise ValueError(
            "No image-mask pairs found. Check the dataset paths."
        )

    if (
        df_rose["image_path"].isna().any()
        or df_rose["mask_path"].isna().any()
    ):
        raise ValueError(
            "Some image-mask paths are missing from the loaded dataset."
        )

    print(f"Number of image-mask pairs: {len(df_rose)}")
    print("\nClass distribution:")
    print(df_rose["class"].value_counts())

    # ========================================================
    # GRID SEARCH
    # ========================================================

    print("\nStarting post-processing grid search...")

    volume_df, topology_df, results, final_config = run_grid_search(
        df_rose=df_rose,
        checkpoint_dir=CHECKPOINT_DIR,
        device=DEVICE,
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    volume_path = RESULTS_DIR / "volume_grid_search.csv"
    topology_path = RESULTS_DIR / "topology_grid_search.csv"
    summary_path = RESULTS_DIR / "grid_search_summary.json"

    volume_df.to_csv(volume_path, index=False)
    topology_df.to_csv(topology_path, index=False)

    def make_json_serializable(value):
        if isinstance(value, dict):
            return {
                str(key): make_json_serializable(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                make_json_serializable(item)
                for item in value
            ]

        if hasattr(value, "to_dict"):
            return make_json_serializable(value.to_dict())

        if hasattr(value, "item"):
            return value.item()

        return value

    summary = {
        "final_config": make_json_serializable(final_config),
        "results": make_json_serializable(results),
        "device": str(DEVICE),
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nGrid search completed.")
    print(f"Volume results:   {volume_path}")
    print(f"Topology results: {topology_path}")
    print(f"Summary:          {summary_path}")

    print("\nSelected configuration:")
    print(final_config)


if __name__ == "__main__":
    main()