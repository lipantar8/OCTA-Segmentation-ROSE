
"""Post-processing grid search for pretrained ROSE-1 segmentation models."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.datasets.segmentation_dataset import (
    SegmentationDataset,
    val_transform,
)
from src.evaluation.topology_metrics import (
    apply_postprocessing_advanced,
    evaluate_topology_thick,
    get_pp_configs,
)
from src.models.attention_unet import get_attention_unet


def run_grid_search(
    df_rose,
    checkpoint_dir,
    device,
):
    """Run volumetric and topology-aware post-processing grid search.

    The evaluation follows five-fold stratified cross-validation.
    Configuration selection uses the mean Dice threshold of 0.01
    below the best configuration, followed by the highest mean clDice.

    Args:
        df_rose: DataFrame containing image_path, mask_path and class.
        checkpoint_dir: Directory containing the five trained checkpoints.
        device: PyTorch device used for model inference.

    Returns:
        volume_df: Per-fold volumetric Dice results.
        topology_df: Per-fold topology-aware clDice results.
        results: Aggregated mean and standard deviation for each config.
        final_config: Selected configuration as a pandas Series.
    """
    checkpoint_dir = Path(checkpoint_dir)

    all_volume_configs_results = []
    all_topology_configs_results = []

    configs = get_pp_configs()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    for fold, (_, val_idx) in enumerate(
        skf.split(df_rose.index, df_rose["class"])
    ):
        print("\n" + "=" * 60)
        print(f"FOLD {fold + 1}/5")
        print("=" * 60)

        # Reconstruct the validation split using the same CV strategy.
        df_val_fold = df_rose.iloc[val_idx]

        # Load the trained model associated with the current fold.
        model = get_attention_unet().to(device)

        checkpoint_path = (
            checkpoint_dir
            / f"final_model_ROSE_fold{fold + 1}.pth"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            # Support PyTorch versions without the weights_only argument.
            state_dict = torch.load(
                checkpoint_path,
                map_location=device,
            )

        model.load_state_dict(state_dict)
        model.eval()

        val_dataset = SegmentationDataset(
            df_val_fold["image_path"].tolist(),
            df_val_fold["mask_path"].tolist(),
            val_transform,
        )

        loader_val = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
        )

        # --------------------------------------------------------
        # Volumetric grid search: Dice score.
        # --------------------------------------------------------
        print("\nVolumetric post-processing grid search...")

        with torch.no_grad():
            for cfg in configs:
                cfg_dice_scores = []

                for images, masks in loader_val:
                    images = images.to(device)
                    masks = masks.to(device)

                    outputs = model(images)

                    preds = (
                        (torch.sigmoid(outputs) > 0.5)
                        .float()
                        .cpu()
                        .numpy()
                    )

                    gt_masks = masks.cpu().numpy()

                    for b in range(preds.shape[0]):
                        processed_mask = apply_postprocessing_advanced(
                            preds[b, 0],
                            morph_op=cfg["morph_op"],
                            kernel_size=cfg["kernel_size"],
                            smooth=cfg["smooth"],
                            min_area=cfg["min_area"],
                            do_skeleton=False,
                        )

                        gt = gt_masks[b, 0]

                        inter = np.sum(processed_mask * gt)

                        dice = (
                            2 * inter + 1e-7
                        ) / (
                            np.sum(processed_mask)
                            + np.sum(gt)
                            + 1e-7
                        )

                        cfg_dice_scores.append(dice)

                mean_dice = np.mean(cfg_dice_scores)

                all_volume_configs_results.append(
                    {
                        "fold": fold + 1,
                        "config": cfg["name"],
                        "dice": mean_dice,
                    }
                )

                print(
                    f"{cfg['name']:<35}"
                    f" Dice={mean_dice:.4f}"
                )

        # --------------------------------------------------------
        # Topology-aware grid search: clDice score.
        # --------------------------------------------------------
        print("\nTopology-aware post-processing grid search...")

        (
            best_topo_cfg,
            fold_cldice,
            topo_scores,
            topo_configs,
        ) = evaluate_topology_thick(
            model,
            df_val_fold,
            device,
            val_transform,
        )

        for i, cfg in enumerate(topo_configs):
            all_topology_configs_results.append(
                {
                    "fold": fold + 1,
                    "config": cfg["name"],
                    "cldice": np.mean(topo_scores[i]),
                }
            )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nGrid search completed.")

    # --------------------------------------------------------
    # Aggregate mean and standard deviation across five folds.
    # --------------------------------------------------------
    volume_df = pd.DataFrame(all_volume_configs_results)
    topology_df = pd.DataFrame(all_topology_configs_results)

    volume_stats = (
        volume_df.groupby("config")["dice"]
        .agg(["mean", "std"])
        .rename(
            columns={
                "mean": "Dice_mean_5fold",
                "std": "Dice_std_5fold",
            }
        )
    )

    topology_stats = (
        topology_df.groupby("config")["cldice"]
        .agg(["mean", "std"])
        .rename(
            columns={
                "mean": "clDice_mean_5fold",
                "std": "clDice_std_5fold",
            }
        )
    )

    # Combine volumetric and topology-aware metrics.
    results = pd.concat(
        [volume_stats, topology_stats],
        axis=1,
    ).dropna()

    print("\n" + "=" * 80)
    print("FIVE-FOLD MEAN RESULTS BY CONFIGURATION")
    print("=" * 80)

    print(
        results.sort_values(
            by="Dice_mean_5fold",
            ascending=False,
        ).head(10)
    )

    # --------------------------------------------------------
    # Select configurations within 0.01 of the best mean Dice.
    # Among these candidates, select the highest mean clDice.
    # --------------------------------------------------------
    best_dice = results["Dice_mean_5fold"].max()

    candidate_configs = results[
        results["Dice_mean_5fold"] >= best_dice - 0.01
    ]

    print("\n" + "=" * 80)
    print("CONFIGURATIONS WITHIN 0.01 OF THE BEST MEAN DICE")
    print("=" * 80)

    print(candidate_configs)

    final_config = candidate_configs.sort_values(
        by="clDice_mean_5fold",
        ascending=False,
    ).iloc[0]

    print("\n" + "=" * 80)
    print("SELECTED POST-PROCESSING CONFIGURATION")
    print("=" * 80)

    print(f"Configuration: {final_config.name}")

    print(
        f"Mean five-fold Dice: "
        f"{final_config['Dice_mean_5fold']:.4f} "
        f"+/- {final_config['Dice_std_5fold']:.4f}"
    )

    print(
        f"Mean five-fold clDice: "
        f"{final_config['clDice_mean_5fold']:.4f} "
        f"+/- {final_config['clDice_std_5fold']:.4f}"
    )

    # --------------------------------------------------------
    # Compare the Raw baseline against the selected configuration.
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("RAW VS FINAL POST-PROCESSING")
    print("=" * 80)

    raw_config_name = [
        config_name
        for config_name in results.index
        if config_name.startswith("Raw")
    ][0]

    raw_dice = (
        volume_df[
            volume_df["config"] == raw_config_name
        ]
        .groupby("fold")["dice"]
        .mean()
    )

    final_dice = (
        volume_df[
            volume_df["config"] == final_config.name
        ]
        .groupby("fold")["dice"]
        .mean()
    )

    raw_cldice = (
        topology_df[
            topology_df["config"] == raw_config_name
        ]
        .groupby("fold")["cldice"]
        .mean()
    )

    final_cldice = (
        topology_df[
            topology_df["config"] == final_config.name
        ]
        .groupby("fold")["cldice"]
        .mean()
    )

    print("\nDice:")
    print(
        f"Raw:   {raw_dice.mean():.4f} "
        f"+/- {raw_dice.std():.4f}"
    )
    print(
        f"Final: {final_dice.mean():.4f} "
        f"+/- {final_dice.std():.4f}"
    )
    print(
        f"Delta: "
        f"{final_dice.mean() - raw_dice.mean():+.4f}"
    )

    print("\nclDice:")
    print(
        f"Raw:   {raw_cldice.mean():.4f} "
        f"+/- {raw_cldice.std():.4f}"
    )
    print(
        f"Final: {final_cldice.mean():.4f} "
        f"+/- {final_cldice.std():.4f}"
    )
    print(
        f"Delta: "
        f"{final_cldice.mean() - raw_cldice.mean():+.4f}"
    )

    print("\n" + "=" * 80)

    return volume_df, topology_df, results, final_config
