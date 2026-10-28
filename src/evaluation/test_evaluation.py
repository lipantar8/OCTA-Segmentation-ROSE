
"""
Test-set inference for the ROSE OCTA segmentation project.

The evaluation procedure follows the original notebook:
- Five-fold model ensemble
- Inference without TTA
- Test-time augmentation with horizontal and vertical flips
- Final post-processing
- Extended segmentation metrics
- Individual evaluation of the five fold models
"""

from pathlib import Path
import gc
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
)
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader

from src.datasets.segmentation_dataset import (
    SegmentationDataset,
    val_transform,
)
from src.evaluation.topology_metrics import (
    apply_postprocessing_advanced,
    calculate_cldice,
)
from src.models.attention_unet import get_attention_unet


# ============================================================
# EXTENDED METRICS
# ============================================================

def calculate_extended_metrics(gt_mask, pred_mask, pred_probs=None):
    """Calculate the extended metrics used in the original notebook."""

    y_true = gt_mask.flatten().astype(int)
    y_pred = pred_mask.flatten().astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)
    g_mean = np.sqrt(sens * spec)
    fdr = fp / (tp + fp + 1e-7)

    kappa = cohen_kappa_score(y_true, y_pred)

    try:
        if pred_probs is not None:
            auc = roc_auc_score(
                y_true,
                pred_probs.flatten(),
            )
        else:
            auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.5

    return acc, auc, g_mean, kappa, fdr


# ============================================================
# TEST DATA
# ============================================================

def append_test_data(img_dir, gt_dir, class_name):
    """
    Pair test images and masks following the sorted-list approach
    used in the original notebook.
    """

    img_dir = Path(img_dir)
    gt_dir = Path(gt_dir)

    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"Test image directory not found: {img_dir}"
        )

    if not gt_dir.is_dir():
        raise FileNotFoundError(
            f"Ground-truth directory not found: {gt_dir}"
        )

    image_paths = sorted(
        path for path in img_dir.iterdir() if path.is_file()
    )
    mask_paths = sorted(
        path for path in gt_dir.iterdir() if path.is_file()
    )

    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"{class_name}: found {len(image_paths)} images but "
            f"{len(mask_paths)} masks. Check the dataset pairing."
        )

    return [
        (str(img_path), str(mask_path), class_name)
        for img_path, mask_path in zip(image_paths, mask_paths)
    ]


# ============================================================
# DEVICE SYNCHRONIZATION
# ============================================================

def sync_device(device):
    """Synchronize CUDA operations when running on a CUDA device."""

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# SUMMARY HELPERS
# ============================================================

METRIC_NAMES = {
    "dice": "Dice",
    "cldice": "clDice",
    "acc": "Accuracy",
    "auc": "AUC",
    "g_mean": "G-mean",
    "kappa": "Kappa",
    "fdr": "FDR",
}


def print_metric_summary(results, categories, heading):
    """Print means and standard deviations for the collected metrics."""

    print("\n" + "=" * 80)
    print(heading)
    print("=" * 80)

    for category in categories:
        print(f"\n--- {category.upper()} ---")

        for metric, label in METRIC_NAMES.items():
            values = np.asarray(results[category][metric], dtype=float)

            if values.size == 0:
                print(f"{label:<12} No results")
                continue

            print(
                f"{label:<12} "
                f"{np.nanmean(values):.4f} "
                f"± {np.nanstd(values):.4f}"
            )

        if "time" in results[category]:
            times = np.asarray(results[category]["time"], dtype=float)

            if times.size:
                print(
                    f"{'Time/img':<12} "
                    f"{np.mean(times):.4f} s"
                )


# ============================================================
# MAIN TEST EVALUATION
# ============================================================

def run_test_evaluation(
    test_svc_images,
    test_svc_masks,
    test_full_images,
    test_full_masks,
    checkpoint_dir,
    output_dir,
    skeleton_dir,
    device,
):
    """
    Run ensemble test inference, TTA comparison, individual-fold
    evaluation, metric summaries, and qualitative visualization.

    Parameters
    ----------
    test_svc_images, test_svc_masks:
        Directories containing SVC test images and thick ground-truth masks.
    test_full_images, test_full_masks:
        Directories containing SVC_DVC test images and thick masks.
    checkpoint_dir:
        Directory containing final_model_ROSE_fold1.pth through fold5.pth.
    output_dir:
        Directory where final prediction masks and the qualitative figure
        will be saved.
    skeleton_dir:
        Directory where final prediction skeletons will be saved.
    device:
        PyTorch device, for example torch.device("cpu") or torch.device("cuda").
    """

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    skeleton_dir = Path(skeleton_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("TEST SET INFERENCE — FIVE-FOLD ENSEMBLE")
    print("=" * 60)
    print(f"Device: {device}")

    # --------------------------------------------------------
    # 1. LOAD TEST SET
    # --------------------------------------------------------

    test_items = []

    test_items.extend(
        append_test_data(
            test_svc_images,
            test_svc_masks,
            "SVC",
        )
    )

    test_items.extend(
        append_test_data(
            test_full_images,
            test_full_masks,
            "Full",
        )
    )

    if not test_items:
        raise RuntimeError("No test images were found.")

    test_image_paths = [item[0] for item in test_items]
    test_mask_paths = [item[1] for item in test_items]
    test_classes = [item[2] for item in test_items]

    print(f"Test images found: {len(test_image_paths)}")
    print(f"SVC images: {test_classes.count('SVC')}")
    print(f"Full images: {test_classes.count('Full')}")

    test_dataset = SegmentationDataset(
        test_image_paths,
        test_mask_paths,
        val_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # 2. LOAD FIVE FOLD MODELS
    # --------------------------------------------------------

    models = []

    for fold in range(5):
        checkpoint_path = (
            checkpoint_dir
            / f"final_model_ROSE_fold{fold + 1}.pth"
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        print(f"Loading model Fold {fold + 1}/5")

        model = get_attention_unet().to(device)

        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            # Compatibility with PyTorch versions without weights_only.
            state_dict = torch.load(
                checkpoint_path,
                map_location=device,
            )

        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    print("Five-fold ensemble loaded.")

    # --------------------------------------------------------
    # 3. SINGLE-MODEL FORWARD-PASS TIMING
    # --------------------------------------------------------

    benchmark_model = models[0]
    images, _ = next(iter(test_loader))
    images = images.to(device)

    if str(device).startswith("cuda") and torch.cuda.is_available():
        with torch.no_grad():
            for _ in range(10):
                _ = benchmark_model(images)

    sync_device(device)
    start_time = time.perf_counter()

    with torch.no_grad():
        _ = benchmark_model(images)

    sync_device(device)
    single_forward_time = time.perf_counter() - start_time

    print(
        f"Single forward time: {single_forward_time:.4f} s/image"
    )

    # --------------------------------------------------------
    # 4. INITIALIZE METRIC STORAGE
    # --------------------------------------------------------

    categories = ["SVC", "Full", "Total"]

    test_results = {
        category: {
            "dice": [],
            "cldice": [],
            "acc": [],
            "auc": [],
            "g_mean": [],
            "kappa": [],
            "fdr": [],
            "time": [],
        }
        for category in categories
    }

    metrics_no_tta = {
        category: {
            metric: []
            for metric in [
                "dice",
                "cldice",
                "acc",
                "auc",
                "g_mean",
                "kappa",
                "fdr",
            ]
        }
        for category in categories
    }

    times_no_tta = []
    times_full_tta = []
    per_image_results = []

    # --------------------------------------------------------
    # 5. ENSEMBLE INFERENCE: WITHOUT TTA AND WITH TTA
    # --------------------------------------------------------

    print("\nStarting test inference: without TTA vs. with TTA")

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device)

            gt_mask = masks.cpu().numpy()[0, 0]
            current_class = test_classes[i]
            filename = Path(test_image_paths[i]).name

            # -----------------------------------------------
            # A. FIVE-MODEL ENSEMBLE WITHOUT TTA
            # -----------------------------------------------

            sync_device(device)
            start_no_tta = time.perf_counter()

            probs_no_tta = []

            for model in models:
                prob = torch.sigmoid(model(images))
                probs_no_tta.append(prob)

            pred_probs_no_tta = torch.stack(
                probs_no_tta
            ).mean(dim=0)

            pred_probs_no_tta_np = (
                pred_probs_no_tta.cpu().numpy()[0, 0]
            )

            sync_device(device)
            time_no_tta = time.perf_counter() - start_no_tta
            times_no_tta.append(time_no_tta)

            pred_bin_no_tta = (
                pred_probs_no_tta_np > 0.5
            ).astype(np.float32)

            intersection_no = np.sum(
                pred_bin_no_tta * gt_mask
            )

            dice_no = (
                2 * intersection_no + 1e-7
            ) / (
                np.sum(pred_bin_no_tta)
                + np.sum(gt_mask)
                + 1e-7
            )

            cldice_no = calculate_cldice(
                pred_bin_no_tta,
                gt_mask,
            )

            (
                acc_no,
                auc_no,
                gm_no,
                kappa_no,
                fdr_no,
            ) = calculate_extended_metrics(
                gt_mask,
                pred_bin_no_tta,
                pred_probs_no_tta_np,
            )

            for category in [current_class, "Total"]:
                metrics_no_tta[category]["dice"].append(dice_no)
                metrics_no_tta[category]["cldice"].append(cldice_no)
                metrics_no_tta[category]["acc"].append(acc_no)
                metrics_no_tta[category]["auc"].append(auc_no)
                metrics_no_tta[category]["g_mean"].append(gm_no)
                metrics_no_tta[category]["kappa"].append(kappa_no)
                metrics_no_tta[category]["fdr"].append(fdr_no)

            # -----------------------------------------------
            # B. FIVE-MODEL ENSEMBLE WITH TTA
            # Original image + horizontal flip + vertical flip
            # -----------------------------------------------

            sync_device(device)
            start_full = time.perf_counter()

            fold_predictions = []

            for model in models:
                pred1 = torch.sigmoid(model(images))

                pred2 = torch.flip(
                    torch.sigmoid(
                        model(torch.flip(images, [3]))
                    ),
                    [3],
                )

                pred3 = torch.flip(
                    torch.sigmoid(
                        model(torch.flip(images, [2]))
                    ),
                    [2],
                )

                pred_tta = (pred1 + pred2 + pred3) / 3.0
                fold_predictions.append(pred_tta)

            pred_probs = torch.stack(
                fold_predictions
            ).mean(dim=0)

            pred_probs_np = pred_probs.cpu().numpy()[0, 0]

            pred_bin_raw = (
                pred_probs_np > 0.5
            ).astype(np.float32)

            # -----------------------------------------------
            # C. FINAL POST-PROCESSING
            # -----------------------------------------------

            pred_final = apply_postprocessing_advanced(
                pred_bin_raw,
                morph_op="closing",
                kernel_size=3,
                smooth="gaussian",
                min_area=0,
                do_skeleton=False,
            )

            sync_device(device)
            time_full = time.perf_counter() - start_full
            times_full_tta.append(time_full)

            # -----------------------------------------------
            # D. SAVE FINAL MASK AND SKELETON
            # -----------------------------------------------

            final_skeleton = skeletonize(
                pred_final > 0.5
            )

            prediction_path = (
                output_dir / f"{current_class}_{filename}"
            )

            skeleton_path = (
                skeleton_dir / f"{current_class}_{filename}"
            )

            cv2.imwrite(
                str(skeleton_path),
                (final_skeleton * 255).astype(np.uint8),
            )

            cv2.imwrite(
                str(prediction_path),
                (pred_final * 255).astype(np.uint8),
            )

            # -----------------------------------------------
            # E. OFFICIAL METRICS: TTA + POST-PROCESSING
            # -----------------------------------------------

            intersection = np.sum(pred_final * gt_mask)

            dice = (
                2 * intersection + 1e-7
            ) / (
                np.sum(pred_final)
                + np.sum(gt_mask)
                + 1e-7
            )

            cldice = calculate_cldice(
                pred_final,
                gt_mask,
            )

            (
                acc,
                auc,
                g_mean,
                kappa,
                fdr,
            ) = calculate_extended_metrics(
                gt_mask,
                pred_final,
                pred_probs_np,
            )

            for category in [current_class, "Total"]:
                test_results[category]["dice"].append(dice)
                test_results[category]["cldice"].append(cldice)
                test_results[category]["acc"].append(acc)
                test_results[category]["auc"].append(auc)
                test_results[category]["g_mean"].append(g_mean)
                test_results[category]["kappa"].append(kappa)
                test_results[category]["fdr"].append(fdr)
                test_results[category]["time"].append(time_full)

            per_image_results.append(
                {
                    "Filename": filename,
                    "Category": current_class,
                    "Dice_No_TTA": dice_no,
                    "clDice_No_TTA": cldice_no,
                    "Accuracy_No_TTA": acc_no,
                    "AUC_No_TTA": auc_no,
                    "GMean_No_TTA": gm_no,
                    "Kappa_No_TTA": kappa_no,
                    "FDR_No_TTA": fdr_no,
                    "Dice_TTA_PP": dice,
                    "clDice_TTA_PP": cldice,
                    "Accuracy_TTA_PP": acc,
                    "AUC_TTA_PP": auc,
                    "GMean_TTA_PP": g_mean,
                    "Kappa_TTA_PP": kappa,
                    "FDR_TTA_PP": fdr,
                    "Time_No_TTA_s": time_no_tta,
                    "Time_TTA_PP_s": time_full,
                    "Prediction_Path": str(prediction_path),
                    "Skeleton_Path": str(skeleton_path),
                }
            )

            print(
                f"[{i + 1}/{len(test_loader)}] "
                f"{current_class}/{filename} | "
                f"No TTA: {time_no_tta:.4f}s | "
                f"TTA + post-processing: {time_full:.4f}s"
            )

    # --------------------------------------------------------
    # 6. SAVE PER-IMAGE RESULTS
    # --------------------------------------------------------

    per_image_csv = output_dir / "test_metrics_per_image.csv"

    pd.DataFrame(per_image_results).to_csv(
        per_image_csv,
        index=False,
    )

    print(f"\nPer-image metrics saved to: {per_image_csv}")

    # --------------------------------------------------------
    # 7. INFERENCE TIMING SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("AVERAGE INFERENCE TIME PER IMAGE")
    print("=" * 65)

    print(
        "Five-model ensemble, without TTA: "
        f"{np.mean(times_no_tta):.4f} "
        f"± {np.std(times_no_tta):.4f} s"
    )

    print(
        "Five-model ensemble, TTA + post-processing: "
        f"{np.mean(times_full_tta):.4f} "
        f"± {np.std(times_full_tta):.4f} s"
    )

    print(f"Number of test images: {len(times_no_tta)}")

    # --------------------------------------------------------
    # 8. COMPARE WITHOUT TTA VS. TTA + POST-PROCESSING
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("METRIC COMPARISON: WITHOUT TTA VS. TTA + POST-PROCESSING")
    print("=" * 80)

    for category in ["Total", "SVC", "Full"]:
        print(f"\n--- {category.upper()} ---")
        print(
            f"{'Metric':<12} "
            f"{'Without TTA':>14} "
            f"{'TTA + PP':>16}"
        )

        for metric, label in METRIC_NAMES.items():
            no_values = metrics_no_tta[category][metric]
            tta_values = test_results[category][metric]

            if not no_values or not tta_values:
                continue

            print(
                f"{label:<12} "
                f"{np.mean(no_values):>14.4f} "
                f"{np.mean(tta_values):>16.4f}"
            )

    # --------------------------------------------------------
    # 9. FINAL TEST METRICS
    # --------------------------------------------------------

    print_metric_summary(
        test_results,
        ["Total", "SVC", "Full"],
        "FINAL RESULTS — INDEPENDENT ROSE-1 TEST SET",
    )

    summary_rows = []

    for category in ["Total", "SVC", "Full"]:
        row = {"Category": category}

        for metric, label in METRIC_NAMES.items():
            values = np.asarray(
                test_results[category][metric],
                dtype=float,
            )

            row[f"{label}_Mean"] = (
                float(np.nanmean(values))
                if values.size
                else np.nan
            )

            row[f"{label}_Std"] = (
                float(np.nanstd(values))
                if values.size
                else np.nan
            )

        values_time = np.asarray(
            test_results[category]["time"],
            dtype=float,
        )

        row["Time_Per_Image_Mean_s"] = (
            float(np.mean(values_time))
            if values_time.size
            else np.nan
        )

        summary_rows.append(row)

    summary_csv = output_dir / "test_metrics_summary.csv"

    pd.DataFrame(summary_rows).to_csv(
        summary_csv,
        index=False,
    )

    print(f"\nSummary metrics saved to: {summary_csv}")

    # --------------------------------------------------------
    # 10. QUALITATIVE VISUALIZATION
    # --------------------------------------------------------

    print("\nGenerating qualitative test-set visualization...")

    idx_svc = next(
        (
            i
            for i, category in enumerate(test_classes)
            if category == "SVC"
        ),
        None,
    )

    idx_full = next(
        (
            i
            for i, category in enumerate(test_classes)
            if category == "Full"
        ),
        None,
    )

    sample_indices = [
        i for i in [idx_svc, idx_full]
        if i is not None
    ]

    if sample_indices:
        import matplotlib.colors as mcolors

        num_samples = len(sample_indices)

        fig, axes = plt.subplots(
            num_samples,
            6,
            figsize=(25, 5 * num_samples),
        )

        fig.suptitle(
            "Qualitative Evaluation - Ensemble 5 Fold + Final Post-processing",
            fontsize=18,
            weight="bold",
        )

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        cmap_error = mcolors.ListedColormap(
            ["black", "blue", "red", "green"]
        )

        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]

        norm = mcolors.BoundaryNorm(
            bounds,
            cmap_error.N,
        )

        for row, idx in enumerate(sample_indices):
            img_path = test_image_paths[idx]
            gt_path = test_mask_paths[idx]
            current_class = test_classes[idx]
            filename = Path(img_path).name

            pred_path = (
                output_dir / f"{current_class}_{filename}"
            )

            skeleton_path = (
                skeleton_dir / f"{current_class}_{filename}"
            )

            img = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE,
            )

            gt_mask = (
                cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 127
            ).astype(np.uint8)

            pred_mask = (
                cv2.imread(
                    str(pred_path),
                    cv2.IMREAD_GRAYSCALE,
                ) > 127
            ).astype(np.uint8)

            pred_skeleton = (
                cv2.imread(
                    str(skeleton_path),
                    cv2.IMREAD_GRAYSCALE,
                ) > 127
            ).astype(np.uint8)

            gt_skeleton = skeletonize(gt_mask > 0)

            error_map = np.zeros_like(
                gt_mask,
                dtype=np.uint8,
            )

            # True positives
            error_map[
                (gt_mask == 1) & (pred_mask == 1)
            ] = 3

            # False positives
            error_map[
                (gt_mask == 0) & (pred_mask == 1)
            ] = 2

            # False negatives
            error_map[
                (gt_mask == 1) & (pred_mask == 0)
            ] = 1

            panels = [
                (img, "OCTA Input", "gray"),
                (gt_mask, "Ground Truth", "gray"),
                (pred_mask, "Final Prediction", "gray"),
                (error_map, "Error Map", "error"),
                (gt_skeleton, "GT Skeleton", "gray"),
                (pred_skeleton, "Pred Skeleton", "gray"),
            ]

            for col, (data, title, cmap) in enumerate(panels):
                ax = axes[row, col]

                if cmap == "error":
                    ax.imshow(
                        data,
                        cmap=cmap_error,
                        norm=norm,
                    )
                else:
                    ax.imshow(data, cmap="gray")

                ax.set_title(title, fontsize=12)
                ax.axis("off")

        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        qualitative_path = (
            output_dir / "qualitative_test_results.png"
        )

        plt.savefig(
            qualitative_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Qualitative figure saved to: {qualitative_path}")

    else:
        print("No test images found for qualitative visualization.")

    # --------------------------------------------------------
    # 11. INDIVIDUAL FOLD MODELS: WITHOUT TTA / WITHOUT PP
    # --------------------------------------------------------

    single_fold_results = {
        category: {
            "DSC": [],
            "clDice": [],
            "AUC": [],
        }
        for category in ["SVC", "Full"]
    }

    for fold_idx, model in enumerate(models):
        print(
            f"\nEvaluating individual model Fold {fold_idx + 1}/5"
        )

        fold_metrics = {
            category: {
                "DSC": [],
                "clDice": [],
                "AUC": [],
            }
            for category in ["SVC", "Full"]
        }

        model.eval()

        with torch.no_grad():
            for i, (images, masks) in enumerate(test_loader):
                images = images.to(device)

                gt_mask = (
                    masks.cpu().numpy()[0, 0].astype(np.uint8)
                )

                category = test_classes[i]

                pred_probs = torch.sigmoid(
                    model(images)
                ).cpu().numpy()[0, 0]

                pred_mask = (
                    pred_probs > 0.5
                ).astype(np.uint8)

                intersection = np.sum(
                    pred_mask * gt_mask
                )

                dsc = (
                    2 * intersection + 1e-7
                ) / (
                    np.sum(pred_mask)
                    + np.sum(gt_mask)
                    + 1e-7
                )

                cldice = calculate_cldice(
                    pred_mask,
                    gt_mask,
                )

                try:
                    auc = roc_auc_score(
                        gt_mask.flatten(),
                        pred_probs.flatten(),
                    )
                except ValueError:
                    auc = np.nan

                fold_metrics[category]["DSC"].append(dsc)
                fold_metrics[category]["clDice"].append(cldice)
                fold_metrics[category]["AUC"].append(auc)

        for category in ["SVC", "Full"]:
            for metric in ["DSC", "clDice", "AUC"]:
                values = fold_metrics[category][metric]

                single_fold_results[category][metric].append(
                    np.nanmean(values) if values else np.nan
                )

    print("\n" + "=" * 65)
    print("INDIVIDUAL MODELS — MEAN AND STD ACROSS FIVE FOLDS")
    print("=" * 65)

    single_model_rows = []

    for category in ["SVC", "Full"]:
        print(f"\n--- {category} ---")

        row = {"Category": category}

        for metric in ["DSC", "clDice", "AUC"]:
            values = np.asarray(
                single_fold_results[category][metric],
                dtype=float,
            )

            mean_value = (
                float(np.nanmean(values))
                if values.size
                else np.nan
            )

            std_value = (
                float(np.nanstd(values, ddof=1))
                if np.count_nonzero(np.isfinite(values)) > 1
                else np.nan
            )

            row[f"{metric}_Mean"] = mean_value
            row[f"{metric}_Std"] = std_value

            print(
                f"{metric:<8}: {mean_value:.4f} "
                f"± {std_value:.4f}"
            )

        single_model_rows.append(row)

    single_models_csv = (
        output_dir / "single_fold_model_metrics.csv"
    )

    pd.DataFrame(single_model_rows).to_csv(
        single_models_csv,
        index=False,
    )

    print(f"\nIndividual-fold metrics saved to: {single_models_csv}")

    # --------------------------------------------------------
    # 12. CLEANUP
    # --------------------------------------------------------

    del models
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nTest-set evaluation completed.")

    return {
        "test_results": test_results,
        "metrics_no_tta": metrics_no_tta,
        "single_fold_results": single_fold_results,
        "per_image_csv": str(per_image_csv),
        "summary_csv": str(summary_csv),
        "single_models_csv": str(single_models_csv),
        "output_dir": str(output_dir),
        "skeleton_dir": str(skeleton_dir),
    }
