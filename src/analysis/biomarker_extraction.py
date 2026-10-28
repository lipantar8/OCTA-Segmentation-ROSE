
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.analysis.biomarker_metrics import (
    calculate_biomarkers,
    calculate_segmentation_metrics,
)


def find_matching_file(directory, base_name):
    """Find the first file matching a filename stem in a directory."""
    directory = Path(directory)

    matches = sorted(
        path
        for path in directory.glob(f"{base_name}.*")
        if path.is_file()
    )

    return matches[0] if matches else None


def load_binary_mask(path):
    """Load an image in grayscale and convert it to a binary mask."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return (image > 127).astype(np.uint8)


def extract_biomarkers(pred_dir, gt_paths, img_paths):
    """
    Match predictions with images and ground-truth masks, then calculate
    segmentation metrics and vascular biomarkers.

    Args:
        pred_dir: Directory containing predicted masks named with an
            SVC_ or Full_ prefix.
        gt_paths: Dictionary with SVC_thick, SVC_thin, Full_thick,
            and Full_thin directories.
        img_paths: Dictionary with SVC and Full image directories.

    Returns:
        df_clinical: DataFrame containing segmentation metrics and
            biomarker estimates and errors.
        vis_data: List of tuples for the qualitative skeleton plots.
    """
    pred_dir = Path(pred_dir)

    clinical_data = []
    vis_data = []

    pred_paths = sorted(
        path for path in pred_dir.glob("*.*")
        if path.is_file()
    )

    print(
        f"Extracting biomarkers from {len(pred_paths)} prediction files..."
    )

    for pred_path in pred_paths:
        filename = pred_path.name

        if filename.startswith("SVC_"):
            category = "SVC"
            clean_name = filename[len("SVC_"):]
        elif filename.startswith("Full_"):
            category = "Full"
            clean_name = filename[len("Full_"):]
        else:
            # Ignore files that do not belong to either category.
            continue

        base_name = Path(clean_name).stem

        if category == "SVC":
            thick_dir = gt_paths["SVC_thick"]
            thin_dir = gt_paths["SVC_thin"]
        else:
            thick_dir = gt_paths["Full_thick"]
            thin_dir = gt_paths["Full_thin"]

        thick_path = find_matching_file(thick_dir, base_name)
        thin_path = find_matching_file(thin_dir, base_name)
        image_path = find_matching_file(
            img_paths[category],
            base_name,
        )

        if (
            thick_path is None
            or thin_path is None
            or image_path is None
        ):
            print(
                f"Skipping {filename}: matching image, thick GT, "
                "or thin GT was not found."
            )
            continue

        pred_mask = load_binary_mask(pred_path)
        gt_thick = load_binary_mask(thick_path)
        gt_thin = load_binary_mask(thin_path)

        if (
            pred_mask.shape != gt_thick.shape
            or pred_mask.shape != gt_thin.shape
        ):
            print(
                f"Skipping {filename}: prediction and ground-truth "
                "dimensions do not match."
            )
            continue

        # Calculate segmentation metrics against the thick GT.
        segmentation_metrics = calculate_segmentation_metrics(
            pred_mask,
            gt_thick,
        )

        # Calculate vascular biomarkers.
        biomarker_metrics = calculate_biomarkers(
            pred_mask,
            gt_thick,
            gt_thin,
        )

        # Store only scalar values in the results table.
        record = {
            "Filename": filename,
            "Category": category,
            **segmentation_metrics,
            **{
                key: value
                for key, value in biomarker_metrics.items()
                if key not in {"skeleton_pred", "skeleton_gt"}
            },
        }

        clinical_data.append(record)

        # Keep skeleton arrays for qualitative visualization.
        vis_data.append(
            (
                str(image_path),
                biomarker_metrics["skeleton_pred"],
                biomarker_metrics["skeleton_gt"],
                category,
            )
        )

    if not clinical_data:
        raise RuntimeError(
            "No biomarker records were extracted. Check the prediction "
            "filenames and dataset paths."
        )

    df_clinical = pd.DataFrame(clinical_data)

    print("\nExtracted biomarker data:")
    print(df_clinical.head())
    print(f"\nMatched images: {len(df_clinical)}")
    print("Mean Dice:", df_clinical["Dice"].mean())
    print("Mean clDice:", df_clinical["clDice"].mean())

    return df_clinical, vis_data


def save_biomarker_results(df_clinical, output_dir):
    """Save the extracted biomarker table to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "clinical_biomarkers_results.csv"
    df_clinical.to_csv(csv_path, index=False)

    print(f"Biomarker results saved to: {csv_path}")

    return csv_path
