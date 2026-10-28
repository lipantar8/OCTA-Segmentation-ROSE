
from pathlib import Path

from src.analysis.biomarker_extraction import (
    extract_biomarkers,
    save_biomarker_results,
)
from src.analysis.biomarker_visualization import (
    plot_agreement_and_errors,
    plot_metric_error_relationships,
    plot_bland_altman,
    plot_topological_overlap,
)
from src.analysis.biomarker_correlation import (
    run_correlation_analysis,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "ROSE" / "ROSE-1"

PRED_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "test_predictions"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "phase4_results"

GT_PATHS = {
    "SVC_thick": DATASET_ROOT / "SVC" / "test" / "gt",
    "SVC_thin": DATASET_ROOT / "SVC" / "test" / "thin_gt",
    "Full_thick": DATASET_ROOT / "SVC_DVC" / "test" / "gt",
    "Full_thin": DATASET_ROOT / "SVC_DVC" / "test" / "thin_gt",
}

IMG_PATHS = {
    "SVC": DATASET_ROOT / "SVC" / "test" / "img",
    "Full": DATASET_ROOT / "SVC_DVC" / "test" / "img",
}


# ============================================================
# VALIDATION
# ============================================================

def validate_paths():
    """Check that all required input directories exist."""
    required_directories = {
        "Prediction directory": PRED_DIR,
        **GT_PATHS,
        **IMG_PATHS,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required_directories.items()
        if not path.is_dir()
    ]

    if missing:
        raise FileNotFoundError(
            "The following directories could not be found:\n"
            + "\n".join(missing)
            + "\n\nUpdate the paths in scripts/analyze_biomarkers.py."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("OCTA ROSE - BIOMARKER ANALYSIS")
    print("=" * 80)

    print(f"Prediction directory: {PRED_DIR}")
    print(f"Dataset directory:    {DATASET_ROOT}")
    print(f"Results directory:    {RESULTS_DIR}")

    validate_paths()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Extract biomarkers and segmentation metrics.
    df_clinical, vis_data = extract_biomarkers(
        pred_dir=PRED_DIR,
        gt_paths=GT_PATHS,
        img_paths=IMG_PATHS,
    )

    # 2. Save the numerical results.
    save_biomarker_results(
        df_clinical=df_clinical,
        output_dir=RESULTS_DIR,
    )

    # 3. Generate agreement plots and relative-error boxplots.
    plot_agreement_and_errors(
        df=df_clinical,
        output_dir=RESULTS_DIR,
    )

    # 4. Analyze Dice and clDice in relation to biomarker errors.
    plot_metric_error_relationships(
        df=df_clinical,
        metric="Dice",
        label_name="Dice",
        line_color="red",
        filename="plot_dice_vs_biomarker_error",
        output_dir=RESULTS_DIR,
    )

    plot_metric_error_relationships(
        df=df_clinical,
        metric="clDice",
        label_name="clDice",
        line_color="darkred",
        filename="plot_cldice_vs_biomarker_error",
        output_dir=RESULTS_DIR,
    )

    # 5. Generate Bland-Altman and topological visualizations.
    plot_bland_altman(
        df=df_clinical,
        output_dir=RESULTS_DIR,
    )

    plot_topological_overlap(
        vis_data=vis_data,
        output_dir=RESULTS_DIR,
    )

    # 6. Calculate correlations and rank the biomarkers.
    run_correlation_analysis(
        df=df_clinical,
        output_dir=RESULTS_DIR,
    )

    print("\n" + "=" * 80)
    print("BIOMARKER ANALYSIS COMPLETED")
    print(f"All results saved in: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
