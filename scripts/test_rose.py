
from pathlib import Path

from src.evaluation.test_evaluation import run_test_evaluation
from src.utils.config import DEVICE


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Update these paths to match the locations of your local files.
DATASET_ROOT = PROJECT_ROOT / "data" / "ROSE" / "ROSE-1"

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "final_models_rose"

# Test images and corresponding ground-truth masks.
TEST_SVC_IMAGES = DATASET_ROOT / "SVC" / "test" / "img"
TEST_SVC_MASKS = DATASET_ROOT / "SVC" / "test" / "gt"

TEST_FULL_IMAGES = DATASET_ROOT / "SVC_DVC" / "test" / "img"
TEST_FULL_MASKS = DATASET_ROOT / "SVC_DVC" / "test" / "gt"

RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
PREDICTIONS_DIR = RESULTS_DIR / "test_predictions"
SKELETONS_DIR = RESULTS_DIR / "test_skeletons"


# ============================================================
# VALIDATION
# ============================================================

def validate_paths():
    """Check that the dataset and model checkpoints are available."""

    required_directories = {
        "SVC test images": TEST_SVC_IMAGES,
        "SVC test masks": TEST_SVC_MASKS,
        "Full test images": TEST_FULL_IMAGES,
        "Full test masks": TEST_FULL_MASKS,
        "Checkpoint directory": CHECKPOINT_DIR,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required_directories.items()
        if not path.is_dir()
    ]

    if missing:
        message = (
            "The following directories could not be found:\n"
            + "\n".join(missing)
            + "\n\nUpdate the paths in scripts/test_rose.py."
        )
        raise FileNotFoundError(message)

    required_checkpoints = [
        CHECKPOINT_DIR / f"final_model_ROSE_fold{i}.pth"
        for i in range(1, 6)
    ]

    missing_checkpoints = [
        str(path)
        for path in required_checkpoints
        if not path.is_file()
    ]

    if missing_checkpoints:
        raise FileNotFoundError(
            "The following model checkpoints are missing:\n"
            + "\n".join(missing_checkpoints)
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("OCTA ROSE - FINAL TEST EVALUATION")
    print("=" * 80)

    print(f"Device: {DEVICE}")
    print(f"Dataset directory: {DATASET_ROOT}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Results directory: {RESULTS_DIR}")

    validate_paths()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    SKELETONS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStarting final test evaluation...")

    results = run_test_evaluation(
        test_svc_images=TEST_SVC_IMAGES,
        test_svc_masks=TEST_SVC_MASKS,
        test_full_images=TEST_FULL_IMAGES,
        test_full_masks=TEST_FULL_MASKS,
        checkpoint_dir=CHECKPOINT_DIR,
        output_dir=PREDICTIONS_DIR,
        skeleton_dir=SKELETONS_DIR,
        device=DEVICE,
    )

    print("\nTest evaluation completed.")
    print(f"Predictions: {PREDICTIONS_DIR}")
    print(f"Skeletons:   {SKELETONS_DIR}")

    if results is not None:
        print("\nReturned evaluation results:")
        print(results)


if __name__ == "__main__":
    main()
