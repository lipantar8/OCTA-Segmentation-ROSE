from pathlib import Path
import torch


# Project paths
ROOT_PATH = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_PATH / "data"
RESULTS_PATH = ROOT_PATH / "results"
FIGURES_PATH = ROOT_PATH / "figures"
CHECKPOINTS_PATH = ROOT_PATH / "checkpoints"


# Image settings
TARGET_RES = 304


# Training settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4

NUM_WORKERS = 2


# Reproducibility
RANDOM_SEED = 42