import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.config import DEVICE, TARGET_RES

from src.models.attention_unet import get_attention_unet

from src.losses.loss import hybrid_loss

from src.datasets.segmentation_dataset import (
    train_transform,
    val_transform
)


print("=" * 50)
print("Repository setup test")
print("=" * 50)


# 1. Device

print(
    f"Device: {DEVICE}"
)


# 2. Model creation

print(
    "\nCreating Attention U-Net..."
)

model = get_attention_unet().to(DEVICE)

print(
    "Model created successfully"
)


# 3. Dummy forward pass

print(
    "\nRunning dummy forward pass..."
)

dummy_input = torch.randn(
    1,
    3,
    TARGET_RES,
    TARGET_RES
).to(DEVICE)


with torch.no_grad():

    output = model(dummy_input)


print(
    f"Input shape:  {dummy_input.shape}"
)

print(
    f"Output shape: {output.shape}"
)


# 4. Loss test

dummy_target = torch.randint(
    0,
    2,
    output.shape
).float().to(DEVICE)


loss = hybrid_loss(
    output,
    dummy_target
)


print(
    f"\nHybrid loss test: {loss.item():.4f}"
)


# 5. Transform test

print(
    "\nTransforms imported correctly"
)


print(
    "\nALL TESTS PASSED"
)