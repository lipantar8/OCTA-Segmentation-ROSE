import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.config import TARGET_RES


class SegmentationDataset(Dataset):
    """
    Dataset class for OCTA vessel segmentation.

    Performs:
    - image loading
    - percentile normalization
    - CLAHE enhancement
    - mask binarization
    - data augmentation
    - conversion to tensors
    """

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Load image
        image = cv2.imread(self.image_paths[idx])

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {self.image_paths[idx]}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # Percentile normalization
        image_float = image.astype(np.float32)

        p_low, p_high = np.percentile(
            image_float,
            (0.5, 99.5)
        )

        image_clipped = np.clip(
            image_float,
            p_low,
            p_high
        )

        image_norm = (
            image_clipped - p_low
        ) / (
            p_high - p_low + 1e-7
        )


        # Convert to uint8 for CLAHE
        image_uint8 = (
            image_norm * 255
        ).astype(np.uint8)


        # CLAHE enhancement
        lab = cv2.cvtColor(
            image_uint8,
            cv2.COLOR_RGB2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        cl = clahe.apply(l)

        image_final = cv2.merge(
            (cl, a, b)
        )

        image_final = cv2.cvtColor(
            image_final,
            cv2.COLOR_LAB2RGB
        )


        # Load mask
        mask = cv2.imread(
            self.mask_paths[idx],
            cv2.IMREAD_GRAYSCALE
        )

        mask = (
            mask > 127
        ).astype(np.float32)


        # Apply augmentation
        if self.transform:

            augmented = self.transform(
                image=image_final,
                mask=mask
            )

            image_final = augmented["image"]
            mask = augmented["mask"]


        # Ensure tensor format
        if torch.is_tensor(image_final):

            image_final = image_final.float()

            if image_final.max() > 1:
                image_final /= 255.0

        else:

            image_final = (
                torch.from_numpy(image_final)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )


        if not torch.is_tensor(mask):

            mask = (
                torch.from_numpy(mask)
                .unsqueeze(0)
                .float()
            )

        else:

            mask = mask.float()

            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)


        return image_final, mask



train_transform = A.Compose([
    A.Resize(
        TARGET_RES,
        TARGET_RES
    ),

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.5),

    A.Rotate(
        limit=35,
        p=0.5
    ),

    A.ElasticTransform(
        alpha=1,
        sigma=50,
        p=0.3
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.3
    ),

    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.2
    ),

    ToTensorV2()
])


val_transform = A.Compose([
    A.Resize(
        TARGET_RES,
        TARGET_RES
    ),

    ToTensorV2()
])