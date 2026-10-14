import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


class OCTA500Dataset(Dataset):
    """
    Dataset class for OCTA500 retinal vessel segmentation.

    The dataset combines large vessel and capillary annotations
    into a single binary vessel mask.
    """

    def __init__(
        self,
        image_paths,
        large_vessel_paths,
        capillary_paths,
        transform=None
    ):
        self.image_paths = image_paths
        self.large_vessel_paths = large_vessel_paths
        self.large_vessel_paths = large_vessel_paths
        self.capillary_paths = capillary_paths
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):

        # Load image
        image = cv2.imread(
            self.image_paths[idx]
        )

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {self.image_paths[idx]}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # Percentile normalization
        image_float = image.astype(
            np.float32
        )

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


        # CLAHE enhancement
        image_uint8 = (
            image_norm * 255
        ).astype(np.uint8)

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


        # Load vessel annotations
        mask_lv = cv2.imread(
            self.large_vessel_paths[idx],
            cv2.IMREAD_GRAYSCALE
        )

        mask_cap = cv2.imread(
            self.capillary_paths[idx],
            cv2.IMREAD_GRAYSCALE
        )


        # Merge vessel annotations
        mask_combined = np.maximum(
            mask_lv,
            mask_cap
        )

        mask = (
            mask_combined > 127
        ).astype(np.float32)


        # Apply augmentation
        if self.transform:

            augmented = self.transform(
                image=image_final,
                mask=mask
            )

            image_final = augmented["image"]
            mask = augmented["mask"]


        # Convert image to tensor
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


        # Convert mask to tensor
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