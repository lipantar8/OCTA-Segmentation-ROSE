from pathlib import Path
from glob import glob

import pandas as pd


def load_data_to_lists(
    img_dir,
    mask_dir,
    class_name,
    thin_dir=None
):
    """
    Load image-mask pairs from a dataset folder.

    Args:
        img_dir: Directory containing images.
        mask_dir: Directory containing segmentation masks.
        class_name: Dataset/class identifier.
        thin_dir: Optional directory containing thin vessel masks.

    Returns:
        Dictionary containing image paths, mask paths,
        thin mask paths and class labels.
    """

    img_map = {
        Path(f).stem: f
        for f in glob(str(Path(img_dir) / "*"))
    }

    mask_map = {
        Path(f).stem: f
        for f in glob(str(Path(mask_dir) / "*"))
    }


    thin_map = {}

    if thin_dir is not None:
        thin_map = {
            Path(f).stem: f
            for f in glob(str(Path(thin_dir) / "*"))
        }


    common_names = sorted(
        set(img_map.keys())
        &
        set(mask_map.keys())
    )


    image_paths = []
    mask_paths = []
    thin_mask_paths = []
    classes = []


    for name in common_names:

        image_paths.append(
            img_map[name]
        )

        mask_paths.append(
            mask_map[name]
        )

        classes.append(
            class_name
        )


        if name in thin_map:
            thin_mask_paths.append(
                thin_map[name]
            )
        else:
            thin_mask_paths.append(
                None
            )


    return {
        "image_paths": image_paths,
        "mask_paths": mask_paths,
        "thin_mask_paths": thin_mask_paths,
        "classes": classes
    }



def load_octa500_to_df(
    img_dir,
    lv_dir,
    cap_dir
):
    """
    Load OCTA500 images and merge large vessel
    and capillary annotations.

    Returns:
        DataFrame containing image and mask paths.
    """

    img_map = {
        Path(f).stem: f
        for f in glob(str(Path(img_dir) / "*"))
    }

    lv_map = {
        Path(f).stem: f
        for f in glob(str(Path(lv_dir) / "*"))
    }

    cap_map = {
        Path(f).stem: f
        for f in glob(str(Path(cap_dir) / "*"))
    }


    common_ids = sorted(
        set(img_map.keys())
        &
        set(lv_map.keys())
        &
        set(cap_map.keys())
    )


    return pd.DataFrame(
        {
            "image_path":
                [img_map[i] for i in common_ids],

            "lv_path":
                [lv_map[i] for i in common_ids],

            "cap_path":
                [cap_map[i] for i in common_ids],

            "class":
                ["OCTA-500"] * len(common_ids)
        }
    )
