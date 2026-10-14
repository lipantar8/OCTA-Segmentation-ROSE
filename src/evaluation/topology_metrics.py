import itertools

import cv2
import numpy as np
import torch

from torch.utils.data import DataLoader

from skimage.morphology import (
    skeletonize,
    remove_small_objects
)

from src.datasets.segmentation_dataset import SegmentationDataset

def get_pp_configs():

    morph_ops = [
        None,
        "closing",
        "opening"
    ]

    kernel_sizes = [
        3,
        5
    ]

    smooth_ops = [
        None,
        "median",
        "gaussian"
    ]

    min_areas = [
        0,
        15,
        30,
        50
    ]

    configs = []

    for m, k, s, a in itertools.product(
        morph_ops,
        kernel_sizes,
        smooth_ops,
        min_areas
    ):

        name = (
            f"{m if m else 'Raw'}"
            f"_k{k}_"
            f"{s if s else 'none'}_"
            f"area{a}"
        )

        configs.append(
            {
                "name": name,
                "morph_op": m,
                "kernel_size": k,
                "smooth": s,
                "min_area": a
            }
        )

    return configs

def calculate_cldice(v_p, v_l):
    """
    Compute clDice between prediction and ground truth masks.

    Args:
        v_p: Binary predicted vessel mask.
        v_l: Binary ground truth vessel mask.

    Returns:
        clDice score.
    """

    v_p_bool = v_p > 0.5
    v_l_bool = v_l > 0.5


    skeleton_pred = skeletonize(
        v_p_bool
    ).astype(np.float32)

    skeleton_gt = skeletonize(
        v_l_bool
    ).astype(np.float32)


    pred_float = v_p_bool.astype(
        np.float32
    )

    gt_float = v_l_bool.astype(
        np.float32
    )


    topology_precision = (
        np.sum(skeleton_pred * gt_float)
        /
        (np.sum(skeleton_pred) + 1e-7)
    )


    topology_sensitivity = (
        np.sum(skeleton_gt * pred_float)
        /
        (np.sum(skeleton_gt) + 1e-7)
    )


    cldice = (
        2
        * topology_precision
        * topology_sensitivity
        /
        (
            topology_precision
            +
            topology_sensitivity
            +
            1e-7
        )
    )


    return cldice



def apply_topological_postprocessing(
    pred_mask,
    morph_op="closing",
    kernel_size=3,
    smooth="median",
    min_area=30
):
    """
    Apply topology-oriented post-processing
    and return skeletonized mask.
    """

    mask_uint8 = (
        pred_mask * 255
    ).astype(np.uint8)


    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (kernel_size, kernel_size)
    )


    if morph_op == "opening":

        processed = cv2.morphologyEx(
            mask_uint8,
            cv2.MORPH_OPEN,
            kernel
        )

    elif morph_op == "closing":

        processed = cv2.morphologyEx(
            mask_uint8,
            cv2.MORPH_CLOSE,
            kernel
        )

    elif morph_op == "erosion":

        processed = cv2.erode(
            mask_uint8,
            kernel,
            iterations=1
        )

    else:

        processed = mask_uint8.copy()


    if smooth == "median":

        processed = cv2.medianBlur(
            processed,
            3
        )

    elif smooth == "gaussian":

        processed = cv2.GaussianBlur(
            processed,
            (3, 3),
            0
        )


    _, binary_mask = cv2.threshold(
        processed,
        127,
        1,
        cv2.THRESH_BINARY
    )


    binary_mask = binary_mask.astype(
        bool
    )


    if min_area > 0:

        binary_mask = remove_small_objects(
            binary_mask,
            min_size=min_area
        )


    return skeletonize(
        binary_mask
    ).astype(np.float32)


def apply_postprocessing_advanced(
    pred_mask,
    morph_op="opening",
    kernel_size=3,
    smooth="median",
    min_area=0,
    do_skeleton=False
):
    """
    Apply post-processing optimized for volumetric segmentation metrics.

    Args:
        pred_mask: Binary predicted mask.
        morph_op: Morphological operation.
        kernel_size: Kernel size.
        smooth: Optional smoothing operation.
        min_area: Minimum object size.
        do_skeleton: Return skeleton instead of binary mask.

    Returns:
        Processed binary mask.
    """

    mask_uint8 = (
        pred_mask * 255
    ).astype(np.uint8)


    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )


    if morph_op == "opening":

        processed = cv2.morphologyEx(
            mask_uint8,
            cv2.MORPH_OPEN,
            kernel
        )

    elif morph_op == "closing":

        processed = cv2.morphologyEx(
            mask_uint8,
            cv2.MORPH_CLOSE,
            kernel
        )

    elif morph_op == "erosion":

        processed = cv2.erode(
            mask_uint8,
            kernel,
            iterations=1
        )

    else:

        processed = mask_uint8.copy()


    if smooth == "median":

        processed = cv2.medianBlur(
            processed,
            3
        )

    elif smooth == "gaussian":

        processed = cv2.GaussianBlur(
            processed,
            (3, 3),
            0
        )


    _, binary_mask = cv2.threshold(
        processed,
        127,
        1,
        cv2.THRESH_BINARY
    )


    binary_mask = binary_mask.astype(bool)


    if min_area > 0:

        binary_mask = remove_small_objects(
            binary_mask,
            min_size=min_area
        )


    if do_skeleton:

        return skeletonize(
            binary_mask
        ).astype(np.float32)


    return binary_mask.astype(
        np.float32
    )



def evaluate_topology_thick(
    model,
    df_val,
    device,
    transform
):
    """
    Evaluate topology-aware performance using clDice
    and post-processing grid search.
    """

    model.eval()


    dataset = SegmentationDataset(
        image_paths=df_val["image_path"].tolist(),
        mask_paths=df_val["mask_path"].tolist(),
        transform=transform
    )


    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )


    configs = get_pp_configs()


    scores = {
        i: []
        for i in range(len(configs))
    }


    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)


            outputs = model(images)


            predictions = (
                torch.sigmoid(outputs) > 0.5
            ).float().cpu().numpy()


            masks = masks.cpu().numpy()


            for b in range(
                predictions.shape[0]
            ):

                prediction = predictions[b, 0]
                ground_truth = masks[b, 0]


                for i, cfg in enumerate(configs):

                    processed_prediction = (
                        apply_postprocessing_advanced(
                            prediction,
                            morph_op=cfg["morph_op"],
                            kernel_size=cfg["kernel_size"],
                            smooth=cfg["smooth"],
                            min_area=cfg["min_area"]
                        )
                    )


                    score = calculate_cldice(
                        processed_prediction,
                        ground_truth
                    )


                    scores[i].append(
                        score
                    )


    best_config = None
    best_score = -1


    for i, cfg in enumerate(configs):

        mean_score = np.mean(
            scores[i]
        )


        if mean_score > best_score:

            best_score = mean_score
            best_config = cfg


    return (
        best_config,
        best_score,
        scores,
        configs
    )