from src.evaluation.topology_metrics import calculate_cldice
import numpy as np
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


def prune_skeleton(skeleton_mask, iterations=1):
    """Remove skeleton endpoints for the specified number of iterations."""
    skel_pruned = (skeleton_mask > 0).astype(np.uint8)

    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])

    for _ in range(iterations):
        neighbors = convolve(
            skel_pruned,
            kernel,
            mode="constant",
            cval=0,
        )
        endpoints = (skel_pruned == 1) & (neighbors == 1)
        skel_pruned[endpoints] = 0

    return skel_pruned


def fractal_dimension_boxcounting(Z, threshold=0.5):
    """Estimate fractal dimension using the box-counting method."""
    Z = Z > threshold

    if len(Z.shape) != 2 or not np.any(Z):
        return 0.0

    p = min(Z.shape)
    n = int(2 ** np.floor(np.log(p) / np.log(2)))
    Z = Z[0:n, 0:n]

    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
    counts = []

    for size in sizes:
        shape = (
            size,
            Z.shape[0] // size,
            size,
            Z.shape[1] // size,
        )

        Z_reshaped = Z.reshape(shape)
        Z_boxes = Z_reshaped.max(axis=(0, 2))
        counts.append(np.sum(Z_boxes))

    if len(counts) > 1 and np.count_nonzero(counts) > 1:
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    return 0.0


def _ordered_branch(coords):
    """
    Order the pixels of a single skeleton branch
    using 8-neighbour connectivity.
    """
    coords = [tuple(c) for c in coords]
    coord_set = set(coords)

    # Build the adjacency graph.
    graph = {}

    for p in coords:
        neighbours = []

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue

                q = (p[0] + dr, p[1] + dc)

                if q in coord_set:
                    neighbours.append(q)

        graph[p] = neighbours

    # An endpoint has degree 1.
    endpoints = [
        p for p, neighbours in graph.items()
        if len(neighbours) == 1
    ]

    if len(endpoints) >= 2:
        start = endpoints[0]
    else:
        # For a small loop, start from an arbitrary point.
        start = coords[0]

    ordered = [start]
    visited = {start}

    current = start
    previous = None

    while True:
        candidates = [
            node for node in graph[current]
            if node != previous
        ]

        next_node = None

        for candidate in candidates:
            if candidate not in visited:
                next_node = candidate
                break

        if next_node is None:
            break

        ordered.append(next_node)
        visited.add(next_node)

        previous = current
        current = next_node

    return np.array(ordered)


def calculate_mean_tortuosity(
    skeleton_mask,
    min_branch_length=5,
    weighted=True,
):
    """Calculate mean skeleton branch tortuosity."""
    skel = (skeleton_mask > 0).astype(np.uint8)

    # Identify junctions.
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])

    neighbors = convolve(skel, kernel, mode="constant")
    junctions = (skel == 1) & (neighbors >= 3)

    branches = skel.copy()
    branches[junctions] = 0

    labeled = label(branches, connectivity=2)

    tortuosities = []
    weights = []

    for region in regionprops(labeled):
        coords = region.coords

        if len(coords) < min_branch_length:
            continue

        ordered = _ordered_branch(coords)

        if len(ordered) < 2:
            continue

        # Arc length.
        diff = np.diff(ordered, axis=0)
        arc_length = np.sum(
            np.sqrt(np.sum(diff**2, axis=1))
        )

        # Chord length.
        chord_length = np.linalg.norm(
            ordered[0] - ordered[-1]
        )

        if chord_length < 1:
            continue

        tortuosity = arc_length / chord_length

        tortuosities.append(tortuosity)
        weights.append(arc_length)

    if len(tortuosities) == 0:
        return 1.0

    if weighted:
        return np.average(tortuosities, weights=weights)

    return np.mean(tortuosities)


def calculate_biomarkers(pred_mask, gt_thick, gt_thin):
    """
    Calculate vessel density, skeleton length, fractal dimension,
    and vessel tortuosity for a prediction and its ground truths.
    """
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_thick = (gt_thick > 0).astype(np.uint8)
    gt_thin = (gt_thin > 0).astype(np.uint8)

    total_pixels = float(pred_mask.size)

    # Preserve the original notebook's processing:
    # prune the predicted skeleton once; use thin GT directly.
    skel_pred = prune_skeleton(
        skeletonize(pred_mask),
        iterations=1,
    )
    skel_gt = gt_thin

    # Vessel density.
    vd_pred = float(np.sum(pred_mask) / total_pixels)
    vd_gt = float(np.sum(gt_thick) / total_pixels)

    # Fractal dimension.
    fd_pred = float(fractal_dimension_boxcounting(pred_mask))
    fd_gt = float(fractal_dimension_boxcounting(gt_thick))

    # Skeleton length.
    sl_pred = float(np.sum(skel_pred))
    sl_gt = float(np.sum(skel_gt))

    # Vessel tortuosity.
    tort_pred = float(calculate_mean_tortuosity(skel_pred))
    tort_gt = float(calculate_mean_tortuosity(skel_gt))

    return {
        "VD_GT": vd_gt,
        "VD_Pred": vd_pred,
        "VD_Error_%": abs(vd_pred - vd_gt) / (vd_gt + 1e-7) * 100,

        "SL_GT": sl_gt,
        "SL_Pred": sl_pred,
        "SL_Error_%": abs(sl_pred - sl_gt) / (sl_gt + 1e-7) * 100,

        "FD_GT": fd_gt,
        "FD_Pred": fd_pred,
        "FD_Error_%": abs(fd_pred - fd_gt) / (fd_gt + 1e-7) * 100,

        "TORT_GT": tort_gt,
        "TORT_Pred": tort_pred,
        "TORT_Error_%": (
            abs(tort_pred - tort_gt) / (tort_gt + 1e-7) * 100
        ),

        # Retained for the qualitative skeleton visualization.
        "skeleton_pred": skel_pred,
        "skeleton_gt": skel_gt,
    }


def calculate_segmentation_metrics(pred_mask, gt_thick):
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_thick = (gt_thick > 0).astype(np.uint8)

    intersection = np.sum(pred_mask * gt_thick)

    dice = (
        2 * intersection + 1e-7
    ) / (
        pred_mask.sum() + gt_thick.sum() + 1e-7
    )

    cldice = calculate_cldice(pred_mask, gt_thick)

    return {
        "Dice": float(dice),
        "clDice": float(cldice),
    }
