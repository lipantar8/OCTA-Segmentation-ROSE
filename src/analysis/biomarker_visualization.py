
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_agreement_and_errors(df, output_dir):
    """Plot biomarker agreement and stratified relative errors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    # Scatter plots: predicted values versus ground truth.
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    biomarkers = [
        ("VD_GT", "VD_Pred", "Vessel Density"),
        ("SL_GT", "SL_Pred", "Skeleton Length"),
        ("FD_GT", "FD_Pred", "Fractal Dimension"),
        ("TORT_GT", "TORT_Pred", "Vessel Tortuosity"),
    ]

    for i, (gt_col, pred_col, title) in enumerate(biomarkers):
        sns.scatterplot(
            data=df,
            x=gt_col,
            y=pred_col,
            hue="Category",
            style="Category",
            palette=["#E63946", "#1D3557"],
            s=90,
            alpha=0.8,
            ax=axes[i],
        )

        min_v = min(df[gt_col].min(), df[pred_col].min())
        max_v = max(df[gt_col].max(), df[pred_col].max())
        margin = (max_v - min_v) * 0.05

        axes[i].plot(
            [min_v - margin, max_v + margin],
            [min_v - margin, max_v + margin],
            "k--",
            lw=2,
            label="Perfect Agreement",
        )

        axes[i].set_title(title, fontsize=14, weight="bold")
        axes[i].set_xlabel("Ground Truth (Medici)")
        axes[i].set_ylabel("Predicted by Model" if i == 0 else "")
        axes[i].legend(title="Stratification", frameon=True)

    plt.tight_layout()
    plt.savefig(
        output_dir / "plot_1_scatter_agreement.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Relative-error boxplots with independent y-axis scales.
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        "Distribution of Relative Error by Biomarker (Independent Scales)",
        fontsize=18,
        weight="bold",
        y=1.05,
    )

    biomarkers_box = [
        ("VD_Error_%", "Vessel Density Error"),
        ("SL_Error_%", "Skeleton Length Error"),
        ("FD_Error_%", "Fractal Dimension Error"),
        ("TORT_Error_%", "Vessel Tortuosity Error"),
    ]

    palette_colors = {"SVC": "#1D3557", "Full": "#E63946"}

    for i, (col, title) in enumerate(biomarkers_box):
        sns.boxplot(
            data=df,
            x="Category",
            y=col,
            hue="Category",
            palette=palette_colors,
            ax=axes[i],
            showfliers=True,
            legend=False,
        )

        axes[i].set_title(title, fontsize=14, weight="bold")
        axes[i].set_ylabel("Error Relative to GT (%)", fontsize=12)
        axes[i].set_xlabel("")
        axes[i].yaxis.grid(True, linestyle="--", alpha=0.7)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette_colors[category])
        for category in ["SVC", "Full"]
    ]

    fig.legend(
        handles,
        ["SVC", "Full"],
        title="Retinal Layer",
        loc="center right",
        bbox_to_anchor=(1.05, 0.5),
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "plot_2_boxplot_stratified.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_metric_error_relationships(
    df,
    metric,
    label_name,
    line_color,
    filename,
    output_dir,
):
    """Plot segmentation metrics against relative biomarker errors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    biomarkers = [
        ("VD_Error_%", "Vessel Density"),
        ("SL_Error_%", "Skeleton Length"),
        ("FD_Error_%", "Fractal Dimension"),
        ("TORT_Error_%", "Vessel Tortuosity"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (error_col, title) in zip(axes.flatten(), biomarkers):
        sns.regplot(
            data=df,
            x=metric,
            y=error_col,
            scatter_kws={"s": 70, "alpha": 0.8},
            line_kws={"color": line_color, "lw": 2},
            ax=ax,
        )

        from scipy.stats import pearsonr

        r, p = pearsonr(df[metric], df[error_col])

        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel(label_name)
        ax.set_ylabel("Relative Error (%)")

        ax.text(
            0.04,
            0.95,
            f"r = {r:.2f}\np = {p:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox={
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "gray",
            },
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        output_dir / f"{filename}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_bland_altman(df, output_dir):
    """Generate Bland-Altman plots for the four biomarkers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        "Bland-Altman Plots (Error vs Ground Truth)",
        fontsize=18,
        weight="bold",
        y=1.05,
    )

    biomarkers = [
        ("VD_GT", "VD_Pred", "Vessel Density"),
        ("SL_GT", "SL_Pred", "Skeleton Length"),
        ("FD_GT", "FD_Pred", "Fractal Dimension"),
        ("TORT_GT", "TORT_Pred", "Vessel Tortuosity"),
    ]

    for i, (gt_col, pred_col, title) in enumerate(biomarkers):
        ax = axes[i]
        diffs = df[pred_col] - df[gt_col]

        plot_df = df.copy()
        plot_df["BA_X"] = df[gt_col]
        plot_df["BA_Diff"] = diffs

        sns.scatterplot(
            data=plot_df,
            x="BA_X",
            y="BA_Diff",
            hue="Category",
            style="Category",
            palette={"SVC": "#1D3557", "Full": "#E63946"},
            s=70,
            alpha=0.8,
            ax=ax,
            legend=False,
        )

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        ax.axhline(
            mean_diff,
            color="black",
            linestyle="-",
            lw=2,
            label=f"Mean Bias: {mean_diff:.3f}",
        )
        ax.axhline(
            upper_loa,
            color="red",
            linestyle="--",
            lw=2,
            label=f"+1.96 SD: {upper_loa:.3f}",
        )
        ax.axhline(
            lower_loa,
            color="red",
            linestyle="--",
            lw=2,
            label=f"-1.96 SD: {lower_loa:.3f}",
        )

        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlabel(f"Ground Truth ({title})")
        ax.set_ylabel("Difference (Prediction - GT)" if i == 0 else "")
        ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    plt.savefig(
        output_dir / "plot_bland_altman.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_topological_overlap(vis_data, output_dir):
    """Plot input images, ground-truth skeletons and predicted skeletons."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_svc = next(
        (item for item in vis_data if item[3].lower() == "svc"),
        None,
    )
    sample_full = next(
        (item for item in vis_data if item[3].lower() == "full"),
        None,
    )

    selected_samples = [
        sample for sample in [sample_svc, sample_full]
        if sample is not None
    ]

    if not selected_samples:
        print("No SVC or Full samples found for topological visualization.")
        return

    fig, axes = plt.subplots(
        len(selected_samples),
        4,
        figsize=(20, 5 * len(selected_samples)),
    )
    fig.suptitle("Topological Analysis", fontsize=18, weight="bold", y=0.98)

    if len(selected_samples) == 1:
        axes = axes.reshape(1, -1)

    for row, (img_path, skel_pred, skel_gt, category) in enumerate(
        selected_samples
    ):
        original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if original_image is None:
            raise ValueError(f"Could not read image: {img_path}")

        overlap_rgb = np.zeros(
            (skel_gt.shape[0], skel_gt.shape[1], 3),
            dtype=np.uint8,
        )

        overlap_rgb[skel_gt > 0] = [255, 0, 0]
        overlap_rgb[skel_pred > 0] = [0, 255, 0]
        overlap_rgb[(skel_gt > 0) & (skel_pred > 0)] = [255, 255, 0]

        axes[row, 0].imshow(original_image, cmap="gray")
        axes[row, 0].set_title(f"OCTA Input ({category})", fontsize=14)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(skel_gt, cmap="gray")
        axes[row, 1].set_title("GT Skeleton", fontsize=14)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(skel_pred, cmap="gray")
        axes[row, 2].set_title("Predicted Skeleton", fontsize=14)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(overlap_rgb)
        axes[row, 3].set_title(
            "Overlap (Red=GT, Green=Pred, Yellow=Match)",
            fontsize=14,
        )
        axes[row, 3].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        output_dir / "plot_4_topological_overlap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
