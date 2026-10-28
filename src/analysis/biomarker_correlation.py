
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr


BIOMARKERS = [
    ("VD_Error_%", "Vessel Density"),
    ("SL_Error_%", "Skeleton Length"),
    ("FD_Error_%", "Fractal Dimension"),
    ("TORT_Error_%", "Vessel Tortuosity"),
]


def run_correlation_analysis(df, output_dir):
    """
    Calculate Pearson and Spearman correlations between Dice/clDice
    and relative biomarker estimation errors.

    Save the correlation table, heatmap and clDice-versus-Dice ranking.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    results = []

    for error_col, name in BIOMARKERS:
        # Pearson and Spearman correlations with Dice.
        r_dice, p_dice = pearsonr(df["Dice"], df[error_col])
        rho_dice, sp_dice = spearmanr(df["Dice"], df[error_col])

        # Pearson and Spearman correlations with clDice.
        r_cldice, p_cldice = pearsonr(df["clDice"], df[error_col])
        rho_cldice, sp_cldice = spearmanr(
            df["clDice"],
            df[error_col],
        )

        results.append({
            "Biomarker": name,
            "Pearson Dice": r_dice,
            "Pearson Dice p": p_dice,
            "Pearson clDice": r_cldice,
            "Pearson clDice p": p_cldice,
            "Spearman Dice": rho_dice,
            "Spearman Dice p": sp_dice,
            "Spearman clDice": rho_cldice,
            "Spearman clDice p": sp_cldice,
        })

    corr_df = pd.DataFrame(results)

    print(corr_df)

    # Save correlation results.
    csv_path = output_dir / "biomarker_correlations.csv"
    corr_df.to_csv(csv_path, index=False)

    print(f"\nCorrelation results saved to: {csv_path}")

    # --------------------------------------------------------
    # CORRELATION HEATMAP
    # --------------------------------------------------------

    heat = corr_df.set_index("Biomarker")[
        [
            "Pearson Dice",
            "Pearson clDice",
            "Spearman Dice",
            "Spearman clDice",
        ]
    ]

    plt.figure(figsize=(7, 4))

    sns.heatmap(
        heat,
        annot=True,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        fmt=".2f",
        vmin=-1,
        vmax=1,
    )

    plt.title(
        "Correlation between segmentation metrics\n"
        "and biomarker estimation error",
        fontsize=13,
        weight="bold",
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "heatmap_correlations.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        output_dir / "heatmap_correlations.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # RANKING: ABSOLUTE PEARSON CORRELATIONS
    # --------------------------------------------------------

    ranking = pd.DataFrame({
        "Biomarker": corr_df["Biomarker"],
        "|Pearson Dice|": np.abs(corr_df["Pearson Dice"]),
        "|Pearson clDice|": np.abs(corr_df["Pearson clDice"]),
    })

    ranking["Difference"] = (
        ranking["|Pearson clDice|"]
        - ranking["|Pearson Dice|"]
    )

    ranking = ranking.sort_values(
        "Difference",
        ascending=False,
    )

    print("\n")
    print("=" * 70)
    print("IMPROVEMENT OF clDice OVER DICE")
    print("=" * 70)

    print(ranking)

    ranking_path = output_dir / "ranking_cldice_vs_dice.csv"
    ranking.to_csv(ranking_path, index=False)

    print(f"\nRanking saved to: {ranking_path}")

    return corr_df, ranking
