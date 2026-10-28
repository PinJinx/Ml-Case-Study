"""
Cross-dataset comparative visualizations and team aggregation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_team_models_summary(results_df: pd.DataFrame, save_path: str, method: str = "none"):
    """Plots a bar chart comparing models across all members."""
    df = results_df[results_df["imbalance_method"] == method].copy()
    if df.empty:
        df = results_df.copy()

    # Filter out baseline for clean visualization of tested models
    candidates = df[df["algorithm"] != "majority_baseline"].copy()
    candidates["Algorithm"] = candidates["algorithm"].str.replace("_", " ").str.title()

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=candidates,
        x="member",
        y="macro_f1",
        hue="Algorithm",
        palette="Set2"
    )
    plt.title(f"Team Benchmark: Clubbed Models (Macro F1, Imbalance: {method})", fontsize=12, weight="bold")
    plt.xlabel("Team Member")
    plt.ylabel("Test Macro F1")
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h + 0.02),
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cross_dataset_heatmap(results_df: pd.DataFrame, save_path: str, method: str = "none"):
    """Generates a heatmap of Macro F1 across algorithms and datasets."""
    df = results_df[results_df["imbalance_method"] == method].copy()
    if df.empty:
        df = results_df.copy()

    pivot = df.pivot_table(index="algorithm", columns="dataset", values="macro_f1", aggfunc="mean")
    annot = pivot.map(lambda v: f"{v:.3f}" if pd.notnull(v) else "-")

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=annot, fmt="", cmap="YlGnBu", cbar_kws={"label": "Macro F1"})
    plt.title(f"Cross-Dataset Benchmark (Macro F1, Imbalance: {method})", fontsize=12, weight="bold")
    plt.ylabel("Algorithm")
    plt.xlabel("Dataset")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ablation_comparison(ablation_results, save_path: str):
    """Plots the target leakage ablation study (with vs without PM2.5/PM10)."""
    df = pd.DataFrame(ablation_results)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="algorithm", y="macro_f1", hue="feature_set", palette="crest")
    plt.title("Target Leakage Ablation Study (India Dataset)", fontsize=12, weight="bold")
    plt.ylabel("Test Macro F1")
    plt.xlabel("Algorithm")
    plt.xticks(ticks=range(len(df["algorithm"].unique())),
               labels=[a.replace("_", " ").title() for a in df["algorithm"].unique()])
    plt.legend(title="Configuration")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
