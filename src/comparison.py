"""
comparison.py
Team-Level Comparative Analysis across all four datasets.
Generates cross-dataset heatmaps, rank analyses, imbalance impact charts, and ablation studies.
"""

import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_team_results(results_list: List[Dict[str, Any]], output_csv: str) -> pd.DataFrame:
    """Combine results from all experiments into a single standardized DataFrame and CSV."""
    df = pd.DataFrame(results_list)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Aggregated {len(df)} experimental runs saved to {output_csv}")
    return df


def plot_cross_dataset_heatmap(df: pd.DataFrame, save_path: str, method: str = "none"):
    """
    Generate a heatmap of Test Macro F1 across algorithms (rows) and datasets (columns)
    for a chosen imbalance handling method.
    """
    subset = df[df["imbalance_method"] == method].copy()
    if subset.empty:
        subset = df.copy()

    pivot = subset.pivot_table(index="algorithm", columns="dataset", values="macro_f1", aggfunc="mean")
    
    # Sort algorithms cleanly
    alg_order = [
        "majority_baseline",
        "naive_bayes",
        "logistic_regression",
        "knn",
        "decision_tree",
        "random_forest",
        "svm",
        "xgboost"
    ]
    present_order = [a for a in alg_order if a in pivot.index]
    pivot = pivot.reindex(present_order)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Test Macro F1"},
        linewidths=0.5
    )
    plt.title(f"Cross-Dataset Algorithm Benchmark (Macro F1, Imbalance: {method})", fontsize=13, weight="bold")
    plt.ylabel("Algorithm")
    plt.xlabel("Dataset / Team Member")
    plt.yticks(ticks=[i + 0.5 for i in range(len(present_order))],
               labels=[a.replace("_", " ").title() for a in present_order], rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_average_algorithm_ranks(df: pd.DataFrame, save_path: str):
    """
    Compute and plot average rank of each algorithm across the four datasets.
    Rank 1 is best (highest Macro F1).
    """
    # Filter for standard imbalance method (none or class_weight)
    subset = df[df["imbalance_method"] == "none"].copy()
    if subset.empty:
        subset = df.copy()

    ranks = []
    for dataset, group in subset.groupby("dataset"):
        g = group.copy()
        g["rank"] = g["macro_f1"].rank(ascending=False, method="min")
        ranks.append(g[["algorithm", "rank"]])

    ranks_df = pd.concat(ranks, ignore_index=True)
    avg_ranks = ranks_df.groupby("algorithm")["rank"].mean().sort_values(ascending=True)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=avg_ranks.values, y=avg_ranks.index, palette="Blues_r")
    plt.title("Average Algorithm Rank Across All Four Datasets\n(Rank 1 = Highest Macro F1)", fontsize=13, weight="bold")
    plt.xlabel("Average Rank (Lower is Better)")
    plt.ylabel("Algorithm")
    plt.yticks(ticks=range(len(avg_ranks)), labels=[a.replace("_", " ").title() for a in avg_ranks.index])

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.2f}", (width + 0.1, p.get_y() + p.get_height() / 2.),
                    va="center", fontsize=10)

    plt.xlim(0, max(avg_ranks.values) + 1.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_ablation_comparison(ablation_results: List[Dict[str, Any]], save_path: str):
    """
    Plot India ablation experiment: Full features (with PM2.5/PM10) vs Ablated (without PM2.5/PM10).
    Demonstrates target leakage effect.
    """
    df = pd.DataFrame(ablation_results)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="algorithm",
        y="macro_f1",
        hue="feature_set",
        palette="crest"
    )
    plt.title("Scientific Ablation: Full Predictors vs. Excluded PM2.5/PM10 (Target Leakage Investigation)",
              fontsize=12, weight="bold")
    plt.xlabel("Algorithm")
    plt.ylabel("Test Macro F1")
    plt.xticks(ticks=range(len(df["algorithm"].unique())),
               labels=[a.replace("_", " ").title() for a in df["algorithm"].unique()], rotation=15)
    plt.legend(title="Feature Configuration")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
