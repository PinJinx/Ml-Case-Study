"""
comparison.py
Team-Level Comparative Analysis across all four datasets and members:
- Vimal (India): Logistic Regression & KNN
- Sreenidhi (Italy): SVM & Naive Bayes
- Rohith (Beijing): Random Forest & Gradient Boosting
- Sruthi (Seoul): Decision Tree & XGBoost
"""

import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

    alg_order = [
        "majority_baseline",
        "logistic_regression",
        "knn",
        "svm",
        "naive_bayes",
        "random_forest",
        "gradient_boosting",
        "decision_tree",
        "xgboost"
    ]
    present_order = [a for a in alg_order if a in pivot.index]
    pivot = pivot.reindex(present_order)

    # Format annotations
    annot_text = pivot.map(lambda v: f"{v:.3f}" if pd.notnull(v) else "-")

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        pivot,
        annot=annot_text,
        fmt="",
        cmap="YlGnBu",
        cbar_kws={"label": "Test Macro F1"},
        linewidths=0.5
    )
    plt.title(f"Cross-Dataset Algorithm Benchmark (Macro F1, Imbalance: {method})", fontsize=13, weight="bold")
    plt.ylabel("Algorithm")
    plt.xlabel("Dataset / Team Member")
    plt.yticks(
        ticks=[i + 0.5 for i in range(len(present_order))],
        labels=[a.replace("_", " ").title() for a in present_order],
        rotation=0
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_team_models_summary(df: pd.DataFrame, save_path: str, method: str = "none"):
    """
    Generate a side-by-side grouped bar chart comparing all 8 clubbed models across the 4 team members.
    """
    subset = df[df["imbalance_method"] == method].copy()
    if subset.empty:
        subset = df.copy()

    # Filter out baseline for candidate model comparison
    candidates = subset[subset["algorithm"] != "majority_baseline"].copy()
    if candidates.empty:
        candidates = subset.copy()

    # Create display label
    candidates["Algorithm_Display"] = candidates["algorithm"].str.replace("_", " ").str.title()
    candidates["Member_Display"] = candidates["member"].apply(lambda m: str(m).split(" ")[0])

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=candidates,
        x="member",
        y="macro_f1",
        hue="Algorithm_Display",
        palette="tab10"
    )
    plt.title(f"Team Member Clubbed Model Comparison (Macro F1, Imbalance: {method})", fontsize=13, weight="bold")
    plt.xlabel("Team Member & Assigned Dataset")
    plt.ylabel("Test Macro F1 Score")
    plt.ylim(0, 1.05)
    plt.legend(title="Clubbed Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")

    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h + 0.02),
                        ha="center", va="bottom", fontsize=9, rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_average_algorithm_ranks(df: pd.DataFrame, save_path: str):
    """
    Compute and plot average rank of each algorithm across their evaluated datasets.
    Rank 1 is best (highest Macro F1).
    """
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
    plt.title("Algorithm Ranks Within Assigned Benchmark Datasets\n(Rank 1 = Highest Macro F1)", fontsize=13, weight="bold")
    plt.xlabel("Average Rank (Lower is Better)")
    plt.ylabel("Algorithm")
    plt.yticks(ticks=range(len(avg_ranks)), labels=[a.replace("_", " ").title() for a in avg_ranks.index])

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.2f}", (width + 0.05, p.get_y() + p.get_height() / 2.),
                    va="center", fontsize=10)

    plt.xlim(0, max(avg_ranks.values) + 0.8)
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
    plt.xticks(
        ticks=range(len(df["algorithm"].unique())),
        labels=[a.replace("_", " ").title() for a in df["algorithm"].unique()],
        rotation=15
    )
    plt.legend(title="Feature Configuration")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
