"""
visualization.py
Publication-quality plotting routines for EDA, Model Comparisons, Confusion Matrices,
Feature Importance, and Imbalance Strategy Analysis.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 300


def plot_missing_values(df: pd.DataFrame, dataset_name: str, save_path: str):
    """Plot percentage of missing values per feature."""
    missing = (df.isnull().sum() / len(df)) * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=missing.values, y=missing.index, hue=missing.index, palette="viridis", legend=False)
    plt.title(f"Missing Value Percentage by Feature — {dataset_name}")
    plt.xlabel("Missing Percentage (%)")
    plt.ylabel("Features")
    plt.xlim(0, 100)

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.1f}%", (width + 1.0, p.get_y() + p.get_height() / 2.),
                    va="center", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_class_distribution(series: pd.Series, dataset_name: str, save_path: str, order: Optional[List[str]] = None):
    """Plot target class distribution with frequency and percentage."""
    counts = series.value_counts()
    if order:
        order = [cat for cat in order if cat in counts.index]
        counts = counts.reindex(order)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="rocket", legend=False)
    plt.title(f"AQI Target Category Distribution — {dataset_name}")
    plt.xlabel("AQI Category")
    plt.ylabel("Observation Count")
    plt.xticks(rotation=15)

    total = len(series)
    for p in ax.patches:
        height = p.get_height()
        pct = (height / total) * 100
        ax.annotate(f"{int(height)}\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height + (max(counts.values) * 0.02)),
                    ha="center", va="bottom", fontsize=10)

    plt.ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, dataset_name: str, save_path: str):
    """Plot Spearman/Pearson correlation heatmap for numeric features."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return

    corr = num_df.corr(method="spearman")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f"Feature Correlation Heatmap (Spearman) — {dataset_name}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm_array: np.ndarray, class_names: List[str], model_name: str, dataset_name: str, save_path: str):
    """Plot normalized confusion matrix heatmap."""
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(f"Normalized Confusion Matrix: {model_name} — {dataset_name}")
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results_df: pd.DataFrame, dataset_name: str, save_path: str):
    """Plot Macro F1 and Balanced Accuracy comparison bar chart."""
    df_sorted = results_df.sort_values("macro_f1", ascending=True)

    plt.figure(figsize=(10, 6))
    y = np.arange(len(df_sorted))
    width = 0.35

    plt.barh(y - width/2, df_sorted["macro_f1"], width, label="Macro F1", color="#2b5c8f")
    plt.barh(y + width/2, df_sorted["balanced_acc"], width, label="Balanced Accuracy", color="#e27c38")

    plt.yticks(y, df_sorted["algorithm"].str.replace("_", " ").str.title())
    plt.xlabel("Score")
    plt.xlim(0, 1.05)
    plt.title(f"Algorithm Performance Comparison — {dataset_name}")
    plt.legend(loc="lower right")

    for i, v in enumerate(df_sorted["macro_f1"]):
        plt.text(v + 0.01, i - width/2, f"{v:.3f}", va="center", fontsize=9)
    for i, v in enumerate(df_sorted["balanced_acc"]):
        plt.text(v + 0.01, i + width/2, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_imbalance_comparison(df: pd.DataFrame, dataset_name: str, save_path: str):
    """Compare Macro F1 across imbalance handling methods."""
    plt.figure(figsize=(10, 5.5))
    sns.barplot(
        data=df,
        x="algorithm",
        y="macro_f1",
        hue="imbalance_method",
        palette="Set2"
    )
    plt.title(f"Impact of Class Imbalance Handling — {dataset_name}")
    plt.xlabel("Algorithm")
    plt.ylabel("Test Macro F1")
    plt.xticks(rotation=20)
    plt.legend(title="Imbalance Method")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importance_dict: Dict[str, float], model_name: str, dataset_name: str, save_path: str, top_n: int = 15):
    """Plot top N feature importances."""
    items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not items:
        return

    features, scores = zip(*items)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=list(scores), y=list(features), palette="mako")
    plt.title(f"Feature Importance ({model_name}) — {dataset_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
