"""
finish_ablation_and_visualizations.py
Aggregates completed experimental results from each member folder:
- src/Vimal/results/vimal_results.csv
- src/Sreenidhi/results/sreenidhi_results.csv
- src/Rohith/results/rohith_results.csv
- src/Sruthi/results/sruthi_results.csv

Generates unified team results.csv, summary metrics table, cross-dataset heatmaps,
team-wide clubbed model comparisons, and average rankings.
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.comparison import (
    aggregate_team_results,
    plot_cross_dataset_heatmap,
    plot_team_models_summary,
    plot_average_algorithm_ranks
)


def main():
    print("=" * 70)
    print("  AGGREGATING INDEPENDENT MEMBER BENCHMARKS INTO TEAM SYNTHESIS")
    print("=" * 70)

    member_csvs = [
        ("Vimal", "src/Vimal/results/vimal_results.csv"),
        ("Sreenidhi", "src/Sreenidhi/results/sreenidhi_results.csv"),
        ("Rohith", "src/Rohith/results/rohith_results.csv"),
        ("Sruthi", "src/Sruthi/results/sruthi_results.csv")
    ]

    dfs = []
    for member_name, csv_path in member_csvs:
        if os.path.exists(csv_path):
            df_m = pd.read_csv(csv_path)
            print(f"Loaded {len(df_m)} experimental runs from {member_name} ({csv_path})")
            dfs.append(df_m)
        else:
            print(f"Warning: {csv_path} not found!")

    if not dfs:
        print("Error: No member results found!")
        return

    full_results_df = pd.concat(dfs, ignore_index=True)

    # Standardize algorithm column name for SVM
    full_results_df["algorithm"] = full_results_df["algorithm"].replace({
        "linear_svc": "svm",
        "svm (LinearSVC)": "svm"
    })

    # Output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)
    os.makedirs("results/feature_importance", exist_ok=True)

    # Save full consolidated results
    results_csv = "results/results.csv"
    full_results_df.to_csv(results_csv, index=False)
    print(f"\nSaved consolidated {len(full_results_df)} experimental runs to {results_csv}")

    # Copy files from member results to global results
    for member_name, _ in member_csvs:
        mem_dir = f"src/{member_name}/results"
        if not os.path.exists(mem_dir):
            continue
        for f in os.listdir(mem_dir):
            src_f = os.path.join(mem_dir, f)
            if f.startswith("cm_"):
                shutil.copy(src_f, os.path.join("results/confusion_matrices", f))
            elif f.startswith("fi_"):
                shutil.copy(src_f, os.path.join("results/feature_importance", f))
            elif f.startswith("ablation_") or f.endswith("_ablation_results.csv"):
                if f.endswith(".png"):
                    shutil.copy(src_f, os.path.join("results/plots", f))
                elif f.endswith(".csv"):
                    shutil.copy(src_f, os.path.join("results/metrics", f))
            elif f.endswith(".png"):
                shutil.copy(src_f, os.path.join("results/plots", f))

    # Generate Team-Level Visualizations
    print("\nGenerating Team-Level Cross-Dataset Visualizations...")
    plot_cross_dataset_heatmap(full_results_df, "results/plots/cross_dataset_heatmap_none.png", method="none")
    plot_cross_dataset_heatmap(full_results_df, "results/plots/cross_dataset_heatmap_class_weight.png", method="class_weight")
    plot_team_models_summary(full_results_df, "results/plots/team_models_summary_none.png", method="none")
    plot_team_models_summary(full_results_df, "results/plots/team_models_summary_class_weight.png", method="class_weight")
    plot_average_algorithm_ranks(full_results_df, "results/plots/average_algorithm_ranks.png")
    print("Team plots saved to results/plots/")

    # Save clean summary metrics
    summary_cols = ["member", "dataset_name", "algorithm", "imbalance_method", "macro_f1", "balanced_acc", "accuracy", "off_by_2_plus", "train_seconds"]
    summary_df = full_results_df[summary_cols].sort_values(by=["member", "imbalance_method"])
    summary_csv = "results/metrics/summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary metrics saved to {summary_csv}")

    print("\n" + "=" * 70)
    print("  TEAM-WIDE BENCHMARK SUMMARY (Imbalance Method: None)")
    print("=" * 70)
    std_summary = summary_df[summary_df["imbalance_method"] == "none"]
    print(std_summary[["member", "algorithm", "macro_f1", "balanced_acc", "accuracy", "off_by_2_plus", "train_seconds"]].to_string(index=False))
    print("=" * 70)
    print("\nALL SYNTHESIS ARTIFACTS AND VISUALIZATIONS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
