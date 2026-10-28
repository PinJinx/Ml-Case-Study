"""
run_all_experiments.py
Main reproducible experiment orchestrator.
Executes the independent member modules for all 4 team members:
1. Vimal (src/Vimal): India Dataset | Logistic Regression & KNN
2. Sreenidhi (src/Sreenidhi): Italy Dataset | SVM (LinearSVC) & Naive Bayes
3. Rohith (src/Rohith): Beijing Dataset | Random Forest & Gradient Boosting
4. Sruthi (src/Sruthi): Seoul Dataset | Decision Tree & XGBoost

Aggregates individual member results from src/<Member>/results/ into results/results.csv,
and generates publication-quality cross-dataset team heatmaps, rankings, and ablation figures.
"""

import os
import sys
import time
import shutil
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.Vimal.run_experiments import run_vimal_experiments
from src.Sreenidhi.run_experiments import run_sreenidhi_experiments
from src.Rohith.run_experiments import run_rohith_experiments
from src.Sruthi.run_experiments import run_sruthi_experiments

from src.comparison import (
    aggregate_team_results,
    plot_cross_dataset_heatmap,
    plot_team_models_summary,
    plot_average_algorithm_ranks
)


def main():
    start_total = time.time()
    print("=" * 70)
    print("  ACADEMIC ML BENCHMARK: 4-MEMBER COLLABORATIVE AQI STUDY")
    print("  8 Classifiers Paired Across 4 International Environmental Datasets")
    print("=" * 70)

    # 1. Execute Member 1 (Vimal - India)
    run_vimal_experiments()

    # 2. Execute Member 2 (Sreenidhi - Italy)
    run_sreenidhi_experiments()

    # 3. Execute Member 3 (Rohith - Beijing)
    run_rohith_experiments()

    # 4. Execute Member 4 (Sruthi - Seoul)
    run_sruthi_experiments()

    # 5. Aggregate all member results
    print("\n" + "=" * 70)
    print("  AGGREGATING ALL TEAM RESULTS & CROSS-DATASET SYNTHESIS")
    print("=" * 70)

    member_csvs = [
        "src/Vimal/results/vimal_results.csv",
        "src/Sreenidhi/results/sreenidhi_results.csv",
        "src/Rohith/results/rohith_results.csv",
        "src/Sruthi/results/sruthi_results.csv"
    ]

    dfs = []
    for m_csv in member_csvs:
        if os.path.exists(m_csv):
            dfs.append(pd.read_csv(m_csv))
        else:
            print(f"Warning: Missing results file {m_csv}")

    if not dfs:
        print("Error: No results found to aggregate!")
        return

    full_results_df = pd.concat(dfs, ignore_index=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)
    os.makedirs("results/feature_importance", exist_ok=True)

    results_csv = "results/results.csv"
    full_results_df.to_csv(results_csv, index=False)
    print(f"Aggregated {len(full_results_df)} experimental runs saved to {results_csv}")

    # Copy plots & metrics from member results to root results
    for root_dir, dirs, files in os.walk("src"):
        if "results" in root_dir.split(os.sep):
            for f in files:
                src_path = os.path.join(root_dir, f)
                if f.startswith("cm_"):
                    shutil.copy(src_path, os.path.join("results/confusion_matrices", f))
                elif f.startswith("fi_"):
                    shutil.copy(src_path, os.path.join("results/feature_importance", f))
                elif f.startswith("ablation_") or f.endswith("_ablation_results.csv"):
                    if f.endswith(".png"):
                        shutil.copy(src_path, os.path.join("results/plots", f))
                    elif f.endswith(".csv"):
                        shutil.copy(src_path, os.path.join("results/metrics", f))
                elif f.endswith(".png") and not f.startswith("cm_") and not f.startswith("fi_"):
                    shutil.copy(src_path, os.path.join("results/plots", f))

    # Generate Team-Level Comparisons
    print("\nGenerating Team-Level Cross-Dataset Visualizations...")
    plot_cross_dataset_heatmap(full_results_df, "results/plots/cross_dataset_heatmap_none.png", method="none")
    plot_cross_dataset_heatmap(full_results_df, "results/plots/cross_dataset_heatmap_class_weight.png", method="class_weight")
    plot_team_models_summary(full_results_df, "results/plots/team_models_summary_none.png", method="none")
    plot_team_models_summary(full_results_df, "results/plots/team_models_summary_class_weight.png", method="class_weight")
    plot_average_algorithm_ranks(full_results_df, "results/plots/average_algorithm_ranks.png")

    # Generate quick summary table
    summary_cols = ["member", "dataset_name", "algorithm", "imbalance_method", "macro_f1", "balanced_acc", "accuracy", "off_by_2_plus", "train_seconds"]
    summary_df = full_results_df[summary_cols]
    summary_df.to_csv("results/metrics/summary_metrics.csv", index=False)

    total_time = time.time() - start_total
    print(f"\nALL EXPERIMENTS & CROSS-DATASET SYNTHESIS COMPLETED IN {total_time/60.0:.2f} MINUTES!")


if __name__ == "__main__":
    main()
