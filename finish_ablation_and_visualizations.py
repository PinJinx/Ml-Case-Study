"""
finish_ablation_and_visualizations.py
Runs the India ablation experiment, joins with the completed results.csv,
and generates all remaining team-level visualizations and metrics tables.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_india_dataset
from src.feature_engineering import add_temporal_features
from src.preprocessing import chronological_train_test_split, build_preprocessor
from src.models import build_full_pipeline, requires_scaling
from src.evaluation import train_and_evaluate_model
from src.comparison import (
    plot_cross_dataset_heatmap,
    plot_average_algorithm_ranks,
    plot_ablation_comparison,
    aggregate_team_results
)


def main():
    print("=== Loading Completed Results ===")
    results_csv = "results/results.csv"
    results_df = pd.read_csv(results_csv)
    print(f"Loaded {len(results_df)} experimental runs from {results_csv}")

    # Run India Ablation
    print("\n=== Running India Ablation Experiment ===")
    df = load_india_dataset("data/raw/india_city_day.csv")
    df = add_temporal_features(df, date_col="Date")
    categorical_cols = ["City", "season"]

    X_train, X_test, y_train, y_test = chronological_train_test_split(
        df, target_col="AQI_Bucket", train_ratio=0.8, date_col="Date"
    )

    ablation_cols = ["PM2.5", "PM10"]
    X_train_abl = X_train.drop(columns=[c for c in ablation_cols if c in X_train.columns])
    X_test_abl = X_test.drop(columns=[c for c in ablation_cols if c in X_test.columns])
    cat_cols = [c for c in categorical_cols if c in X_train_abl.columns]
    num_cols = [c for c in X_train_abl.columns if c not in cat_cols]

    prep_scaled = build_preprocessor(num_cols, cat_cols, scale=True)
    prep_unscaled = build_preprocessor(num_cols, cat_cols, scale=False)

    algs = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
    ablation_rows = []

    for alg in algs:
        print(f"  --> Testing {alg:<20} without PM2.5 & PM10 ...", end=" ", flush=True)
        prep = prep_scaled if requires_scaling(alg) else prep_unscaled
        pipe, param_grid = build_full_pipeline(prep, alg, imbalance_method="none", random_state=42)
        if "classifier__n_estimators" in param_grid:
            param_grid["classifier__n_estimators"] = [50]
        if alg == "xgboost":
            param_grid = {
                "classifier__n_estimators": [50],
                "classifier__max_depth": [3],
                "classifier__learning_rate": [0.1]
            }

        eval_res = train_and_evaluate_model(
            pipeline=pipe,
            param_grid=param_grid,
            X_train=X_train_abl,
            y_train=y_train,
            X_test=X_test_abl,
            y_test=y_test,
            cv_splits=5
        )
        test_m = eval_res["test_metrics"]
        print(f"Done! Test Macro F1: {test_m['macro_f1']:.4f}")

        ablation_rows.append({
            "algorithm": alg,
            "feature_set": "Without PM2.5 & PM10 (Ablated)",
            "macro_f1": test_m["macro_f1"],
            "balanced_acc": test_m["balanced_acc"],
            "accuracy": test_m["accuracy"]
        })

        # Match baseline with PM2.5 & PM10
        m = results_df[(results_df["dataset"] == "india") & (results_df["algorithm"] == alg) & (results_df["imbalance_method"] == "none")]
        if not m.empty:
            ablation_rows.append({
                "algorithm": alg,
                "feature_set": "Full Features (with PM2.5 & PM10)",
                "macro_f1": float(m["macro_f1"].iloc[0]),
                "balanced_acc": float(m["balanced_acc"].iloc[0]),
                "accuracy": float(m["accuracy"].iloc[0])
            })

    # Save ablation results & plot
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    pd.DataFrame(ablation_rows).to_csv("results/metrics/india_ablation_results.csv", index=False)
    plot_ablation_comparison(ablation_rows, "results/plots/ablation_india_leakage.png")
    print("Ablation plot saved to results/plots/ablation_india_leakage.png")

    # Generate Team-Level Comparisons
    print("\n=== Generating Team-Level Comparisons ===")
    plot_cross_dataset_heatmap(results_df, "results/plots/cross_dataset_heatmap_none.png", method="none")
    plot_cross_dataset_heatmap(results_df, "results/plots/cross_dataset_heatmap_class_weight.png", method="class_weight")
    plot_average_algorithm_ranks(results_df, "results/plots/average_algorithm_ranks.png")
    print("Cross-dataset heatmaps and rank charts saved to results/plots/")
    print("\nALL RUNS AND VISUALIZATIONS COMPLETE!")


if __name__ == "__main__":
    main()
