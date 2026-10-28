"""
Independent Experiment Runner for India Dataset (Vimal)
Evaluates: Logistic Regression & K-Nearest Neighbours (KNN)
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root and current folder are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
for p in [project_root, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_loader import load_india_dataset
from src.feature_engineering import add_temporal_features
from src.preprocessing import chronological_train_test_split, build_preprocessor
from src.evaluation import train_and_evaluate
from src.visualization import plot_confusion_matrix, plot_model_comparison
from src.comparison import plot_ablation_comparison
from models import get_models


def main():
    print("=" * 60)
    print("  INDIA DATASET BENCHMARK")
    print("  Models: Logistic Regression & K-Nearest Neighbours (KNN)")
    print("=" * 60)

    # 1. Load data
    data_path = os.path.join(current_dir, "data", "india_city_day.csv")
    df = load_india_dataset(data_path)
    df = add_temporal_features(df, date_col="Date")

    # 2. Output directory
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 3. Train/Test split (80/20 chronological)
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        df, target_col="AQI_Bucket", train_ratio=0.8, date_col="Date"
    )

    cat_cols = [c for c in ["City", "season"] if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Preprocessors (scaled for distance/linear models)
    prep_scaled = build_preprocessor(num_cols, cat_cols, scale=True)
    prep_unscaled = build_preprocessor(num_cols, cat_cols, scale=False)

    models = get_models()
    results = []

    # 4. Train and evaluate
    print("\nTraining models...")
    for name, model in models.items():
        print(f"  --> Training {name:<20} ...", end=" ", flush=True)

        prep = prep_scaled if name in ["logistic_regression", "knn"] else prep_unscaled
        X_tr = prep.fit_transform(X_train)
        X_te = prep.transform(X_test)

        metrics = train_and_evaluate(model, X_tr, y_train, X_te, y_test)
        print(f"Done! Macro F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f}")

        results.append({
            "member": "Vimal",
            "dataset": "india",
            "dataset_name": "India (city_day.csv)",
            "algorithm": name,
            "imbalance_method": "none",
            "macro_f1": metrics["macro_f1"],
            "balanced_acc": metrics["balanced_acc"],
            "accuracy": metrics["accuracy"],
            "off_by_2_plus": metrics["off_by_2_plus"],
            "train_seconds": metrics["train_seconds"]
        })

        # Save confusion matrix
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            metrics["class_names"],
            name.replace("_", " ").title(),
            "India (city_day.csv)",
            os.path.join(results_dir, f"cm_india_{name}.png")
        )

    # Save metrics table
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "vimal_results.csv")
    res_df.to_csv(csv_path, index=False)
    plot_model_comparison(res_df, "India Dataset (Vimal)", os.path.join(results_dir, "model_compare_india.png"))

    # 5. Target Leakage Ablation (without PM2.5 and PM10)
    print("\nRunning Target Leakage Ablation Study (excluding PM2.5/PM10)...")
    ablation_cols = ["PM2.5", "PM10"]
    X_train_abl = X_train.drop(columns=[c for c in ablation_cols if c in X_train.columns])
    X_test_abl = X_test.drop(columns=[c for c in ablation_cols if c in X_test.columns])
    num_abl = [c for c in X_train_abl.columns if c not in cat_cols]

    prep_abl = build_preprocessor(num_abl, cat_cols, scale=True)
    X_tr_abl = prep_abl.fit_transform(X_train_abl)
    X_te_abl = prep_abl.transform(X_test_abl)

    ablation_rows = []
    for name in ["logistic_regression", "knn"]:
        model = get_models()[name]
        m = train_and_evaluate(model, X_tr_abl, y_train, X_te_abl, y_test)
        ablation_rows.append({
            "algorithm": name,
            "feature_set": "Without PM2.5/PM10",
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"]
        })
        # Full feature match
        full_m = res_df[res_df["algorithm"] == name].iloc[0]
        ablation_rows.append({
            "algorithm": name,
            "feature_set": "Full Features",
            "macro_f1": full_m["macro_f1"],
            "accuracy": full_m["accuracy"]
        })

    pd.DataFrame(ablation_rows).to_csv(os.path.join(results_dir, "india_ablation_results.csv"), index=False)
    plot_ablation_comparison(ablation_rows, os.path.join(results_dir, "ablation_india_leakage.png"))

    print("\nResults Summary:")
    print(res_df[["algorithm", "macro_f1", "balanced_acc", "accuracy", "train_seconds"]].to_string(index=False))
    print(f"\nAll outputs saved to: {results_dir}\n")


if __name__ == "__main__":
    main()
