"""
Independent Experiment Runner for Italy UCI Air Quality Dataset (Sreenidhi)
Evaluates: Support Vector Machine (LinearSVC) & Gaussian Naive Bayes
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

from src.data_loader import load_italy_dataset
from src.feature_engineering import add_temporal_features
from src.preprocessing import chronological_train_test_split, build_preprocessor
from src.evaluation import train_and_evaluate
from src.visualization import plot_confusion_matrix, plot_model_comparison
from models import get_models


def main():
    print("=" * 60)
    print("  ITALY DATASET BENCHMARK")
    print("  Models: Support Vector Machine (LinearSVC) & Naive Bayes")
    print("=" * 60)

    # 1. Load data
    data_path = os.path.join(current_dir, "data", "italy_air_quality_uci.csv")
    df = load_italy_dataset(data_path)
    df = add_temporal_features(df, date_col="DateTime")

    # 2. Output directory
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 3. Train/Test split (80/20 chronological)
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        df, target_col="AQI_Category", train_ratio=0.8, date_col="DateTime"
    )

    cat_cols = [c for c in ["season"] if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    prep_scaled = build_preprocessor(num_cols, cat_cols, scale=True)
    prep_unscaled = build_preprocessor(num_cols, cat_cols, scale=False)

    models = get_models()
    results = []

    # 4. Train and evaluate
    print("\nTraining models...")
    for name, model in models.items():
        print(f"  --> Training {name:<20} ...", end=" ", flush=True)

        prep = prep_scaled if name == "svm" else prep_unscaled
        X_tr = prep.fit_transform(X_train)
        X_te = prep.transform(X_test)

        metrics = train_and_evaluate(model, X_tr, y_train, X_te, y_test)
        print(f"Done! Macro F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f}")

        results.append({
            "member": "Sreenidhi",
            "dataset": "italy",
            "dataset_name": "Italy (UCI Air Quality)",
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
            "Italy (UCI Air Quality)",
            os.path.join(results_dir, f"cm_italy_{name}.png")
        )

    # Save metrics table
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "sreenidhi_results.csv")
    res_df.to_csv(csv_path, index=False)
    plot_model_comparison(res_df, "Italy Dataset (Sreenidhi)", os.path.join(results_dir, "model_compare_italy.png"))

    print("\nResults Summary:")
    print(res_df[["algorithm", "macro_f1", "balanced_acc", "accuracy", "train_seconds"]].to_string(index=False))
    print(f"\nAll outputs saved to: {results_dir}\n")


if __name__ == "__main__":
    main()
