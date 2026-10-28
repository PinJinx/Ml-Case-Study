"""
Independent Experiment Runner for Beijing PM2.5 Dataset (Rohith)
Evaluates: Random Forest & Gradient Boosting
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

from src.data_loader import load_beijing_dataset
from src.feature_engineering import add_temporal_features
from src.preprocessing import chronological_train_test_split, build_preprocessor
from src.evaluation import train_and_evaluate
from src.visualization import plot_confusion_matrix, plot_model_comparison, plot_feature_importance
from models import get_models


def main():
    print("=" * 60)
    print("  BEIJING DATASET BENCHMARK")
    print("  Models: Random Forest & Gradient Boosting")
    print("=" * 60)

    # 1. Load data
    data_path = os.path.join(current_dir, "data", "beijing_pm25.csv")
    df = load_beijing_dataset(data_path, aggregate_daily=True)
    df = add_temporal_features(df, date_col="Date")

    # 2. Output directory
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 3. Train/Test split (80/20 chronological)
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        df, target_col="AQI_Category", train_ratio=0.8, date_col="Date"
    )

    cat_cols = [c for c in ["cbwd", "season"] if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Preprocessor (unscaled for tree-based models)
    preprocessor = build_preprocessor(num_cols, cat_cols, scale=False)
    X_tr = preprocessor.fit_transform(X_train)
    X_te = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    models = get_models()
    results = []

    # 4. Train and evaluate
    print("\nTraining models...")
    for name, model in models.items():
        print(f"  --> Training {name:<20} ...", end=" ", flush=True)

        metrics = train_and_evaluate(model, X_tr, y_train, X_te, y_test)
        print(f"Done! Macro F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f}")

        results.append({
            "member": "Rohith",
            "dataset": "beijing",
            "dataset_name": "Beijing (UCI PM2.5)",
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
            "Beijing (UCI PM2.5)",
            os.path.join(results_dir, f"cm_beijing_{name}.png")
        )

        # Save feature importance for tree models
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(clean_feature_names, model.feature_importances_))
            plot_feature_importance(
                importances,
                name.replace("_", " ").title(),
                "Beijing (UCI PM2.5)",
                os.path.join(results_dir, f"fi_beijing_{name}.png")
            )

    # Save metrics table
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "rohith_results.csv")
    res_df.to_csv(csv_path, index=False)
    plot_model_comparison(res_df, "Beijing Dataset (Rohith)", os.path.join(results_dir, "model_compare_beijing.png"))

    print("\nResults Summary:")
    print(res_df[["algorithm", "macro_f1", "balanced_acc", "accuracy", "train_seconds"]].to_string(index=False))
    print(f"\nAll outputs saved to: {results_dir}\n")


if __name__ == "__main__":
    main()
