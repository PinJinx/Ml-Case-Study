"""
run_experiments.py (Sreenidhi - Member 2)
Independent experiment runner for Italy UCI Air Quality dataset (italy_air_quality_uci.csv).
Evaluates Sreenidhi's 2 clubbed models:
1. Support Vector Machine (LinearSVC / RBF)
2. Gaussian Naive Bayes (GaussianNB)
(+ Majority Baseline)
Across 3 imbalance handling strategies: None, Class-Weighted, and SMOTE.
All results and figures are saved locally in src/Sreenidhi/results/.
"""

import os
import sys
import json
import time
import importlib.util
import numpy as np
import pandas as pd

# Support running from within this folder or from project root
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load local models strictly from this directory
spec = importlib.util.spec_from_file_location("sreenidhi_local_models", os.path.join(current_dir, "models.py"))
local_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_mod)

build_full_pipeline = local_mod.build_full_pipeline
requires_scaling = local_mod.requires_scaling
ALGORITHMS = local_mod.ALGORITHMS
MEMBER_NAME = local_mod.MEMBER_NAME

from src.data_loader import load_italy_dataset
from src.feature_engineering import add_temporal_features
from src.preprocessing import chronological_train_test_split, build_preprocessor
from src.evaluation import train_and_evaluate_model
from src.visualization import (
    plot_missing_values,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_imbalance_comparison
)
from src.labeling import AQI_CATEGORIES


def run_sreenidhi_experiments():
    print("=" * 65)
    print(f"  RUNNING INDEPENDENT BENCHMARK: {MEMBER_NAME} (Italy Dataset)")
    print(f"  Clubbed Models: Support Vector Machine (SVM) & Naive Bayes")
    print("=" * 65)

    # 1. Locate dataset
    local_data_path = os.path.join(current_dir, "data", "italy_air_quality_uci.csv")
    if not os.path.exists(local_data_path):
        local_data_path = os.path.join(project_root, "data", "raw", "italy_air_quality_uci.csv")

    print(f"Loading dataset from: {local_data_path}")
    df = load_italy_dataset(local_data_path)
    df = add_temporal_features(df, date_col="DateTime")

    # 2. Output directory
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 3. EDA plots
    print("Generating EDA visualizations...")
    plot_missing_values(df, "Italy (UCI Air Quality)", os.path.join(results_dir, "missing_italy.png"))
    plot_class_distribution(df["AQI_Category"], "Italy (UCI Air Quality)", os.path.join(results_dir, "class_dist_italy.png"), order=AQI_CATEGORIES)
    plot_correlation_heatmap(df, "Italy (UCI Air Quality)", os.path.join(results_dir, "corr_italy.png"))

    # 4. Train/Test Split (80/20 chronological)
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        df, target_col="AQI_Category", train_ratio=0.8, date_col="DateTime"
    )
    print(f"Chronological Split: {len(X_train)} Train samples, {len(X_test)} Test samples")
    print("Train Class Distribution:\n", y_train.value_counts().to_dict())

    categorical_cols = ["season"]
    all_cat = [c for c in categorical_cols if c in X_train.columns]
    all_num = [c for c in X_train.columns if c not in all_cat]

    preprocessor_scaled = build_preprocessor(all_num, all_cat, scale=True, impute_strategy="median")
    preprocessor_unscaled = build_preprocessor(all_num, all_cat, scale=False, impute_strategy="median")

    algorithms = ["majority_baseline", "svm", "naive_bayes"]
    imbalance_methods = ["none", "class_weight", "smote"]
    min_train_samples = y_train.value_counts().min()

    results = []

    print("\n--- Training Models ---")
    for imb_method in imbalance_methods:
        for alg in algorithms:
            if imb_method == "class_weight" and alg in ["naive_bayes", "majority_baseline"]:
                continue
            if imb_method == "smote" and (alg == "majority_baseline" or min_train_samples < 2):
                continue

            current_alg = alg
            if len(X_train) > 5000 and alg == "svm":
                current_alg = "linear_svc"

            print(f"  Training {current_alg:<20} | Imbalance: {imb_method:<12} ...", end=" ", flush=True)

            prep = preprocessor_scaled if requires_scaling(current_alg) else preprocessor_unscaled
            pipe, param_grid = build_full_pipeline(
                preprocessor=prep,
                algorithm=current_alg,
                imbalance_method=imb_method,
                random_state=42,
                min_class_samples=min_train_samples
            )

            eval_res = train_and_evaluate_model(
                pipeline=pipe,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv_splits=5
            )

            test_m = eval_res["test_metrics"]
            print(f"Done! Macro F1: {test_m['macro_f1']:.4f} | Balanced Acc: {test_m['balanced_acc']:.4f} | Accuracy: {test_m['accuracy']:.4f}")

            rec = {
                "member": "Sreenidhi (Member 2)",
                "dataset": "italy",
                "dataset_name": "Italy (UCI Air Quality)",
                "algorithm": current_alg,
                "imbalance_method": imb_method,
                "macro_f1": test_m["macro_f1"],
                "balanced_acc": test_m["balanced_acc"],
                "accuracy": test_m["accuracy"],
                "off_by_2_plus": test_m["off_by_2_plus"],
                "train_seconds": eval_res["train_seconds"],
                "cv_mean": eval_res["cv_mean"],
                "cv_std": eval_res["cv_std"],
                "best_params": json.dumps(eval_res["best_params"]),
                "per_class": json.dumps(test_m["per_class"])
            }
            results.append(rec)

            if imb_method == "none":
                cm_path = os.path.join(results_dir, f"cm_italy_{current_alg}.png")
                plot_confusion_matrix(
                    np.array(test_m["confusion_matrix"]),
                    test_m["class_names"],
                    current_alg.replace("_", " ").title(),
                    "Italy (UCI Air Quality)",
                    cm_path
                )

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "sreenidhi_results.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark metrics to {csv_path}")

    # Visualizations
    std_df = res_df[res_df["imbalance_method"] == "none"]
    plot_model_comparison(std_df, "Italy (UCI Air Quality) - Sreenidhi", os.path.join(results_dir, "model_compare_italy.png"))
    plot_imbalance_comparison(res_df, "Italy (UCI Air Quality) - Sreenidhi", os.path.join(results_dir, "imbalance_compare_italy.png"))

    # Summary Display
    print("\n" + "=" * 65)
    print("  SREENIDHI EXPERIMENTAL RESULTS SUMMARY:")
    print("=" * 65)
    print(res_df[["algorithm", "imbalance_method", "macro_f1", "balanced_acc", "accuracy", "off_by_2_plus", "train_seconds"]].to_string(index=False))
    print("=" * 65)
    print(f"All outputs successfully generated in: {results_dir}\n")


if __name__ == "__main__":
    run_sreenidhi_experiments()
