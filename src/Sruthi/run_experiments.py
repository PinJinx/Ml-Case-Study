"""
run_experiments.py (Sruthi - Member 4)
Independent experiment runner for Seoul Air Pollution dataset (seoul_air_pollution.csv).
Evaluates Sruthi's 2 clubbed models:
1. Decision Tree (DecisionTreeClassifier)
2. XGBoost (XGBClassifier)
(+ Majority Baseline)
Across 3 imbalance handling strategies: None, Class-Weighted, and SMOTE.
All results and figures are saved locally in src/Sruthi/results/.
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
spec = importlib.util.spec_from_file_location("sruthi_local_models", os.path.join(current_dir, "models.py"))
local_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_mod)

build_full_pipeline = local_mod.build_full_pipeline
requires_scaling = local_mod.requires_scaling
ALGORITHMS = local_mod.ALGORITHMS
MEMBER_NAME = local_mod.MEMBER_NAME

from src.data_loader import load_seoul_dataset
from src.feature_engineering import add_temporal_features
from src.preprocessing import chronological_train_test_split, build_preprocessor
from src.evaluation import train_and_evaluate_model
from src.visualization import (
    plot_missing_values,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_imbalance_comparison,
    plot_feature_importance
)
from src.labeling import AQI_CATEGORIES


def run_sruthi_experiments():
    print("=" * 65)
    print(f"  RUNNING INDEPENDENT BENCHMARK: {MEMBER_NAME} (Seoul Dataset)")
    print(f"  Clubbed Models: Decision Tree & XGBoost")
    print("=" * 65)

    # 1. Locate dataset
    local_data_path = os.path.join(current_dir, "data", "seoul_air_pollution.csv")
    if not os.path.exists(local_data_path):
        local_data_path = os.path.join(project_root, "data", "raw", "seoul_air_pollution.csv")

    print(f"Loading dataset from: {local_data_path}")
    df = load_seoul_dataset(local_data_path)
    df = add_temporal_features(df, date_col="Date")

    # 2. Output directory
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 3. EDA plots
    print("Generating EDA visualizations...")
    plot_missing_values(df, "Seoul (Air Pollution Dataset)", os.path.join(results_dir, "missing_seoul.png"))
    plot_class_distribution(df["AQI_Category"], "Seoul (Air Pollution Dataset)", os.path.join(results_dir, "class_dist_seoul.png"), order=AQI_CATEGORIES)
    plot_correlation_heatmap(df, "Seoul (Air Pollution Dataset)", os.path.join(results_dir, "corr_seoul.png"))

    # 4. Train/Test Split (80/20 chronological)
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        df, target_col="AQI_Category", train_ratio=0.8, date_col="Date"
    )
    print(f"Chronological Split: {len(X_train)} Train samples, {len(X_test)} Test samples")
    print("Train Class Distribution:\n", y_train.value_counts().to_dict())

    categorical_cols = ["Station code", "season"]
    all_cat = [c for c in categorical_cols if c in X_train.columns]
    all_num = [c for c in X_train.columns if c not in all_cat]

    preprocessor_scaled = build_preprocessor(all_num, all_cat, scale=True, impute_strategy="median")
    preprocessor_unscaled = build_preprocessor(all_num, all_cat, scale=False, impute_strategy="median")

    algorithms = ["majority_baseline", "decision_tree", "xgboost"]
    imbalance_methods = ["none", "class_weight", "smote"]
    min_train_samples = y_train.value_counts().min()

    results = []

    print("\n--- Training Models ---")
    for imb_method in imbalance_methods:
        for alg in algorithms:
            if imb_method == "class_weight" and alg in ["majority_baseline"]:
                continue
            if imb_method == "smote" and (alg == "majority_baseline" or min_train_samples < 2):
                continue

            print(f"  Training {alg:<20} | Imbalance: {imb_method:<12} ...", end=" ", flush=True)

            prep = preprocessor_scaled if requires_scaling(alg) else preprocessor_unscaled
            pipe, param_grid = build_full_pipeline(
                preprocessor=prep,
                algorithm=alg,
                imbalance_method=imb_method,
                random_state=42,
                min_class_samples=min_train_samples
            )

            # Fast tuning for XGBoost on large datasets
            if alg == "xgboost":
                param_grid = {
                    "classifier__n_estimators": [50],
                    "classifier__max_depth": [4],
                    "classifier__learning_rate": [0.1]
                }

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
                "member": "Sruthi (Member 4)",
                "dataset": "seoul",
                "dataset_name": "Seoul (Air Pollution Dataset)",
                "algorithm": alg,
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
                cm_path = os.path.join(results_dir, f"cm_seoul_{alg}.png")
                plot_confusion_matrix(
                    np.array(test_m["confusion_matrix"]),
                    test_m["class_names"],
                    alg.replace("_", " ").title(),
                    "Seoul (Air Pollution Dataset)",
                    cm_path
                )

                est = eval_res["best_model"].named_steps.get("classifier")
                if hasattr(est, "feature_importances_"):
                    preproc = eval_res["best_model"].named_steps.get("preprocessor")
                    feature_names = preproc.get_feature_names_out()
                    clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
                    importances = dict(zip(clean_names, est.feature_importances_))
                    fi_path = os.path.join(results_dir, f"fi_seoul_{alg}.png")
                    plot_feature_importance(
                        importances,
                        alg.replace("_", " ").title(),
                        "Seoul (Air Pollution Dataset)",
                        fi_path
                    )

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "sruthi_results.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark metrics to {csv_path}")

    # Visualizations
    std_df = res_df[res_df["imbalance_method"] == "none"]
    plot_model_comparison(std_df, "Seoul (Air Pollution Dataset) - Sruthi", os.path.join(results_dir, "model_compare_seoul.png"))
    plot_imbalance_comparison(res_df, "Seoul (Air Pollution Dataset) - Sruthi", os.path.join(results_dir, "imbalance_compare_seoul.png"))

    # Summary Display
    print("\n" + "=" * 65)
    print("  SRUTHI EXPERIMENTAL RESULTS SUMMARY:")
    print("=" * 65)
    print(res_df[["algorithm", "imbalance_method", "macro_f1", "balanced_acc", "accuracy", "off_by_2_plus", "train_seconds"]].to_string(index=False))
    print("=" * 65)
    print(f"All outputs successfully generated in: {results_dir}\n")


if __name__ == "__main__":
    run_sruthi_experiments()
