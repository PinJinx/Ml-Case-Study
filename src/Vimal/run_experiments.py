"""
run_experiments.py (Vimal - Member 1)
Independent experiment runner for India dataset (city_day.csv).
Evaluates Vimal's 2 clubbed models:
1. Logistic Regression
2. K-Nearest Neighbours (KNN)
(+ Majority Baseline)
Across 3 imbalance handling strategies: None, Class-Weighted, and SMOTE.
Also runs the India Target Leakage Ablation Study (excluding PM2.5 and PM10).
All results and figures are saved locally in src/Vimal/results/.
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
spec = importlib.util.spec_from_file_location("vimal_local_models", os.path.join(current_dir, "models.py"))
local_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_mod)

build_full_pipeline = local_mod.build_full_pipeline
requires_scaling = local_mod.requires_scaling
ALGORITHMS = local_mod.ALGORITHMS
MEMBER_NAME = local_mod.MEMBER_NAME

from src.data_loader import load_india_dataset
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
from src.comparison import plot_ablation_comparison
from src.labeling import AQI_CATEGORIES


def run_vimal_experiments():
    print("=" * 65)
    print(f"  RUNNING INDEPENDENT BENCHMARK: {MEMBER_NAME} (India Dataset)")
    print(f"  Clubbed Models: Logistic Regression & K-Nearest Neighbours (KNN)")
    print("=" * 65)

    # 1. Locate dataset
    local_data_path = os.path.join(current_dir, "data", "india_city_day.csv")
    if not os.path.exists(local_data_path):
        local_data_path = os.path.join(project_root, "data", "raw", "india_city_day.csv")
    
    print(f"Loading dataset from: {local_data_path}")
    df = load_india_dataset(local_data_path)
    df = add_temporal_features(df, date_col="Date")

    # 2. Output directory
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 3. EDA plots
    print("Generating EDA visualizations...")
    plot_missing_values(df, "India (city_day.csv)", os.path.join(results_dir, "missing_india.png"))
    plot_class_distribution(df["AQI_Bucket"], "India (city_day.csv)", os.path.join(results_dir, "class_dist_india.png"), order=AQI_CATEGORIES)
    plot_correlation_heatmap(df, "India (city_day.csv)", os.path.join(results_dir, "corr_india.png"))

    # 4. Train/Test Split (80/20 chronological)
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        df, target_col="AQI_Bucket", train_ratio=0.8, date_col="Date"
    )
    print(f"Chronological Split: {len(X_train)} Train samples, {len(X_test)} Test samples")

    categorical_cols = ["City", "season"]
    all_cat = [c for c in categorical_cols if c in X_train.columns]
    all_num = [c for c in X_train.columns if c not in all_cat]

    preprocessor_scaled = build_preprocessor(all_num, all_cat, scale=True, impute_strategy="median")
    preprocessor_unscaled = build_preprocessor(all_num, all_cat, scale=False, impute_strategy="median")

    algorithms = ["majority_baseline", "logistic_regression", "knn"]
    imbalance_methods = ["none", "class_weight", "smote"]
    min_train_samples = y_train.value_counts().min()

    results = []

    print("\n--- Training Models ---")
    for imb_method in imbalance_methods:
        for alg in algorithms:
            if imb_method == "class_weight" and alg in ["knn", "majority_baseline"]:
                continue
            if imb_method == "smote" and alg == "majority_baseline":
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
                "member": "Vimal (Member 1)",
                "dataset": "india",
                "dataset_name": "India (city_day.csv)",
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
                cm_path = os.path.join(results_dir, f"cm_india_{alg}.png")
                plot_confusion_matrix(
                    np.array(test_m["confusion_matrix"]),
                    test_m["class_names"],
                    alg.replace("_", " ").title(),
                    "India (city_day.csv)",
                    cm_path
                )

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "vimal_results.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark metrics to {csv_path}")

    # Visualizations
    std_df = res_df[res_df["imbalance_method"] == "none"]
    plot_model_comparison(std_df, "India (city_day.csv) - Vimal", os.path.join(results_dir, "model_compare_india.png"))
    plot_imbalance_comparison(res_df, "India (city_day.csv) - Vimal", os.path.join(results_dir, "imbalance_compare_india.png"))

    # 5. Target Leakage Ablation Study
    print("\n--- Running India Target Leakage Ablation Study ---")
    ablation_cols = ["PM2.5", "PM10"]
    X_train_abl = X_train.drop(columns=[c for c in ablation_cols if c in X_train.columns])
    X_test_abl = X_test.drop(columns=[c for c in ablation_cols if c in X_test.columns])
    cat_cols = [c for c in categorical_cols if c in X_train_abl.columns]
    num_cols = [c for c in X_train_abl.columns if c not in cat_cols]

    prep_scaled_abl = build_preprocessor(num_cols, cat_cols, scale=True)
    prep_unscaled_abl = build_preprocessor(num_cols, cat_cols, scale=False)

    ablation_rows = []
    for alg in ["logistic_regression", "knn"]:
        print(f"  Testing {alg:<20} without PM2.5 & PM10 ...", end=" ", flush=True)
        prep = prep_scaled_abl if requires_scaling(alg) else prep_unscaled_abl
        pipe, param_grid = build_full_pipeline(prep, alg, imbalance_method="none", random_state=42)

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
        print(f"Done! Macro F1: {test_m['macro_f1']:.4f}")

        ablation_rows.append({
            "algorithm": alg,
            "feature_set": "Without PM2.5 & PM10 (Ablated)",
            "macro_f1": test_m["macro_f1"],
            "balanced_acc": test_m["balanced_acc"],
            "accuracy": test_m["accuracy"]
        })

        m = res_df[(res_df["algorithm"] == alg) & (res_df["imbalance_method"] == "none")]
        if not m.empty:
            ablation_rows.append({
                "algorithm": alg,
                "feature_set": "Full Features (with PM2.5 & PM10)",
                "macro_f1": float(m["macro_f1"].iloc[0]),
                "balanced_acc": float(m["balanced_acc"].iloc[0]),
                "accuracy": float(m["accuracy"].iloc[0])
            })

    abl_csv_path = os.path.join(results_dir, "india_ablation_results.csv")
    pd.DataFrame(ablation_rows).to_csv(abl_csv_path, index=False)
    abl_plot_path = os.path.join(results_dir, "ablation_india_leakage.png")
    plot_ablation_comparison(ablation_rows, abl_plot_path)
    print(f"Saved ablation study results to {abl_csv_path} and {abl_plot_path}")

    # Summary Display
    print("\n" + "=" * 65)
    print("  VIMAL EXPERIMENTAL RESULTS SUMMARY:")
    print("=" * 65)
    print(res_df[["algorithm", "imbalance_method", "macro_f1", "balanced_acc", "accuracy", "off_by_2_plus", "train_seconds"]].to_string(index=False))
    print("=" * 65)
    print(f"All outputs successfully generated in: {results_dir}\n")


if __name__ == "__main__":
    run_vimal_experiments()
