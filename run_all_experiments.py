"""
run_all_experiments.py
Main reproducible experiment runner.
Executes end-to-end benchmarks across all 4 datasets:
- India (city_day.csv)
- Italy (AirQualityUCI.csv)
- Beijing (PRSA_data_2010.1.1-2014.12.31.csv)
- Seoul (Measurement_summary.csv)
Trains 7 algorithms + majority baseline across 3 imbalance strategies (none, class_weight, smote).
Runs India ablation study (without PM2.5/PM10).
Generates all publication-quality figures and logs results into results.csv.
"""

import os
import sys
import json
import time
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import (
    load_india_dataset,
    load_italy_dataset,
    load_beijing_dataset,
    load_seoul_dataset
)
from src.feature_engineering import add_temporal_features
from src.preprocessing import chronological_train_test_split, build_preprocessor
from src.models import build_full_pipeline, requires_scaling, get_base_model
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
from src.comparison import (
    aggregate_team_results,
    plot_cross_dataset_heatmap,
    plot_average_algorithm_ranks,
    plot_ablation_comparison
)
from src.labeling import ID_TO_CATEGORY, AQI_CATEGORIES


def run_dataset_experiments(
    dataset_key: str,
    config: Dict[str, Any],
    algorithms: List[str],
    imbalance_methods: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run full suite of experiments for a single dataset.
    """
    ds_cfg = config["datasets"][dataset_key]
    name = ds_cfg["name"]
    member = ds_cfg["member"]
    print(f"\n=======================================================")
    print(f"Starting Experiments for {name} ({member})")
    print(f"=======================================================")

    # 1. Load Data
    raw_path = ds_cfg["raw_path"]
    if dataset_key == "india":
        df = load_india_dataset(raw_path)
        date_col = "Date"
        target_col = "AQI_Bucket"
        categorical_cols = ["City", "season"]
    elif dataset_key == "italy":
        df = load_italy_dataset(raw_path)
        date_col = "DateTime"
        target_col = "AQI_Category"
        categorical_cols = ["season"]
    elif dataset_key == "beijing":
        df = load_beijing_dataset(raw_path, aggregate_daily=True)
        date_col = "Date"
        target_col = "AQI_Category"
        categorical_cols = ["cbwd", "season"]
    elif dataset_key == "seoul":
        df = load_seoul_dataset(raw_path)
        date_col = "Date"
        target_col = "AQI_Category"
        categorical_cols = ["Station code", "season"]
    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    # 2. Add temporal feature engineering
    df = add_temporal_features(df, date_col=date_col)

    # 3. Generate EDA Visualizations
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)

    plot_missing_values(df, name, f"{plots_dir}/missing_{dataset_key}.png")
    plot_class_distribution(df[target_col], name, f"{plots_dir}/class_dist_{dataset_key}.png", order=AQI_CATEGORIES)
    plot_correlation_heatmap(df, name, f"{plots_dir}/corr_{dataset_key}.png")

    # 4. Chronological train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        df, target_col=target_col, train_ratio=0.8, date_col=date_col
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print("Train Class Distribution:\n", y_train.value_counts().to_dict())

    # Identify numeric and categorical feature columns
    all_cat = [c for c in categorical_cols if c in X_train.columns]
    all_num = [c for c in X_train.columns if c not in all_cat]

    # Preprocessors
    preprocessor_scaled = build_preprocessor(all_num, all_cat, scale=True, impute_strategy="median")
    preprocessor_unscaled = build_preprocessor(all_num, all_cat, scale=False, impute_strategy="median")

    dataset_results = []
    best_overall_f1 = -1.0
    best_overall_info = None

    # Check minimum class size in training data for SMOTE
    min_train_samples = y_train.value_counts().min()

    for imb_method in imbalance_methods:
        for alg in algorithms:
            # Check applicability
            if imb_method == "class_weight" and alg in ["naive_bayes", "knn", "majority_baseline"]:
                # Algorithm does not support class_weight natively
                continue
            if imb_method == "smote" and (alg == "majority_baseline" or min_train_samples < 2):
                continue

            current_alg = alg
            if len(X_train) > 5000 and alg == "svm":
                current_alg = "linear_svc"

            print(f"  --> Running {current_alg:<20} | Imbalance: {imb_method:<12} ...", end=" ", flush=True)

            prep = preprocessor_scaled if requires_scaling(current_alg) else preprocessor_unscaled
            pipe, param_grid = build_full_pipeline(
                preprocessor=prep,
                algorithm=current_alg,
                imbalance_method=imb_method,
                random_state=42,
                min_class_samples=min_train_samples
            )

            # Simplify grids for large datasets (Seoul, India) to ensure fast completion
            if len(X_train) > 5000:
                if "classifier__n_estimators" in param_grid:
                    param_grid["classifier__n_estimators"] = [50]

            try:
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
                record = {
                    "member": member,
                    "dataset": dataset_key,
                    "dataset_name": name,
                    "algorithm": "svm (LinearSVC)" if current_alg == "linear_svc" else alg,
                    "imbalance_method": imb_method,
                    "macro_f1": test_m["macro_f1"],
                    "balanced_acc": test_m["balanced_acc"],
                    "accuracy": test_m["accuracy"],
                    "off_by_2_plus": test_m["off_by_2_plus"],
                    "train_seconds": eval_res["train_seconds"],
                    "cv_mean": eval_res["cv_mean"],
                    "cv_std": eval_res["cv_std"],
                    "best_params": str(eval_res["best_params"]),
                    "per_class": test_m["per_class"]
                }
                dataset_results.append(record)
                print(f"Done! Test Macro F1: {test_m['macro_f1']:.4f} | Bal Acc: {test_m['balanced_acc']:.4f} | Time: {eval_res['train_seconds']:.1f}s")

                # Track best model for confusion matrix and feature importance
                if test_m["macro_f1"] > best_overall_f1:
                    best_overall_f1 = test_m["macro_f1"]
                    best_overall_info = {
                        "algorithm": alg,
                        "imbalance_method": imb_method,
                        "model": eval_res["best_model"],
                        "cm": np.array(test_m["confusion_matrix"]),
                        "class_names": test_m["class_names"]
                    }

            except Exception as e:
                print(f"FAILED with error: {e}")

    # Generate Model Comparison and Confusion Matrix plots
    res_df = pd.DataFrame(dataset_results)
    
    # Standard models comparison plot (method = "none")
    std_df = res_df[res_df["imbalance_method"] == "none"]
    if not std_df.empty:
        plot_model_comparison(std_df, name, f"{plots_dir}/model_compare_{dataset_key}.png")

    # Imbalance strategy comparison plot
    plot_imbalance_comparison(res_df, name, f"{plots_dir}/imbalance_compare_{dataset_key}.png")

    # Best Model Confusion Matrix
    if best_overall_info is not None:
        cm_path = f"results/confusion_matrices/cm_{dataset_key}_{best_overall_info['algorithm']}.png"
        plot_confusion_matrix(
            best_overall_info["cm"],
            best_overall_info["class_names"],
            best_overall_info["algorithm"].replace("_", " ").title(),
            name,
            cm_path
        )

        # Feature Importance if model is tree-based
        best_est = best_overall_info["model"].named_steps.get("classifier")
        if hasattr(best_est, "feature_importances_"):
            preproc = best_overall_info["model"].named_steps.get("preprocessor")
            feature_names = preproc.get_feature_names_out()
            # Clean feature names (remove num__ and cat__)
            clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
            importances = dict(zip(clean_names, best_est.feature_importances_))
            
            fi_path = f"results/feature_importance/fi_{dataset_key}_{best_overall_info['algorithm']}.png"
            plot_feature_importance(
                importances,
                best_overall_info["algorithm"].replace("_", " ").title(),
                name,
                fi_path
            )

    return dataset_results, {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "categorical_cols": all_cat,
        "numeric_cols": all_num
    }


def run_india_ablation_experiment(
    india_data_dict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Scientific Ablation Experiment for India dataset:
    Investigates target leakage by comparing classifiers trained WITH PM2.5/PM10
    vs. WITHOUT PM2.5/PM10.
    """
    print("\n=======================================================")
    print("Running India Ablation Experiment (Target Leakage Analysis)")
    print("=======================================================")

    X_train = india_data_dict["X_train"]
    X_test = india_data_dict["X_test"]
    y_train = india_data_dict["y_train"]
    y_test = india_data_dict["y_test"]
    cat_cols = india_data_dict["categorical_cols"]

    # Ablated feature set: drop PM2.5 and PM10
    ablation_cols = ["PM2.5", "PM10"]
    X_train_ablated = X_train.drop(columns=[c for c in ablation_cols if c in X_train.columns])
    X_test_ablated = X_test.drop(columns=[c for c in ablation_cols if c in X_test.columns])

    num_cols_ablated = [c for c in X_train_ablated.columns if c not in cat_cols]

    preprocessor_scaled = build_preprocessor(num_cols_ablated, cat_cols, scale=True)
    preprocessor_unscaled = build_preprocessor(num_cols_ablated, cat_cols, scale=False)

    algorithms = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
    ablation_results = []

    for alg in algorithms:
        print(f"  --> Ablation testing {alg:<20} without PM2.5 & PM10 ...", end=" ", flush=True)
        prep = preprocessor_scaled if requires_scaling(alg) else preprocessor_unscaled
        pipe, param_grid = build_full_pipeline(prep, alg, imbalance_method="none", random_state=42)

        # Quick evaluation
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
            X_train=X_train_ablated,
            y_train=y_train,
            X_test=X_test_ablated,
            y_test=y_test,
            cv_splits=5
        )
        test_m = eval_res["test_metrics"]
        print(f"Done! Test Macro F1: {test_m['macro_f1']:.4f}")

        ablation_results.append({
            "algorithm": alg,
            "feature_set": "Without PM2.5 & PM10 (Ablated)",
            "macro_f1": test_m["macro_f1"],
            "balanced_acc": test_m["balanced_acc"],
            "accuracy": test_m["accuracy"]
        })

    return ablation_results


def main():
    start_total = time.time()

    # Load configuration
    with open("config/datasets.yaml", "r") as f:
        config = yaml.safe_load(f)

    algorithms = [
        "majority_baseline",
        "logistic_regression",
        "knn",
        "decision_tree",
        "random_forest",
        "svm",
        "naive_bayes",
        "xgboost"
    ]

    imbalance_methods = ["none", "class_weight", "smote"]

    all_experiments = []
    dataset_keys = ["india", "italy", "beijing", "seoul"]
    india_context = None

    for d_key in dataset_keys:
        ds_results, ds_context = run_dataset_experiments(
            dataset_key=d_key,
            config=config,
            algorithms=algorithms,
            imbalance_methods=imbalance_methods
        )
        all_experiments.extend(ds_results)

        if d_key == "india":
            india_context = ds_context

        # Save interim results
        pd.DataFrame(all_experiments).to_csv("results/results.csv", index=False)

    # Run India Ablation Study
    if india_context is not None:
        ablation_res = run_india_ablation_experiment(india_context)
        
        # Add baseline (with PM2.5/PM10) for matching algorithms from results
        extra_rows = []
        for r in ablation_res:
            matching = [x for x in all_experiments if x["dataset"] == "india" and x["algorithm"] == r["algorithm"] and x["imbalance_method"] == "none"]
            if matching:
                extra_rows.append({
                    "algorithm": r["algorithm"],
                    "feature_set": "Full Features (with PM2.5 & PM10)",
                    "macro_f1": matching[0]["macro_f1"],
                    "balanced_acc": matching[0]["balanced_acc"],
                    "accuracy": matching[0]["accuracy"]
                })
        ablation_res.extend(extra_rows)

        plot_ablation_comparison(ablation_res, "results/plots/ablation_india_leakage.png")
        pd.DataFrame(ablation_res).to_csv("results/metrics/india_ablation_results.csv", index=False)

    # Save full results
    results_csv = "results/results.csv"
    results_df = aggregate_team_results(all_experiments, results_csv)

    # Cross-Dataset Comparisons
    print("\n=======================================================")
    print("Generating Team-Level Cross-Dataset Visualizations")
    print("=======================================================")
    plot_cross_dataset_heatmap(results_df, "results/plots/cross_dataset_heatmap_none.png", method="none")
    plot_cross_dataset_heatmap(results_df, "results/plots/cross_dataset_heatmap_class_weight.png", method="class_weight")
    plot_average_algorithm_ranks(results_df, "results/plots/average_algorithm_ranks.png")

    total_time = time.time() - start_total
    print(f"\nALL EXPERIMENTS SUCCESSFULLY COMPLETED IN {total_time/60.0:.2f} MINUTES!")


if __name__ == "__main__":
    main()
