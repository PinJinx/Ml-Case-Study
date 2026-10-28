"""
Master Experiment Runner
Executes all 4 independent member benchmarks and aggregates results into a consolidated team comparison.
"""

import os
import sys
import shutil
import subprocess
import pandas as pd

from src.comparison import plot_team_models_summary, plot_cross_dataset_heatmap

MEMBERS = ["Vimal", "Sreenidhi", "Rohith", "Sruthi"]


def run_all_benchmarks():
    """Runs each member's experiment script in their respective directory."""
    print("=" * 65)
    print("  RUNNING ALL 4 TEAM MEMBER BENCHMARKS")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for member in MEMBERS:
        member_dir = os.path.join(base_dir, "src", member)
        script_path = os.path.join(member_dir, "run_experiments.py")

        if os.path.exists(script_path):
            print(f"\n>>> Executing benchmark for {member}...")
            subprocess.run([sys.executable, "run_experiments.py"], cwd=member_dir, check=True)
        else:
            print(f"Warning: Script not found at {script_path}")


def aggregate_results():
    """Combines individual member CSVs into results/results.csv and generates team plots."""
    print("\n" + "=" * 65)
    print("  AGGREGATING TEAM RESULTS")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    metrics_dir = os.path.join(results_dir, "metrics")
    cm_dir = os.path.join(results_dir, "confusion_matrices")
    fi_dir = os.path.join(results_dir, "feature_importance")

    for d in [results_dir, plots_dir, metrics_dir, cm_dir, fi_dir]:
        os.makedirs(d, exist_ok=True)

    dfs = []
    for member in MEMBERS:
        csv_path = os.path.join(base_dir, "src", member, "results", f"{member.lower()}_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"Loaded {len(df)} runs from {member}")

    if not dfs:
        print("Error: No results found to aggregate!")
        return

    full_results = pd.concat(dfs, ignore_index=True)
    full_csv = os.path.join(results_dir, "results.csv")
    full_results.to_csv(full_csv, index=False)
    print(f"Consolidated results saved to: {full_csv}")

    # Copy member plots to global results folder
    for member in MEMBERS:
        mem_results_dir = os.path.join(base_dir, "src", member, "results")
        if not os.path.exists(mem_results_dir):
            continue
        for file_name in os.listdir(mem_results_dir):
            src_file = os.path.join(mem_results_dir, file_name)
            if file_name.startswith("cm_"):
                shutil.copy(src_file, os.path.join(cm_dir, file_name))
            elif file_name.startswith("fi_"):
                shutil.copy(src_file, os.path.join(fi_dir, file_name))
            elif file_name.endswith(".png"):
                shutil.copy(src_file, os.path.join(plots_dir, file_name))

    # Generate cross-dataset summary figures
    print("\nGenerating Team Comparison Visualizations...")
    plot_team_models_summary(full_results, os.path.join(plots_dir, "team_models_summary_none.png"), method="none")
    plot_cross_dataset_heatmap(full_results, os.path.join(plots_dir, "cross_dataset_heatmap_none.png"), method="none")

    # Save summary metrics table
    summary_cols = ["member", "dataset_name", "algorithm", "macro_f1", "balanced_acc", "accuracy", "train_seconds"]
    summary_df = full_results[summary_cols]
    summary_df.to_csv(os.path.join(metrics_dir, "summary_metrics.csv"), index=False)

    print("\n" + "=" * 65)
    print("  TEAM-WIDE BENCHMARK SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65)
    print(f"\nAll team outputs saved to: {results_dir}\n")


def main():
    # If called with --aggregate, only aggregates existing results without re-running
    if len(sys.argv) > 1 and sys.argv[1] == "--aggregate":
        aggregate_results()
    else:
        run_all_benchmarks()
        aggregate_results()


if __name__ == "__main__":
    main()
