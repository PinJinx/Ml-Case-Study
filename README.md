# Comparative Analysis of Machine Learning Algorithms for Air Quality Classification Across Multiple Datasets

An empirical, multi-dataset Machine Learning case study comparing **8 classifiers** paired across **4 international environmental datasets** (India, Italy, Beijing, Seoul) using the official Central Pollution Control Board (CPCB) breakpoint framework.

---

## 1. Project Overview

Air Quality Index (AQI) assessment is a critical public health and environmental monitoring challenge. Rather than treating AQI classification as a synthetic toy problem on a single curated dataset, this case study implements an academic, collaborative, four-member team architecture where each team member independently analyzes an international benchmark dataset with **two clubbed conceptually similar models**:

1. **Vimal (Member 1 - India):** Daily urban air quality across Indian cities (`india_city_day.csv`), evaluating multi-pollutant measurements and target leakage.
   - **Clubbed Models:** Logistic Regression & K-Nearest Neighbours (KNN)
   - **Location:** `src/Vimal/`
2. **Sreenidhi (Member 2 - Italy):** Hourly gas multisensor response arrays from the UCI Air Quality dataset in Rome (`italy_air_quality_uci.csv`), classifying air quality bands derived from $CO$ and $NO_2$.
   - **Clubbed Models:** Support Vector Machine (LinearSVC/RBF) & Gaussian Naive Bayes
   - **Location:** `src/Sreenidhi/`
3. **Rohith (Member 3 - Beijing):** Hourly meteorological observations from UCI Beijing PM2.5 dataset (`beijing_pm25.csv`), classifying PM2.5-based CPCB categories from purely atmospheric/weather predictors.
   - **Clubbed Models:** Random Forest & Gradient Boosting (`GradientBoostingClassifier`)
   - **Location:** `src/Rohith/`
4. **Sruthi (Member 4 - Seoul):** Multi-station continuous air monitoring network across 25 Seoul districts (`seoul_air_pollution.csv`), evaluating extreme class imbalance on PM2.5 categories.
   - **Clubbed Models:** Decision Tree & Extreme Gradient Boosting (`XGBClassifier`)
   - **Location:** `src/Sruthi/`

All models are tuned using chronological 5-fold cross-validation (`TimeSeriesSplit`) on an 80% chronological train partition and tested on a held-out 20% future partition.

---

## 2. Target Categories and CPCB Breakpoints

The target labels represent the six official categories defined by the Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change (MoEFCC), Government of India:

| Category ID | AQI Category | CPCB PM2.5 Range (24-hr, $\mu g/m^3$) | CPCB NO2 Range (24-hr, $\mu g/m^3$) | CPCB CO Range (8-hr, $mg/m^3$) | Health Impact Description |
|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | **Good** | $0 - 30$ | $0 - 40$ | $0.0 - 1.0$ | Minimal impact |
| **1** | **Satisfactory** | $31 - 60$ | $41 - 80$ | $1.1 - 2.0$ | Minor breathing discomfort to sensitive people |
| **2** | **Moderate** | $61 - 90$ | $81 - 180$ | $2.1 - 10.0$ | Breathing discomfort to the people with lungs, asthma and heart diseases |
| **3** | **Poor** | $91 - 120$ | $181 - 280$ | $10.1 - 17.0$ | Breathing discomfort to most people on prolonged exposure |
| **4** | **Very Poor** | $121 - 250$ | $281 - 400$ | $17.1 - 34.0$ | Respiratory illness on prolonged exposure |
| **5** | **Severe** | $> 250$ | $> 400$ | $> 34.0$ | Affects healthy people and seriously impacts those with existing diseases |

*Reference: Central Pollution Control Board (CPCB), "National Air Quality Index," Ministry of Environment, Forest & Climate Change, New Delhi, 2014.*

---

## 3. Project Directory Structure

Each team member has an independent folder containing their dataset, clubbed models, runner script, and dedicated results, while sharing common preprocessing, data loaders, labeling, and visualization modules:

```text
aqi-ml-case-study/
│── README.md                                # Comprehensive documentation & project guide
│── requirements.txt                         # Pinned Python package dependencies
│── run_all_experiments.py                   # Master orchestrator executing all member benchmarks
│── finish_ablation_and_visualizations.py    # Cross-dataset team aggregator & visualizer
│── config/
│   └── datasets.yaml                        # Dataset metadata, column schemas, and mappings
│── reports/
│   └── final_report.md                      # Comprehensive academic case study report
│── results/                                 # Team-level consolidated results
│   ├── results.csv                          # Full experimental metrics across all members
│   ├── confusion_matrices/                  # Normalized confusion matrices
│   ├── feature_importance/                  # Top feature importances
│   ├── metrics/                             # summary_metrics.csv, ablation results
│   └── plots/                               # Cross-dataset heatmaps, rankings, summaries
└── src/
    ├── labeling.py                          # Common CPCB breakpoints & ordinal error metrics
    ├── data_loader.py                       # Modular schema-validated loaders for all 4 datasets
    ├── feature_engineering.py               # Common temporal & cyclical feature engineering
    ├── preprocessing.py                     # Common chronological split & ColumnTransformers
    ├── evaluation.py                        # Common TimeSeriesSplit & metric evaluation engine
    ├── visualization.py                     # Common plotting functions (EDA, CM, FI, bar charts)
    ├── comparison.py                        # Common cross-dataset aggregation & rank plotting
    ├── models.py                            # Central unified model interface
    │
    ├── Vimal/                               # Member 1 (India: city_day.csv)
    │   ├── data/india_city_day.csv          # Local dataset copy
    │   ├── models.py                        # Logistic Regression & KNN
    │   ├── run_experiments.py               # Independent benchmark runner (+ ablation study)
    │   └── results/                         # Local results, metrics CSV, CMs, and plots
    │
    ├── Sreenidhi/                           # Member 2 (Italy: italy_air_quality_uci.csv)
    │   ├── data/italy_air_quality_uci.csv   # Local dataset copy
    │   ├── models.py                        # Support Vector Machine (LinearSVC/RBF) & Naive Bayes
    │   ├── run_experiments.py               # Independent benchmark runner
    │   └── results/                         # Local results, metrics CSV, CMs, and plots
    │
    ├── Rohith/                              # Member 3 (Beijing: beijing_pm25.csv)
    │   ├── data/beijing_pm25.csv            # Local dataset copy
    │   ├── models.py                        # Random Forest & Gradient Boosting
    │   ├── run_experiments.py               # Independent benchmark runner
    │   └── results/                         # Local results, metrics CSV, CMs, and plots
    │
    └── Sruthi/                              # Member 4 (Seoul: seoul_air_pollution.csv)
        ├── data/seoul_air_pollution.csv     # Local dataset copy
        ├── models.py                        # Decision Tree & XGBoost
        ├── run_experiments.py               # Independent benchmark runner
        └── results/                         # Local results, metrics CSV, CMs, and plots
```

---

## 4. How to Run

### Running Individual Member Benchmarks Independently
Each team member can run their experiments directly from their assigned directory:

```bash
# Member 1: Vimal (India: Logistic Regression & KNN + Ablation Study)
cd src/Vimal
python run_experiments.py

# Member 2: Sreenidhi (Italy: SVM & Naive Bayes)
cd src/Sreenidhi
python run_experiments.py

# Member 3: Rohith (Beijing: Random Forest & Gradient Boosting)
cd src/Rohith
python run_experiments.py

# Member 4: Sruthi (Seoul: Decision Tree & XGBoost)
cd src/Sruthi
python run_experiments.py
```
*All metrics CSVs, confusion matrices, and model comparison figures will be generated and saved directly inside that member's `results/` folder.*

### Running the Full Team Benchmark & Synthesis
To run all 4 member benchmarks sequentially and generate the complete consolidated cross-dataset comparison:

```bash
python run_all_experiments.py
```

To quickly aggregate existing member results and regenerate all team plots without re-training:
```bash
python finish_ablation_and_visualizations.py
```

---

## 5. Team Benchmark Results Summary

### Standard Imbalance Baseline (`imbalance_method = "none"`)

| Member | Dataset | Algorithm | Test Macro F1 | Balanced Acc | Accuracy | Off-by-2+ Error (%) | Training Time (s) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Vimal** | India (`city_day.csv`) | **Logistic Regression** | **0.7013** | **0.6965** | **75.61%** | 0.62% | 7.76s |
| **Vimal** | India (`city_day.csv`) | K-Nearest Neighbours | 0.5297 | 0.5365 | 60.10% | 2.60% | 3.64s |
| **Vimal** | India (`city_day.csv`) | Majority Baseline | 0.0843 | 0.1667 | 33.86% | 16.38% | 0.39s |
| **Sreenidhi** | Italy (`AirQualityUCI`) | **Linear SVC (SVM)** | **0.3515** | **0.4244** | **61.22%** | 10.02% | 5.48s |
| **Sreenidhi** | Italy (`AirQualityUCI`) | Gaussian Naive Bayes | 0.2791 | 0.3805 | 44.62% | 10.88% | 2.65s |
| **Sreenidhi** | Italy (`AirQualityUCI`) | Majority Baseline | 0.1556 | 0.2000 | 63.68% | 4.36% | 0.21s |
| **Rohith** | Beijing (`beijing_pm25`) | **Random Forest** | **0.3913** | **0.3723** | **38.83%** | 27.09% | 6.64s |
| **Rohith** | Beijing (`beijing_pm25`) | Gradient Boosting | 0.3457 | 0.3390 | 36.59% | 25.70% | 12.55s |
| **Rohith** | Beijing (`beijing_pm25`) | Majority Baseline | 0.0584 | 0.1667 | 21.23% | 58.10% | 0.19s |
| **Sruthi** | Seoul (`Measurement_summary`) | **XGBoost** | **0.2837** | **0.2630** | **86.32%** | 2.09% | 2.42s |
| **Sruthi** | Seoul (`Measurement_summary`) | Decision Tree | 0.2706 | 0.2596 | 83.25% | 2.80% | 6.77s |
| **Sruthi** | Seoul (`Measurement_summary`) | Majority Baseline | 0.1502 | 0.1667 | 82.01% | 3.33% | 0.36s |
