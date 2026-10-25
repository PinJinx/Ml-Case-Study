# Comparative Analysis of Machine Learning Algorithms for Air Quality Classification Across Multiple Datasets

An empirical, multi-dataset Machine Learning case study comparing 7 classifiers across 4 international environmental datasets (India, Italy, Beijing, Seoul) using the official Central Pollution Control Board (CPCB) breakpoint framework.

---

## 1. Project Overview

Air Quality Index (AQI) assessment is a critical public health and environmental monitoring challenge. Rather than treating AQI classification as a synthetic toy problem on a single curated dataset, this case study implements an academic, collaborative, four-member team architecture where each team member independently analyzes an international benchmark dataset using a shared, standardized experimental methodology:

1. **Member 1 (India):** Daily urban air quality across Indian cities (`city_day.csv`), evaluating multi-pollutant measurements and target leakage.
2. **Member 2 (Italy):** Hourly gas multisensor response arrays from the UCI Air Quality dataset in Rome, classifying air quality bands derived from $CO$ and $NO_2$.
3. **Member 3 (Beijing):** Hourly meteorological and PM2.5 observations from UCI Beijing PM2.5 dataset, classifying PM2.5-based CPCB categories from purely atmospheric/weather predictors.
4. **Member 4 (Seoul):** Multi-station continuous air monitoring network across 25 Seoul districts, evaluating extreme class imbalance on PM2.5 categories.

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

```text
aqi-ml-case-study/
├── README.md                                # Comprehensive project documentation & viva guide
├── requirements.txt                         # Pinned Python package dependencies
├── run_all_experiments.py                   # Automated end-to-end benchmark execution script
├── finish_ablation_and_visualizations.py    # Target leakage ablation & cross-dataset visualizer
├── config/
│   └── datasets.yaml                        # Dataset metadata, column schemas, and paths
├── data/
│   ├── raw/                                 # Raw downloaded dataset files
│   │   ├── india_city_day.csv
│   │   ├── italy_air_quality_uci.csv
│   │   ├── beijing_pm25.csv
│   │   └── seoul_air_pollution.csv
│   └── processed/                           # Cleaned, chronologically sorted, preprocessed data
├── src/
│   ├── __init__.py
│   ├── labeling.py                          # Shared CPCB breakpoint classifiers & ordinal error metrics
│   ├── data_loader.py                       # Modular schema-validated loaders for all 4 datasets
│   ├── feature_engineering.py               # Temporal, cyclical, and meteorological feature engineering
│   ├── preprocessing.py                     # Chronological train/test split & scikit-learn ColumnTransformers
│   ├── models.py                            # 7 ML estimators + majority baseline + SMOTE/weighting pipelines
│   ├── evaluation.py                        # TimeSeriesSplit GridSearchCV, Macro F1, confusion matrices
│   ├── visualization.py                     # Publication-quality plotting routines (300 DPI)
│   └── comparison.py                        # Cross-dataset aggregation, heatmap, rank, and ablation plots
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb   # Interactive EDA, distributions, missing values
│   ├── 02_preprocessing.ipynb               # Splitting, pipeline verification, leakage checks
│   ├── 03_model_training.ipynb              # Model fitting, hyperparameter grids, tuning logs
│   ├── 04_evaluation.ipynb                  # Confusion matrices, per-class F1, off-by-2+ errors
│   └── 05_team_comparison.ipynb             # Cross-dataset synthesis, rankings, ablation discussion
├── results/
│   ├── results.csv                          # Combined 80-run team experimental benchmark
│   ├── confusion_matrices/                  # Normalized confusion matrix plots per dataset
│   ├── feature_importance/                  # Gini/MDI feature importance plots for tree ensembles
│   ├── metrics/                             # Detailed CSVs (ablation results, per-class metrics)
│   └── plots/                               # EDA, comparison bar charts, heatmaps, rank plots
└── reports/
    └── final_report.md                      # Complete 14-section academic paper & viva reference
```

---

## 4. Environment Setup and Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit)
- PowerShell (Windows) or Bash (Linux/macOS)

### Step 1: Clone or Navigate to Project
```powershell
cd C:\Users\LENOVO\Desktop\ML\aqi-ml-case-study
```

### Step 2: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

---

## 5. Execution Instructions

### Running All Benchmarks in One Command:
To reproduce all 80 experimental runs across all 4 datasets, execute the ablation experiment, and generate all 300 DPI publication plots:
```powershell
python run_all_experiments.py
```

### Running Interactive Jupyter Notebooks:
Launch Jupyter Lab or Notebook:
```powershell
jupyter notebook notebooks/
```
Open each notebook sequentially from `01_exploratory_data_analysis.ipynb` through `05_team_comparison.ipynb`.

---

## 6. Key Empirical Findings Summary

| Dataset | Best Algorithm | Optimal Imbalance Strategy | Test Macro F1 | Balanced Accuracy | Overall Accuracy | Off-by-2+ Error Rate | Train Time |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **India** (`city_day`) | **XGBoost** | `class_weight` | **0.7832** | 0.7679 | 81.95% | **0.28%** | 5.4s |
| **Italy** (`AirQualityUCI`) | **XGBoost** | `class_weight` | **0.5639** | 0.5666 | 76.58% | **0.74%** | 1.6s |
| **Beijing** (`PRSA PM2.5`) | **Logistic Regression** | `class_weight` | **0.4345** | 0.4649 | 41.62% | 17.88% | 0.4s |
| **Seoul** (`Measurement`) | **Random Forest** | `class_weight` | **0.3064** | 0.2895 | 85.63% | **2.15%** | 1.8s |

### Scientific Ablation Result (India Target Leakage):
- **With PM2.5 & PM10:** XGBoost achieved **0.7832** Macro F1.
- **Without PM2.5 & PM10 (Ablated):** XGBoost dropped to **0.4405** Macro F1 (a severe drop of **34.27 percentage points**).
- *Takeaway:* Predictor sets containing pollutants directly used in formulaic target definition cause massive target leakage.

---

## 7. Viva Voce & Academic Defense Preparation Guide

This section provides clear, technically precise, beginner-friendly explanations to standard academic viva questions.

### Q1: Why is Macro F1 chosen as the primary evaluation metric instead of standard Accuracy?
> **Answer:** Standard Accuracy is severely misleading in class-imbalanced domains. In environmental monitoring, hazardous events ("Severe" or "Very Poor") occur infrequently (e.g., in our Seoul dataset, "Severe" accounts for only 19 out of 27,006 observations). A trivial majority-class classifier that always predicts "Good" achieves over 85% accuracy while completely failing to identify life-threatening pollution episodes (Recall = 0.0, Macro F1 = 0.15). Macro F1 computes the unweighted arithmetic mean of F1-scores across all classes, giving equal importance to rare extreme events and dominant benign days.

### Q2: Why is random k-fold cross-validation forbidden for time-series datasets?
> **Answer:** Shuffling temporal data causes **temporal lookahead leakage** (data snooping). If an algorithm trains on observations from December 2018 and tests on January 2018, it leverages future seasonal, meteorological, and autoregressive information that would never be available in real-world deployment. We strictly use chronological splitting (first 80% train, last 20% test) and `TimeSeriesSplit(n_splits=5)` for hyperparameter tuning, where each fold only trains on past observations and validates on the immediate successive temporal window.

### Q3: Why is Median Imputation preferred over Mean Imputation?
> **Answer:** Environmental pollutant concentrations (such as $SO_2$, $NO_x$, and $PM_{2.5}$) follow heavily right-skewed, log-normal distributions with acute outlier spikes during firework events, agricultural burning, or dust storms. The sample mean is highly sensitive to extreme outliers, which inflates imputed baseline values. The median is robust against extreme positive skewness and represents the true typical ambient baseline.

### Q4: Why must feature scaling be applied to Logistic Regression, KNN, and SVM, but omitted for Tree-based models?
> **Answer:** 
> - **K-Nearest Neighbours** relies on Euclidean distance; features with larger magnitudes (e.g., $PM_{10}$ in hundreds of $\mu g/m^3$) dominate features with small numerical ranges (e.g., $CO$ in $mg/m^3$).
> - **Support Vector Machines (RBF)** compute dot products and radial kernel distances; unscaled features distort the margin geometry.
> - **Logistic Regression** with $L_2$ regularization penalizes large coefficients; unscaled features receive disproportionate, unfair regularization penalties.
> - **Decision Trees, Random Forests, and XGBoost** are monotonically scale-invariant: a decision threshold split on $X_j \le \theta$ is identical whether $X_j$ is raw or z-score scaled.

### Q5: Why must SMOTE be embedded strictly inside the training folds of an Imbalanced-Learn Pipeline?
> **Answer:** If SMOTE (Synthetic Minority Over-sampling Technique) is applied to the whole dataset prior to cross-validation or train/test splitting, synthetic samples are generated using convex combinations of training and testing instances. Information from the test partition leaks directly into the training partition, creating artificial test performance that collapses in production. Placing SMOTE inside an `imblearn.pipeline.Pipeline` guarantees that oversampling occurs strictly within the designated training fold.

### Q6: What does the "Off-by-2+" metric measure, and why is it essential for AQI classification?
> **Answer:** Standard classification metrics (Accuracy, Macro F1) treat all classification errors equally: misclassifying "Good" as "Satisfactory" is penalized identically to misclassifying "Severe" as "Good". In environmental protection and medical alert systems, AQI categories are ordinal. Predicting "Good" when air is "Severe" poses extreme health risks to asthmatics and cardiopulmonary patients. The "Off-by-2+" metric explicitly quantifies the percentage of errors that deviate by two or more ordinal ranks, directly penalizing dangerous misclassifications.

### Q7: What did our India ablation experiment prove regarding target leakage?
> **Answer:** The official CPCB NAQI algorithm calculates the overall AQI bucket by computing individual pollutant sub-indices and taking the maximum ($AQI = \max(I_{PM2.5}, I_{PM10}, I_{NO2}, \dots)$). In India, particulate matter ($PM_{2.5}$ and $PM_{10}$) is the responsible maximum sub-index in over 85% of observations. When $PM_{2.5}$ and $PM_{10}$ are provided as input features, models simply reconstruct the deterministic CPCB lookup formula, achieving an artificially inflated Macro F1 of **0.7832**. When ablated (removed), XGBoost performance drops to **0.4405**. This empirically proves that models with raw target-defining pollutants evaluate lookup reconstruction rather than genuine meteorological-atmospheric pattern recognition.

### Q8: Why did the optimal algorithm differ across datasets (e.g., XGBoost on India/Italy vs. Logistic Regression on Beijing vs. Random Forest on Seoul)?
> **Answer:** The optimal algorithm reflects dataset-specific feature distributions, signal-to-noise ratios, and class structures:
> - **India and Italy** feature continuous multi-pollutant and chemical sensor arrays with complex nonlinear interactions between precursors ($NO_x$, volatile organics) and secondary pollutants, which gradient boosting (XGBoost) models exceptionally well.
> - **Beijing** predicts $PM_{2.5}$ categories solely from macro-weather features (temperature, pressure, dew point, wind direction) where the relationship between wind flushing and particulate buildup is broadly linear and noisy. Complex trees easily overfit noisy meteorological variations, allowing regularized linear models (Logistic Regression with class weighting) to achieve superior test generalization.
> - **Seoul** features massive sample size ($N = 27,006$) with acute class sparsity (only 7 Severe cases in training). Random Forest with balanced class weights provides stable, low-variance bagging ensembles that resist extreme minority class outliers better than individual deep trees or sensitive linear boundaries.
#   M l - C a s e - S t u d y  
 