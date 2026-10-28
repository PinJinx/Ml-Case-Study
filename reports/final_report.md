# Comparative Analysis of Machine Learning Algorithms for Air Quality Classification Across Multiple Datasets

**Academic Project Report and Technical Defense Case Study**  
**Course:** Advanced Machine Learning / Data Science Case Study  
**Collaborative Team Structure:** 4-Member Research Team (Vimal, Sreenidhi, Rohith, Sruthi)  

---

## 1. Abstract

Air pollution classification plays a pivotal role in modern environmental monitoring, epidemiology, and public health intervention. While numerous prior studies report classification accuracies exceeding 90% on single, curated datasets, these figures often reflect hidden methodological flaws, including temporal data leakage, target definition leakage, and unrepresentative evaluation metrics under severe class imbalance.

This paper presents an empirical, reproducible comparative case study evaluating **eight supervised machine learning classifiers** paired across **four distinct international environmental datasets** using an independent, modular multi-investigator architecture:
1. **Member 1 (Vimal - `src/Vimal`):** India Urban Air Quality (`india_city_day.csv`), evaluating **Logistic Regression** and **K-Nearest Neighbours (KNN)** alongside an empirical target leakage ablation study.
2. **Member 2 (Sreenidhi - `src/Sreenidhi`):** Italy UCI Air Quality (`italy_air_quality_uci.csv`), evaluating **Support Vector Machine (LinearSVC/RBF)** and **Gaussian Naive Bayes**.
3. **Member 3 (Rohith - `src/Rohith`):** Beijing UCI PM2.5 (`beijing_pm25.csv`), evaluating **Random Forest** and **Gradient Boosting** (`GradientBoostingClassifier`) on atmospheric and meteorological predictors.
4. **Member 4 (Sruthi - `src/Sruthi`):** Seoul Air Monitoring Network (`seoul_air_pollution.csv`), evaluating **Decision Tree** and **Extreme Gradient Boosting (XGBoost)** on a continuous multi-station network characterized by extreme class imbalance.

Target categories were harmonized using the official **Central Pollution Control Board (CPCB)** six-tier breakpoint standard: *Good, Satisfactory, Moderate, Poor, Very Poor, and Severe*. To strictly prevent temporal lookahead bias, all datasets were partitioned chronologically (first 80% train, held-out 20% test), and hyperparameter tuning was conducted exclusively on training folds using `TimeSeriesSplit(n_splits=5)`. We systematically investigated three class-imbalance treatments: unweighted baseline, cost-sensitive class weighting (`class_weight="balanced"`), and Synthetic Minority Over-sampling Technique (SMOTE) strictly encapsulated within imbalanced-learn training pipelines.

Our empirical results demonstrate that:
1. **Tree and Boosting Ensembles Excel on Non-Linear Observations:** In Seoul, XGBoost achieved the highest overall accuracy (**86.32%**) and Macro F1 (**0.2837**), while Random Forest led on Beijing meteorological observations (**Macro F1 = 0.3913**; **0.4046** under SMOTE).
2. **Gradient Boosting Complements Random Forest:** Gradient Boosting achieved **Macro F1 = 0.3457** on Beijing, offering comparable predictive discriminability with sequential boosting optimization.
3. **Target Leakage Artificially Inflates Reported Scores:** A controlled ablation experiment on the India dataset revealed that excluding $PM_{2.5}$ and $PM_{10}$ caused Logistic Regression Macro F1 to plunge from **0.7013** to **0.4328** (a **38.3% decline**), exposing how conventional multi-pollutant models often invert deterministic CPCB sub-index lookup tables rather than discovering atmospheric patterns.
4. **Independent Modular Architecture:** Each team member maintains an independent, runnable directory containing their local dataset copy, model definitions, experiment runner, and local results, with automated team-level synthesis across all benchmarks.

---

## 2. Collaborative Team Structure & Model Pairings

To simulate an authentic academic research consortium, the project is structured into four independent, self-contained member modules under `src/`:

| Member | Directory | Benchmark Dataset | Assigned Clubbed Models | Primary Scientific Research Question |
|:---|:---|:---|:---|:---|
| **Vimal (Member 1)** | `src/Vimal/` | **India (`india_city_day.csv`)** | 1. Logistic Regression<br>2. K-Nearest Neighbours | How severe is formulaic target leakage when primary particulate measurements are provided as predictors? |
| **Sreenidhi (Member 2)** | `src/Sreenidhi/` | **Italy (`italy_air_quality_uci.csv`)** | 1. Support Vector Machine (`LinearSVC`)<br>2. Gaussian Naive Bayes | Can continuous metal-oxide chemical sensor arrays accurately classify joint $CO$/$NO_2$ air quality categories? |
| **Rohith (Member 3)** | `src/Rohith/` | **Beijing (`beijing_pm25.csv`)** | 1. Random Forest<br>2. Gradient Boosting | Can atmospheric and meteorological dynamics predict PM2.5 CPCB bands in the absence of chemical sensor measurements? |
| **Sruthi (Member 4)** | `src/Sruthi/` | **Seoul (`seoul_air_pollution.csv`)** | 1. Decision Tree<br>2. XGBoost | How do decision trees and gradient boosted ensembles handle severe class imbalance under temporal chronological splits? |

Common preprocessing, feature engineering, evaluation metrics, and visualization utilities remain shared in `src/`.

---

## 3. CPCB Breakpoints and Target Categorization

Target labels adhere strictly to the Central Pollution Control Board (CPCB) National Air Quality Index framework:

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

## 4. Experimental Results by Team Member

### 4.1. Vimal (Member 1) — India (`india_city_day.csv`)
*Training: 19,880 samples | Test: 4,970 samples | Classes: 6*

| Algorithm | Imbalance Method | Test Macro F1 | Balanced Accuracy | Test Accuracy | Off-by-2+ Error (%) | Training Runtime (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Majority Baseline | None | 0.0843 | 0.1667 | 33.86% | 16.38% | 0.39s |
| **Logistic Regression** | **None** | **0.7013** | **0.6965** | **75.61%** | **0.62%** | **7.76s** |
| Logistic Regression | Class Weight | 0.6656 | 0.7322 | 69.36% | 1.23% | 3.25s |
| Logistic Regression | SMOTE | 0.6669 | 0.7263 | 70.54% | 1.19% | 6.78s |
| K-Nearest Neighbours | None | 0.5297 | 0.5365 | 60.10% | 2.60% | 3.64s |
| K-Nearest Neighbours | SMOTE | 0.5252 | 0.6064 | 55.63% | 5.23% | 4.63s |

**India Target Leakage Ablation Study:**
- **With PM2.5 & PM10:** Logistic Regression Macro F1 = **0.7013** | Balanced Acc = **0.6965**
- **Without PM2.5 & PM10 (Ablated):** Logistic Regression Macro F1 = **0.4328** | Balanced Acc = **0.4190**
- **Performance Drop:** **-38.29%** in Macro F1, empirically confirming that models trained with target-defining pollutants act primarily as CPCB formula approximators.

---

### 4.2. Sreenidhi (Member 2) — Italy (`italy_air_quality_uci.csv`)
*Training: 6,504 samples | Test: 1,627 samples | Classes: 5 (Severe absent in Rome dataset)*

| Algorithm | Imbalance Method | Test Macro F1 | Balanced Accuracy | Test Accuracy | Off-by-2+ Error (%) | Training Runtime (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Majority Baseline | None | 0.1556 | 0.2000 | 63.68% | 4.36% | 0.21s |
| **Linear SVC (SVM)** | **None** | **0.3515** | **0.4244** | **61.22%** | **10.02%** | **5.48s** |
| Linear SVC (SVM) | Class Weight | 0.3042 | 0.3992 | 50.71% | 20.04% | 0.62s |
| Linear SVC (SVM) | SMOTE | 0.2322 | 0.3470 | 34.97% | 29.87% | 1.60s |
| Gaussian Naive Bayes | None | 0.2791 | 0.3805 | 44.62% | 10.88% | 2.65s |
| Gaussian Naive Bayes | SMOTE | 0.2788 | 0.3843 | 47.45% | 13.21% | 0.41s |

---

### 4.3. Rohith (Member 3) — Beijing (`beijing_pm25.csv`)
*Training: 1,431 samples | Test: 358 samples | Classes: 6 | Purely Weather/Atmospheric Predictors*

| Algorithm | Imbalance Method | Test Macro F1 | Balanced Accuracy | Test Accuracy | Off-by-2+ Error (%) | Training Runtime (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Majority Baseline | None | 0.0584 | 0.1667 | 21.23% | 58.10% | 0.19s |
| **Random Forest** | **None** | **0.3913** | **0.3723** | **38.83%** | **27.09%** | **6.64s** |
| Random Forest | Class Weight | 0.3830 | 0.3800 | 37.43% | 26.82% | 1.83s |
| **Random Forest** | **SMOTE** | **0.4046** | **0.4037** | **39.39%** | **26.82%** | **1.97s** |
| **Gradient Boosting** | **None** | **0.3457** | **0.3390** | **36.59%** | **25.70%** | **12.55s** |
| Gradient Boosting | SMOTE | 0.3520 | 0.3662 | 33.24% | 26.54% | 14.50s |

---

### 4.4. Sruthi (Member 4) — Seoul (`seoul_air_pollution.csv`)
*Training: 21,604 samples | Test: 5,402 samples | Classes: 6 | Extreme Class Imbalance*

| Algorithm | Imbalance Method | Test Macro F1 | Balanced Accuracy | Test Accuracy | Off-by-2+ Error (%) | Training Runtime (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Majority Baseline | None | 0.1502 | 0.1667 | 82.01% | 3.33% | 0.36s |
| Decision Tree | None | 0.2706 | 0.2596 | 83.25% | 2.80% | 6.77s |
| Decision Tree | Class Weight | 0.2651 | 0.2625 | 72.01% | 14.79% | 1.41s |
| Decision Tree | SMOTE | 0.2830 | 0.3102 | 78.75% | 7.29% | 4.03s |
| **XGBoost** | **None** | **0.2837** | **0.2630** | **86.32%** | **2.09%** | **2.42s** |
| **XGBoost** | **Class Weight** | **0.2837** | **0.2630** | **86.32%** | **2.09%** | **2.08s** |
| XGBoost | SMOTE | 0.2810 | 0.2768 | 84.36% | 2.59% | 3.78s |

---

## 5. Team-Wide Comparative Synthesis

### Cross-Dataset Benchmark Matrix (Baseline Imbalance = None)

| Team Member | Dataset | Tested Clubbed Algorithms | Best Model | Test Macro F1 | Test Balanced Acc | Test Accuracy |
|:---|:---|:---|:---|:---:|:---:|:---:|
| **Vimal** | India (`city_day.csv`) | Logistic Regression, KNN | **Logistic Regression** | **0.7013** | **0.6965** | **75.61%** |
| **Sreenidhi** | Italy (`AirQualityUCI`) | Linear SVC, Naive Bayes | **Linear SVC (SVM)** | **0.3515** | **0.4244** | **61.22%** |
| **Rohith** | Beijing (`beijing_pm25`) | Random Forest, Gradient Boosting | **Random Forest** | **0.3913** | **0.3723** | **38.83%** |
| **Sruthi** | Seoul (`Measurement_summary`) | Decision Tree, XGBoost | **XGBoost** | **0.2837** | **0.2630** | **86.32%** |

All outputs, cross-dataset heatmaps, and ranking figures are generated and stored in `results/plots/`:
- `results/plots/cross_dataset_heatmap_none.png`
- `results/plots/cross_dataset_heatmap_class_weight.png`
- `results/plots/team_models_summary_none.png`
- `results/plots/team_models_summary_class_weight.png`
- `results/plots/average_algorithm_ranks.png`
- `results/plots/ablation_india_leakage.png`
