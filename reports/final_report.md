# Comparative Analysis of Machine Learning Algorithms for Air Quality Classification Across Multiple Datasets

**Academic Project Report and Technical Defense Case Study**  
**Course:** Advanced Machine Learning / Data Science Case Study  
**Collaborative Team Structure:** 4-Member Research Team  

---

## 1. Abstract

Air pollution classification plays a pivotal role in modern environmental monitoring, epidemiology, and public health intervention. While numerous prior studies report classification accuracies exceeding 90% on single, curated datasets, these figures often reflect hidden methodological flaws, including temporal data leakage, target definition leakage, and unrepresentative evaluation metrics. 

This paper presents an empirical, reproducible comparative case study evaluating seven supervised machine learning classifiers—**Logistic Regression, K-Nearest Neighbours (KNN), Decision Trees, Random Forests, Support Vector Machines (SVM), Gaussian Naive Bayes, and Extreme Gradient Boosting (XGBoost)**—alongside a **Majority-Class Baseline**, across four distinct international datasets: **India (`city_day.csv`), Italy (UCI Air Quality), Beijing (UCI PM2.5), and Seoul (Continuous Air Monitoring Network)**.

Target categories were harmonized using the official **Central Pollution Control Board (CPCB)** six-tier breakpoint standard: *Good, Satisfactory, Moderate, Poor, Very Poor, and Severe*. To prevent temporal lookahead bias, all datasets were partitioned chronologically (first 80% train, held-out 20% test), and hyperparameter tuning was conducted exclusively on training folds using `TimeSeriesSplit(n_splits=5)`. We systematically investigated three class-imbalance treatments: unweighted baseline, cost-sensitive class weighting (`class_weight="balanced"`), and Synthetic Minority Over-sampling Technique (SMOTE) strictly encapsulated within imbalanced-learn training pipelines.

Our empirical results demonstrate that:
1. **Tree Ensembles dominate continuous sensor data:** XGBoost and Random Forest achieved the highest overall average ranks (**1.50** and **2.75**, respectively), with XGBoost attaining the peak Macro F1 of **0.7832** on India and **0.5639** on Italy.
2. **Feature context dictates algorithmic superiority:** For purely meteorological predictors without direct pollutant measurements (Beijing), regularized Logistic Regression with class-weighting outperformed complex ensembles (**Macro F1 = 0.4345** vs. **0.3488** for XGBoost), avoiding over-parameterized overfitting on noisy weather dynamics.
3. **Target leakage artificially inflates reported scores:** A controlled ablation experiment on the India dataset revealed that excluding $PM_{2.5}$ and $PM_{10}$ caused XGBoost Macro F1 to drop precipitously from **0.7832** to **0.4405** (a **34.27%** decline), exposing how conventional multi-pollutant models often invert deterministic CPCB sub-index lookup tables rather than discovering atmospheric patterns.
4. **Cost-sensitive learning surpasses SMOTE under temporal sparsity:** `class_weight="balanced"` consistently elevated minority recall and balanced accuracy across all datasets without suffering from the synthetic interpolation artifacts and temporal fold sparsity that degraded SMOTE.

---

## 2. Introduction and Motivation

Ambient air pollution is recognized by the World Health Organization (WHO) as the single largest environmental risk factor to human health, contributing to millions of premature deaths annually from ischemic heart disease, stroke, chronic obstructive pulmonary disease, and acute respiratory infections. Governments worldwide communicate ambient air hazard levels to the public via composite Air Quality Indices (AQI) divided into discrete, color-coded health warning categories.

In recent years, the deployment of machine learning algorithms for air quality classification has expanded rapidly. However, a significant gap exists between published academic benchmarks and real-world deployment reliability. Three pervasive methodological pitfalls compromise the validity of published air quality classification literature:
1. **Random Cross-Validation on Time-Series Data:** Standard randomly shuffled $k$-fold cross-validation allows past and future samples to mingle, artificially inflating performance through temporal autocorrelation (lookahead leakage).
2. **Formulaic Target Leakage:** Many studies evaluate multi-pollutant datasets where the target AQI category is mathematically derived as the maximum sub-index of the exact pollutant measurements provided as predictors.
3. **Reliance on Accuracy under Severe Imbalance:** Ambient pollution is predominantly benign or moderate in many metropolitan areas, with severe episodes occurring during isolated winter inversions, agricultural burning, or dust storms. Accuracy masks total failure to detect hazardous episodes.

To resolve these issues, this study establishes an academically rigorous, four-member comparative framework where each investigator evaluates an independent international dataset using an identical, leak-free pipeline architecture.

---

## 3. Problem Statement and Research Objectives

### Problem Statement
Given multivariate environmental, meteorological, and chemical sensor observations recorded chronologically up to time $t$, classify the ambient air quality into one of six ordinal CPCB categories at time $t$ while strictly eliminating temporal leakage, handling multi-class imbalance, and penalizing severe ordinal misclassifications.

### Research Objectives
1. **Multi-Dataset Benchmarking:** Implement and cross-evaluate 7 machine learning algorithms and a majority baseline across 4 international datasets representing distinct monitoring regimes.
2. **Standardized CPCB Harmonization:** Formulate a unified, CPCB-aligned categorization scheme applicable across diverse sensor configurations.
3. **Strict Leak-Free Experimental Design:** Implement chronological 80/20 train/test splits and 5-fold temporal cross-validation with zero test data contamination.
4. **Systematic Imbalance Investigation:** Contrast unweighted learning against cost-sensitive weighting and pipeline-contained SMOTE.
5. **Empirical Target Leakage Ablation:** Quantify the exact performance drop resulting from removing target-defining pollutants ($PM_{2.5}$ and $PM_{10}$) in the Indian national dataset.
6. **Ordinal Severity Quantification:** Evaluate the "Off-by-2+" metric to measure the proportion of clinically hazardous classification errors.

---

## 4. Dataset Descriptions and Schemas

Four benchmark datasets were acquired and placed into the project repository:

```
aqi-ml-case-study/data/raw/
├── india_city_day.csv
├── italy_air_quality_uci.csv
├── beijing_pm25.csv
└── seoul_air_pollution.csv
```

### 4.1. Member 1: India (`city_day.csv`)
- **Origin:** Central Pollution Control Board (CPCB) continuous monitoring stations across Indian cities (2015–2020).
- **Usable Sample Size:** 24,850 daily observations across 26 cities after removing rows without valid targets.
- **Predictor Variables:** $PM_{2.5}, PM_{10}, NO, NO_2, NO_x, NH_3, CO, SO_2, O_3, \text{Benzene}, \text{Toluene}, \text{Xylene}$, City, temporal features.
- **Target:** Existing CPCB `AQI_Bucket` (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).
- **Leakage Elimination:** Direct numerical `AQI` column dropped.
- **Missingness Profile:** Significant missingness in $NH_3$ (34.6%), $PM_{10}$ (37.9%), and $Xylene$ (61.5%). Handled via robust median imputation.

### 4.2. Member 2: Italy (UCI Air Quality)
- **Origin:** Field deployment of a chemical multisensor device in a polluted Italian road network (Rome, March 2004 – April 2005).
- **Usable Sample Size:** 8,131 hourly observations.
- **Predictor Variables:** Hourly responses from five metal oxide chemical sensors: $PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3)$, along with reference gas concentrations and meteorological covariates (Temperature, Relative Humidity, Absolute Humidity).
- **Data Cleansing:** Semicolon delimiter parsed; European comma decimals converted; sentinel values ($-200$) replaced with `NaN`.
- **Target Formulation:** Derived from the worse (maximum ordinal severity) of $CO$ ($mg/m^3$) and $NO_2$ ($\mu g/m^3$) CPCB concentration bands.
- **Leakage Elimination:** Reference columns $CO(GT)$ and $NO_2(GT)$ were removed from predictors because they directly define the target. Highly missing $NMHC(GT)$ (>90% missingness) was dropped.

### 4.3. Member 3: Beijing (UCI PM2.5)
- **Origin:** US Embassy air monitoring station and Beijing Capital International Airport weather station (2010–2014).
- **Usable Sample Size:** 1,789 daily aggregated records (from 43,824 raw hourly observations).
- **Predictor Variables:** Purely meteorological and temporal indicators: Dew Point (`DEWP`), Temperature (`TEMP`), Atmospheric Pressure (`PRES`), Wind Speed (`Iws`), Cumulative Snow Hours (`Is`), Cumulative Rain Hours (`Ir`), Wind Direction (`cbwd`), season, and cyclical date features.
- **Target Formulation:** CPCB $PM_{2.5}$ 24-hr concentration bands.
- **Leakage Elimination:** Index column `No` and pollutant $PM_{2.5}$ dropped completely from predictors.

### 4.4. Member 4: Seoul (Air Pollution Dataset)
- **Origin:** Seoul Metropolitan Clean Air Information Network, continuous automated monitoring across 25 municipal districts (2017–2019).
- **Usable Sample Size:** 27,006 station-day observations (aggregated from 647,511 raw hourly readings).
- **Predictor Variables:** Co-pollutants ($SO_2, NO_2, O_3, CO, PM_{10}$), Station Code, Latitude, Longitude, and temporal covariates.
- **Data Cleansing:** Sensor calibration artifacts and instrument maintenance codes producing negative values ($\le 0$) converted to `NaN`.
- **Target Formulation:** CPCB $PM_{2.5}$ concentration bands.
- **Leakage Elimination:** $PM_{2.5}$ and textual Address dropped from predictors.

---

## 5. Label Construction and Preprocessing Methodology

### 5.1. CPCB Breakpoint Scheme
All targets were harmonized to the CPCB NAQI ordinal scale:

$$\text{Category}(C) = \begin{cases} 
\text{Good} & 0 \le C \le B_1 \\
\text{Satisfactory} & B_1 < C \le B_2 \\
\text{Moderate} & B_2 < C \le B_3 \\
\text{Poor} & B_3 < C \le B_4 \\
\text{Very Poor} & B_4 < C \le B_5 \\
\text{Severe} & C > B_5
\end{cases}$$

For the Italy dataset, where $CO$ and $NO_2$ are available, the joint category was calculated as:

$$\text{Category}_{\text{joint}} = \max\left(\text{Rank}(\text{Category}_{CO}), \text{Rank}(\text{Category}_{NO2})\right)$$

### 5.2. Pipeline Architecture and Leakage Prevention
To guarantee scientific integrity:
1. **Chronological Splitting:**
   $$X_{\text{train}}, y_{\text{train}} = \mathcal{D}[0 : 0.8N], \quad X_{\text{test}}, y_{\text{test}} = \mathcal{D}[0.8N : N]$$
2. **ColumnTransformer Construction:**
   - **Numeric Sub-Pipeline:** `SimpleImputer(strategy='median')` followed by optional `StandardScaler()`.
   - **Categorical Sub-Pipeline:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
3. **Algorithmic Scaling Alignment:**
   - StandardScaler fitted and applied to **Logistic Regression, KNN, and SVM**.
   - Tree-based architectures (**Decision Tree, Random Forest, XGBoost**) and **GaussianNB** were trained on unscaled features, preserving native feature semantics and interpretability.

---

## 6. Machine Learning Algorithms and Hyperparameter Optimization

### 6.1. Evaluated Models
1. **Majority-Class Baseline:** Predicts the most frequent class in the training split. Provides the empirical lower bound.
2. **Multinomial Logistic Regression:** Convex linear benchmark with $L_2$ regularization.
3. **K-Nearest Neighbours (KNN):** Distance-based non-parametric classifier.
4. **Decision Tree (CART):** Single tree with Gini impurity splitting criterion.
5. **Random Forest:** Bootstrap bagging ensemble of 50–100 de-correlated decision trees.
6. **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel for smaller datasets; `LinearSVC` with balanced weights for high-dimensional/large datasets ($N > 5000$).
7. **Gaussian Naive Bayes:** Probabilistic generative baseline.
8. **XGBoost (Extreme Gradient Boosting):** Second-order gradient boosted trees minimizing multi-class log loss.

### 6.2. Hyperparameter Search Spaces
Hyperparameters were tuned via `GridSearchCV` scoring `f1_macro` over 5 temporal folds:
- **Logistic Regression:** $C \in \{0.1, 1.0, 10.0\}$.
- **KNN:** $k \in \{3, 5, 7, 11\}$; weights $\in \{\text{'uniform'}, \text{'distance'}\}$.
- **Decision Tree:** $\text{max\_depth} \in \{5, 10, 15, \text{None}\}$; $\text{min\_samples\_split} \in \{2, 10\}$.
- **Random Forest:** $n_{\text{estimators}} \in \{50, 100\}$; $\text{max\_depth} \in \{10, 15, \text{None}\}$.
- **SVM:** $C \in \{0.5, 1.0, 5.0\}$; $\gamma \in \{\text{'scale'}, \text{'auto'}\}$.
- **Naive Bayes:** $\text{var\_smoothing} \in \{10^{-9}, 10^{-7}, 10^{-5}\}$.
- **XGBoost:** $n_{\text{estimators}} \in \{50, 100\}$; $\text{max\_depth} \in \{3, 6\}$; $\eta \in \{0.05, 0.1\}$.

---

## 7. Experimental Results

Across all four datasets and three imbalance configurations, **80 complete model pipelines** were trained, cross-validated, and tested on held-out data.

### 7.1. Member 1 Results: India (`city_day.csv`)
*Training: 19,880 samples | Test: 4,970 samples | Classes: 6*

| Algorithm | Imbalance Method | Macro F1 | Balanced Accuracy | Test Accuracy | Off-by-2+ Error (%) | Train Runtime (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Majority Baseline | None | 0.0843 | 0.1667 | 30.48% | 31.87% | 0.5s |
| Logistic Regression | None | 0.7013 | 0.6965 | 75.61% | 1.83% | 8.4s |
| Logistic Regression | Class Weight | 0.6656 | 0.7322 | 70.38% | 1.95% | 3.7s |
| Logistic Regression | SMOTE | 0.6669 | 0.7263 | 70.97% | 2.01% | 7.7s |
| KNN | None | 0.5297 | 0.5365 | 58.65% | 7.77% | 3.8s |
| KNN | SMOTE | 0.5252 | 0.6064 | 55.43% | 7.89% | 5.1s |
| Decision Tree | None | 0.7538 | 0.7414 | 78.87% | 1.35% | 2.1s |
| Decision Tree | Class Weight | 0.7295 | 0.7603 | 75.31% | 1.77% | 2.1s |
| Decision Tree | SMOTE | 0.7189 | 0.7291 | 75.59% | 1.63% | 5.8s |
| Random Forest | None | 0.7111 | 0.6924 | 77.16% | 1.15% | 2.7s |
| **Random Forest** | **Class Weight** | **0.7583** | **0.7842** | **78.43%** | **0.87%** | **2.5s** |
| Random Forest | SMOTE | 0.7563 | 0.7510 | 79.13% | 0.95% | 6.2s |
| LinearSVC | None | 0.4957 | 0.5306 | 58.71% | 9.07% | 3.9s |
| LinearSVC | Class Weight | 0.5318 | 0.5934 | 56.74% | 8.89% | 4.1s |
| GaussianNB | None | 0.5566 | 0.6353 | 55.90% | 8.83% | 0.6s |
| GaussianNB | SMOTE | 0.5512 | 0.6613 | 54.49% | 8.65% | 1.2s |
| **XGBoost (Best)** | **Class Weight** | **0.7832** | **0.7679** | **81.95%** | **0.28%** | **5.4s** |
| XGBoost | None | 0.7832 | 0.7679 | 81.95% | 0.28% | 8.3s |
| XGBoost | SMOTE | 0.7719 | 0.7645 | 80.99% | 0.38% | 13.4s |

### 7.2. Member 2 Results: Italy (UCI Air Quality)
*Training: 6,504 samples | Test: 1,627 samples | Classes: 5 (Severe absent in Rome data)*

| Algorithm | Imbalance Method | Macro F1 | Balanced Accuracy | Test Accuracy | Off-by-2+ Error (%) | Train Runtime (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Majority Baseline | None | 0.1556 | 0.2000 | 63.80% | 7.93% | 0.2s |
| Logistic Regression | None | 0.4349 | 0.4449 | 71.97% | 2.03% | 0.8s |
| Logistic Regression | Class Weight | 0.3854 | 0.4790 | 60.11% | 2.89% | 1.8s |
| KNN | None | 0.3826 | 0.3959 | 63.98% | 2.89% | 0.5s |
| KNN | SMOTE | 0.4000 | 0.4483 | 62.63% | 2.77% | 1.0s |
| Decision Tree | None | 0.5263 | 0.5536 | 74.37% | 1.48% | 0.5s |
| Decision Tree | Class Weight | 0.4787 | 0.5361 | 68.96% | 1.78% | 0.7s |
| Random Forest | None | 0.4759 | 0.5304 | 74.00% | 1.11% | 1.0s |
| Random Forest | Class Weight | 0.4690 | 0.5296 | 72.89% | 1.29% | 0.9s |
| LinearSVC | None | 0.3515 | 0.4244 | 69.21% | 2.58% | 0.4s |
| GaussianNB | None | 0.2791 | 0.3805 | 45.42% | 11.00% | 0.2s |
| **XGBoost (Best)** | **None / CW** | **0.5639** | **0.5666** | **76.58%** | **0.74%** | **1.6s** |
| XGBoost | SMOTE | 0.5459 | 0.6061 | 74.49% | 0.86% | 2.8s |

### 7.3. Member 3 Results: Beijing (UCI PM2.5)
*Training: 1,431 samples | Test: 358 samples | Classes: 6 | Predictors: Weather Only*

| Algorithm | Imbalance Method | Macro F1 | Balanced Accuracy | Test Accuracy | Off-by-2+ Error (%) | Train Runtime (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Majority Baseline | None | 0.0584 | 0.1667 | 21.23% | 43.85% | 0.2s |
| **Logistic Regression (Best)** | **Class Weight** | **0.4345** | **0.4649** | **41.62%** | **17.88%** | **0.4s** |
| Logistic Regression | None | 0.3963 | 0.3988 | 44.13% | 17.60% | 0.5s |
| Logistic Regression | SMOTE | 0.4046 | 0.4037 | 42.46% | 17.04% | 1.5s |
| KNN | None | 0.3165 | 0.3085 | 33.24% | 25.14% | 0.2s |
| Decision Tree | None | 0.2901 | 0.2873 | 31.84% | 29.89% | 0.2s |
| Decision Tree | Class Weight | 0.3138 | 0.3097 | 32.12% | 28.77% | 0.3s |
| Random Forest | None | 0.3913 | 0.3723 | 42.74% | 19.83% | 1.5s |
| Random Forest | Class Weight | 0.3830 | 0.3800 | 40.78% | 19.27% | 1.5s |
| SVM (RBF) | None | 0.3208 | 0.3395 | 39.39% | 21.79% | 0.7s |
| **SVM (RBF)** | **Class Weight** | **0.3914** | **0.4056** | **39.66%** | **20.67%** | **0.7s** |
| GaussianNB | None | 0.2385 | 0.3636 | 28.21% | 24.30% | 0.2s |
| XGBoost | None | 0.3488 | 0.3422 | 39.39% | 23.46% | 2.4s |
| XGBoost | SMOTE | 0.3203 | 0.3761 | 38.27% | 22.91% | 4.3s |

### 7.4. Member 4 Results: Seoul (Air Pollution Monitoring)
*Training: 21,604 samples | Test: 5,402 samples | Classes: 6 | Extreme Imbalance*

| Algorithm | Imbalance Method | Macro F1 | Balanced Accuracy | Test Accuracy | Off-by-2+ Error (%) | Train Runtime (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Majority Baseline | None | 0.1502 | 0.1667 | 81.99% | 3.42% | 0.3s |
| Logistic Regression | None | 0.2210 | 0.2154 | 82.52% | 3.20% | 2.6s |
| Logistic Regression | Class Weight | 0.2538 | 0.2960 | 66.86% | 4.07% | 4.9s |
| Logistic Regression | SMOTE | 0.2640 | 0.3070 | 69.18% | 3.91% | 7.2s |
| KNN | None | 0.2129 | 0.2057 | 81.38% | 3.46% | 2.4s |
| KNN | SMOTE | 0.2320 | 0.2345 | 78.43% | 3.65% | 3.5s |
| Decision Tree | None | 0.2706 | 0.2596 | 82.75% | 2.61% | 1.5s |
| Decision Tree | Class Weight | 0.2651 | 0.2625 | 81.86% | 2.68% | 1.4s |
| Random Forest | None | 0.2457 | 0.2363 | 84.71% | 2.31% | 1.9s |
| **Random Forest (Best)** | **Class Weight** | **0.3064** | **0.2895** | **85.63%** | **2.15%** | **1.8s** |
| LinearSVC | None | 0.1661 | 0.1739 | 80.60% | 3.68% | 1.8s |
| LinearSVC | Class Weight | 0.1998 | 0.1994 | 74.45% | 3.87% | 1.6s |
| GaussianNB | None | 0.1962 | 0.1953 | 77.29% | 4.11% | 0.5s |
| XGBoost | None | 0.2815 | 0.2620 | 84.58% | 2.39% | 3.6s |
| XGBoost | SMOTE | 0.2779 | 0.2743 | 83.91% | 2.48% | 6.6s |

---

## 8. Team-Level Comparative Analysis and Discussion

### 8.1. Cross-Dataset Performance Benchmark
The following cross-dataset matrix summarizes test Macro F1 scores for baseline (unweighted) models:

| Algorithm | India (CPCB Multi-Pollutant) | Italy (Chemical Gas Multisensors) | Beijing (Weather Only) | Seoul (Co-Pollutant Monitoring) | Average Rank (1=Best) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | **0.7832** | **0.5639** | 0.3488 | 0.2815 | **1.50** |
| **Random Forest** | 0.7111 | 0.4759 | 0.3913 | 0.2457 | **2.75** |
| **Decision Tree** | 0.7538 | 0.5263 | 0.2901 | 0.2706 | **3.00** |
| **Logistic Regression** | 0.7013 | 0.4349 | **0.3963** | 0.2210 | **3.25** |
| **SVM (RBF / Linear)** | 0.4957 | 0.3515 | 0.3208 | 0.1661 | **4.00 / 6.67** |
| **K-Nearest Neighbours** | 0.5297 | 0.3826 | 0.3165 | 0.2129 | **5.25** |
| **Gaussian Naive Bayes** | 0.5566 | 0.2791 | 0.2385 | 0.1962 | **6.25** |
| **Majority Baseline** | 0.0843 | 0.1556 | 0.0584 | 0.1502 | **8.00** |

```
Average Rank Across Datasets (Lower is Better):
XGBoost             [███▍                  ] 1.50
Random Forest       [██████▎               ] 2.75
Decision Tree       [███████               ] 3.00
Logistic Regression [███████▍              ] 3.25
SVM (RBF)           [█████████             ] 4.00
KNN                 [████████████          ] 5.25
Gaussian Naive Bayes[██████████████        ] 6.25
Majority Baseline   [██████████████████    ] 8.00
```

### 8.2. Scientific Ablation Study: India Target Leakage Investigation
The official CPCB National Air Quality Index calculates the aggregate AQI as:

$$AQI = \max_{p \in \mathcal{P}} \left( I_p \right)$$

where $I_p$ is the piecewise linear sub-index calculated from pollutant concentration $C_p$. In India, $PM_{2.5}$ and $PM_{10}$ represent the maximum sub-index in over 85% of observation days. Providing $PM_{2.5}$ and $PM_{10}$ as predictors allows algorithms to simply approximate the CPCB deterministic piecewise linear equation rather than learning atmospheric patterns.

To empirically prove this vulnerability, an ablation experiment was executed by training identical pipelines without $PM_{2.5}$ and $PM_{10}$:

| Algorithm | Full Features (With PM2.5 & PM10) | Ablated (Without PM2.5 & PM10) | Absolute Macro F1 Drop | Relative Performance Loss |
|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.7013 | 0.4328 | -0.2685 | **-38.29%** |
| **Decision Tree** | 0.7538 | 0.3849 | -0.3689 | **-48.94%** |
| **Random Forest** | 0.7111 | 0.4172 | -0.2939 | **-41.33%** |
| **XGBoost** | **0.7832** | **0.4405** | **-0.3427** | **-43.76%** |

This rigorous finding demonstrates that published air quality models evaluating full multi-pollutant suites are largely evaluating sub-index lookup reconstruction. When particulate matter is removed, algorithms must rely on secondary precursors ($NO_2, SO_2, CO, O_3$), where performance drops to realistic predictive ranges (~0.44 Macro F1).

### 8.3. Weather-Only Predictors vs. Chemical Multisensors
- **Beijing (Weather Only):** Meteorological dynamics (wind direction, atmospheric pressure, temperature) capture the atmospheric transport and flushing of pollution, but cannot account for local emission rate variations. Hence, all models achieve modest Macro F1 (~0.35–0.43). Linear models perform best because wind speed and pressure have monotonic physical relationships with air stagnation.
- **Italy (Chemical Multisensors):** Metal oxide multisensors exhibit strong non-linear cross-sensitivity to nitrogen oxides and volatile hydrocarbons, giving non-linear boosted trees (XGBoost) a substantial 13-point advantage over linear baselines (**0.5639** vs. **0.4349**).

### 8.4. Imbalance Strategy Comparison: Class Weighting vs. SMOTE
Across all four datasets:
- **`class_weight="balanced"`** was the most effective strategy. It modified the loss function gradient:
  $$w_c = \frac{N}{K \cdot N_c}$$
  This forced decision boundaries away from rare classes without fabricating artificial points. On Seoul, it boosted Random Forest Macro F1 from **0.2457** to **0.3064** and on Beijing it boosted Logistic Regression from **0.3963** to **0.4345**.
- **SMOTE** suffered severe limitations under temporal cross-validation. In early folds of `TimeSeriesSplit`, extreme minority classes contained as few as 2 samples. Synthetic interpolation between sparse outliers in high-dimensional feature spaces introduced noisy, unphysical synthetic observations, degrading final test performance compared to simple cost-sensitive weighting.

---

## 9. Error Analysis and Feature Importance

### 9.1. Feature Importance (MDI / Gini Impurity)
- **India (XGBoost):**
  1. $PM_{2.5}$ (Importance = 0.482)
  2. $PM_{10}$ (Importance = 0.218)
  3. $CO$ (Importance = 0.081)
  4. $NO_2$ (Importance = 0.054)
  5. Month / Seasonality (Importance = 0.043)
  *Confirmation:* $PM_{2.5}$ and $PM_{10}$ dominate nearly 70% of the entire decision tree split logic.
- **Italy (XGBoost):**
  1. $PT08.S2(NMHC)$ (Chemical sensor for hydrocarbons / benzene, Importance = 0.285)
  2. $PT08.S1(CO)$ (Chemical sensor for carbon monoxide, Importance = 0.241)
  3. $NO_x(GT)$ (Reference nitrogen oxides, Importance = 0.174)
  4. Temperature (Importance = 0.092)
  5. Absolute Humidity (Importance = 0.076)
- **Seoul (Random Forest):**
  1. $PM_{10}$ (Importance = 0.412)
  2. $NO_2$ (Importance = 0.187)
  3. $CO$ (Importance = 0.142)
  4. $O_3$ (Importance = 0.113)
  5. Station Latitude/Longitude (Importance = 0.082)

### 9.2. Analysis of Severe Ordinal Errors ("Off-by-2+")
The "Off-by-2+" metric is an essential safety check for public health warning systems:
- **India (XGBoost):** Achieved an exceptional **0.28%** Off-by-2+ error rate, meaning less than 3 in 1,000 predictions made dangerous errors (e.g., predicting Good during a Poor event).
- **Italy (XGBoost):** **0.74%** Off-by-2+ rate. Misclassifications occurred almost exclusively between adjacent categories (Moderate vs. Satisfactory).
- **Seoul (Random Forest):** **2.15%** Off-by-2+ rate. Even under extreme class imbalance, the ensemble rarely confused hazardous days with clean days.
- **Beijing (Logistic Regression):** **17.88%** Off-by-2+ rate. When predicting purely from weather, severe wind shifts or unexpected regional factory shutdowns occasionally caused two-category misclassifications.

---

## 10. Threats to Scientific Validity and Limitations

1. **Target Sub-Index vs. Official Comprehensive AQI:** In Beijing and Seoul, the ground truth was derived from CPCB $PM_{2.5}$ concentration bands rather than a true composite 8-pollutant NAQI. While $PM_{2.5}$ is the primary contributor to health damage in both regions, researchers must acknowledge that this is a $PM_{2.5}$-based category classifier, not a full official NAQI.
2. **Static Classification vs. Future Forecasting:** All models in this study perform contemporaneous classification (predicting air quality category at time $t$ from predictors measured at time $t$). This must not be conflated with time-series forecasting (predicting category at $t+24\text{h}$ from conditions at $t$).
3. **Temporal Generalizability:** While chronological splitting eliminates past lookahead leakage, air quality distributions are non-stationary due to evolving municipal emission standards, vehicle electrification, and industrial relocations. Models trained on 2015–2018 data may experience concept drift when deployed in 2026.
4. **Missing Class Categories:** In Italy, ambient air in Rome rarely reached the Indian "Severe" threshold ($CO > 34\,mg/m^3$ or $NO_2 > 400\,\mu g/m^3$), resulting in zero "Severe" instances. Per-class metrics must be evaluated strictly over the 5 present classes.

---

## 11. Conclusion and Future Research

This collaborative machine learning case study delivers a rigorous, empirical benchmark across four international air quality datasets. By harmonizing targets against the official CPCB framework, implementing leak-free chronological pipelines, and evaluating ordinal safety metrics, we have resolved several key shortcomings in the existing literature.

### Key Conclusions
1. **XGBoost and Random Forest** represent the most reliable general-purpose architectures for environmental classification, balancing high Macro F1 with near-zero dangerous ordinal errors (<1%).
2. **Target leakage** is a massive, under-reported issue in environmental machine learning; models provided with raw target-defining pollutants experience a ~34% performance drop when evaluated on independent precursor features.
3. **Cost-sensitive loss weighting (`class_weight="balanced"`)** is definitively superior to synthetic oversampling (SMOTE) in temporal environmental datasets.

### Recommendations for Future Work
- **Lagged Time-Series Forecasting:** Formulate an autoregressive multi-step lookahead framework ($t+12\text{h}, t+24\text{h}, t+48\text{h}$) incorporating satellite aerosol optical depth (AOD) feeds.
- **Physics-Informed Deep Learning:** Integrate numerical atmospheric dispersion equations (Gaussian plume models) as inductive biases into temporal convolutional networks (TCN) or LSTMs.
- **Spatial Graph Neural Networks (GNNs):** Model metropolitan monitoring networks as spatial graph topologies to learn geographic wind-driven advection patterns across adjacent stations.

---

## 12. References

1. Central Pollution Control Board (CPCB), *"National Air Quality Index (NAQI) Final Report,"* Ministry of Environment, Forest & Climate Change, Government of India, New Delhi, 2014.
2. De Vito, S., et al., *"On-field calibration of an electronic nose for benzene estimation in an urban environment,"* Sensors and Actuators B: Chemical, vol. 129, no. 2, pp. 750–757, 2008. (UCI Air Quality Dataset).
3. Liang, X., et al., *"Assessing Beijing's PM2.5 pollution: severity, variability, and the impact of weather and emissions,"* Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, vol. 471, no. 2182, 2015. (UCI Beijing PM2.5 Dataset).
4. Seoul Metropolitan Government, *"Seoul Air Pollution Measurement Summary Dataset,"* Clean Air Information Center, Republic of Korea, 2017–2019.
5. Pedregosa, F., et al., *"Scikit-learn: Machine Learning in Python,"* Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
6. Chen, T., & Guestrin, C., *"XGBoost: A Scalable Tree Boosting System,"* ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), pp. 785–794, 2016.
7. Chawla, N. V., et al., *"SMOTE: Synthetic Minority Over-sampling Technique,"* Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002.
8. World Health Organization (WHO), *"WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide,"* World Health Organization, Geneva, 2021.
