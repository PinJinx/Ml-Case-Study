# Air Quality Classification Case Study

A comparative machine learning case study evaluating **8 classifiers** paired across **4 international environmental datasets** using the Central Pollution Control Board (CPCB) breakpoint framework.

---

## 1. Project Organization & Model Pairings

The project is structured into **4 independent member folders** under `src/`. Each member analyzes one benchmark dataset using **two clubbed models**:

1. **Vimal (`src/Vimal` - India Dataset):**
   - **Dataset:** `india_city_day.csv` (CPCB urban air quality measurements across Indian cities)
   - **Clubbed Models:** Logistic Regression & K-Nearest Neighbours (KNN)
   - **Ablation Study:** Evaluates performance impact when omitting target-defining pollutants (PM2.5 & PM10).

2. **Sreenidhi (`src/Sreenidhi` - Italy Dataset):**
   - **Dataset:** `italy_air_quality_uci.csv` (Chemical multisensor arrays from Rome)
   - **Clubbed Models:** Support Vector Machine (LinearSVC) & Gaussian Naive Bayes

3. **Rohith (`src/Rohith` - Beijing Dataset):**
   - **Dataset:** `beijing_pm25.csv` (Hourly meteorological and weather observations)
   - **Clubbed Models:** Random Forest & Gradient Boosting (`GradientBoostingClassifier`)

4. **Sruthi (`src/Sruthi` - Seoul Dataset):**
   - **Dataset:** `seoul_air_pollution.csv` (Continuous 25-district monitoring network with extreme class imbalance)
   - **Clubbed Models:** Decision Tree & Extreme Gradient Boosting (`XGBClassifier`)

---

## 2. Directory Structure

```text
aqi-ml-case-study/
│── README.md                # Project documentation and guide
│── requirements.txt         # Python package dependencies
│── run_all_experiments.py   # Master runner executing all 4 benchmarks and team aggregation
│── reports/
│   └── final_report.md      # Detailed academic report
│── results/                 # Consolidated team results, plots, and metrics
└── src/
    ├── labeling.py          # CPCB breakpoints and error metrics
    ├── data_loader.py       # Modular CSV loaders
    ├── feature_engineering.py # Temporal feature extraction
    ├── preprocessing.py     # Chronological train/test split & scikit-learn transformers
    ├── evaluation.py        # Model evaluation and metric calculations
    ├── visualization.py     # Confusion matrix and comparison plotting
    ├── comparison.py        # Team aggregation and cross-dataset heatmaps
    ├── models.py            # Central dictionary of all 8 models
    │
    ├── Vimal/               # Member 1 (India)
    │   ├── data/            # Local dataset
    │   ├── models.py        # Logistic Regression & KNN
    │   ├── run_experiments.py # Independent runner
    │   └── results/         # Local metrics and plots
    │
    ├── Sreenidhi/           # Member 2 (Italy)
    │   ├── data/            # Local dataset
    │   ├── models.py        # LinearSVC & Naive Bayes
    │   ├── run_experiments.py # Independent runner
    │   └── results/         # Local metrics and plots
    │
    ├── Rohith/              # Member 3 (Beijing)
    │   ├── data/            # Local dataset
    │   ├── models.py        # Random Forest & Gradient Boosting
    │   ├── run_experiments.py # Independent runner
    │   └── results/         # Local metrics and plots
    │
    └── Sruthi/              # Member 4 (Seoul)
        ├── data/            # Local dataset
        ├── models.py        # Decision Tree & XGBoost
        ├── run_experiments.py # Independent runner
        └── results/         # Local metrics and plots
```

---

## 3. How to Run

### Run Any Member Benchmark Independently
Each person's experiments can be run directly from their folder:

```bash
# Vimal (India: Logistic Regression & KNN)
cd src/Vimal
python run_experiments.py

# Sreenidhi (Italy: SVM & Naive Bayes)
cd src/Sreenidhi
python run_experiments.py

# Rohith (Beijing: Random Forest & Gradient Boosting)
cd src/Rohith
python run_experiments.py

# Sruthi (Seoul: Decision Tree & XGBoost)
cd src/Sruthi
python run_experiments.py
```
*Outputs are saved directly inside that folder's `results/` directory.*

### Run All Benchmarks & Team Synthesis
To run all 4 member benchmarks sequentially and generate consolidated cross-dataset heatmaps and summary plots:

```bash
python run_all_experiments.py
```

To quickly re-aggregate existing results without re-training:
```bash
python run_all_experiments.py --aggregate
```
