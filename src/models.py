"""
models.py
Implementation and hyperparameter tuning configurations for the 7 ML classifiers:
1. Logistic Regression
2. K-Nearest Neighbours
3. Decision Tree
4. Random Forest
5. Support Vector Machine (RBF kernel)
6. Naive Bayes (GaussianNB)
7. XGBoost (XGBClassifier)
Plus Majority-Class Baseline.
Supports Imbalance handling: None, class_weight='balanced', and SMOTE.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def get_base_model(
    algorithm: str,
    imbalance_method: str = "none",
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Instantiate base estimator and default parameter grid for TimeSeriesSplit GridSearchCV.
    Handles class_weight where supported.
    """
    has_cw = (imbalance_method == "class_weight")

    if algorithm == "majority_baseline":
        model = DummyClassifier(strategy="most_frequent")
        param_grid = {}

    elif algorithm == "logistic_regression":
        cw = "balanced" if has_cw else None
        model = LogisticRegression(
            max_iter=1000,
            class_weight=cw,
            random_state=random_state,
            solver="lbfgs"
        )
        param_grid = {
            "classifier__C": [0.1, 1.0, 10.0]
        }

    elif algorithm == "knn":
        # KNN does not support class_weight natively
        model = KNeighborsClassifier()
        param_grid = {
            "classifier__n_neighbors": [3, 5, 7, 11],
            "classifier__weights": ["uniform", "distance"]
        }

    elif algorithm == "decision_tree":
        cw = "balanced" if has_cw else None
        model = DecisionTreeClassifier(
            class_weight=cw,
            random_state=random_state
        )
        param_grid = {
            "classifier__max_depth": [5, 10, 15, None],
            "classifier__min_samples_split": [2, 10]
        }

    elif algorithm == "random_forest":
        cw = "balanced" if has_cw else None
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight=cw,
            random_state=random_state,
            n_jobs=-1
        )
        param_grid = {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [10, 15, None]
        }

    elif algorithm == "svm":
        cw = "balanced" if has_cw else None
        model = SVC(
            kernel="rbf",
            class_weight=cw,
            random_state=random_state,
            max_iter=3000
        )
        param_grid = {
            "classifier__C": [0.5, 1.0, 5.0],
            "classifier__gamma": ["scale", "auto"]
        }

    elif algorithm == "linear_svc":
        cw = "balanced" if has_cw else None
        model = LinearSVC(
            dual=False,
            class_weight=cw,
            random_state=random_state,
            max_iter=2000
        )
        param_grid = {
            "classifier__C": [0.1, 1.0, 10.0]
        }

    elif algorithm == "naive_bayes":
        # GaussianNB does not support class_weight natively
        model = GaussianNB()
        param_grid = {
            "classifier__var_smoothing": [1e-9, 1e-7, 1e-5]
        }

    elif algorithm == "xgboost":
        # XGBoost multi-class classifier
        model = XGBClassifier(
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
        param_grid = {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [3, 6],
            "classifier__learning_rate": [0.05, 0.1]
        }

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model, param_grid


def requires_scaling(algorithm: str) -> bool:
    """Return whether the algorithm benefits from standard feature scaling."""
    return algorithm in ["logistic_regression", "knn", "svm", "linear_svc"]


def build_full_pipeline(
    preprocessor,
    algorithm: str,
    imbalance_method: str = "none",
    random_state: int = 42,
    min_class_samples: int = 6
) -> Tuple[ImbPipeline, Dict[str, Any]]:
    """
    Build integrated end-to-end pipeline:
    [Preprocessor] -> (optional SMOTE) -> [Classifier]
    SMOTE is only evaluated inside training folds via imblearn Pipeline.
    """
    model, param_grid = get_base_model(algorithm, imbalance_method, random_state)

    steps = [("preprocessor", preprocessor)]

    if imbalance_method == "smote":
        k_neighbors = max(1, min(5, min_class_samples - 1))
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        steps.append(("smote", smote))

    steps.append(("classifier", model))

    pipeline = ImbPipeline(steps=steps)
    return pipeline, param_grid
