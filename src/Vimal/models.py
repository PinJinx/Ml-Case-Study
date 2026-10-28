"""
Vimal's Module - Member 1 (India Dataset: city_day.csv)
Clubbed Models:
1. Logistic Regression (L2 regularized)
2. K-Nearest Neighbours (KNN)
"""

from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

MEMBER_NAME = "Vimal"
DATASET_KEY = "india"
ALGORITHMS = ["logistic_regression", "knn"]


def get_base_model(
    algorithm: str,
    imbalance_method: str = "none",
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    has_cw = (imbalance_method == "class_weight")

    if algorithm == "logistic_regression":
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
        model = KNeighborsClassifier()
        param_grid = {
            "classifier__n_neighbors": [3, 5, 7, 11],
            "classifier__weights": ["uniform", "distance"]
        }

    elif algorithm == "majority_baseline":
        model = DummyClassifier(strategy="most_frequent")
        param_grid = {}

    else:
        raise ValueError(f"Algorithm '{algorithm}' is not assigned to {MEMBER_NAME}. Assigned: {ALGORITHMS}")

    return model, param_grid


def requires_scaling(algorithm: str) -> bool:
    return algorithm in ["logistic_regression", "knn"]


def build_full_pipeline(
    preprocessor,
    algorithm: str,
    imbalance_method: str = "none",
    random_state: int = 42,
    min_class_samples: int = 6
) -> Tuple[ImbPipeline, Dict[str, Any]]:
    model, param_grid = get_base_model(algorithm, imbalance_method, random_state)

    steps = [("preprocessor", preprocessor)]

    if imbalance_method == "smote":
        k_neighbors = max(1, min(5, min_class_samples - 1))
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        steps.append(("smote", smote))

    steps.append(("classifier", model))

    pipeline = ImbPipeline(steps=steps)
    return pipeline, param_grid
