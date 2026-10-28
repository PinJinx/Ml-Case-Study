"""
Sreenidhi's Module - Member 2 (Italy Dataset: AirQualityUCI.csv)
Clubbed Models:
1. Support Vector Machine (LinearSVC / SVC)
2. Gaussian Naive Bayes (GaussianNB)
"""

from typing import Dict, Any, Tuple
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

MEMBER_NAME = "Sreenidhi"
DATASET_KEY = "italy"
ALGORITHMS = ["svm", "linear_svc", "naive_bayes"]


def get_base_model(
    algorithm: str,
    imbalance_method: str = "none",
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    has_cw = (imbalance_method == "class_weight")

    if algorithm in ["svm", "linear_svc", "svm (LinearSVC)"]:
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

    elif algorithm == "svm_rbf":
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

    elif algorithm == "naive_bayes":
        model = GaussianNB()
        param_grid = {
            "classifier__var_smoothing": [1e-9, 1e-7, 1e-5]
        }

    elif algorithm == "majority_baseline":
        model = DummyClassifier(strategy="most_frequent")
        param_grid = {}

    else:
        raise ValueError(f"Algorithm '{algorithm}' is not assigned to {MEMBER_NAME}. Assigned: {ALGORITHMS}")

    return model, param_grid


def requires_scaling(algorithm: str) -> bool:
    return algorithm in ["svm", "linear_svc", "svm (LinearSVC)", "svm_rbf"]


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
