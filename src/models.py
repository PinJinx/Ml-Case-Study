"""
models.py
Central Unified Interface delegating to individual team members' modules:

- src.Vimal (Member 1 - India): Logistic Regression, KNN
- src.Sreenidhi (Member 2 - Italy): SVM, Naive Bayes
- src.Rohith (Member 3 - Beijing): Random Forest, Gradient Boosting
- src.Sruthi (Member 4 - Seoul): Decision Tree, XGBoost
"""

from typing import Dict, Any, Tuple
from imblearn.pipeline import Pipeline as ImbPipeline

import src.Vimal.models as vimal_mod
import src.Sreenidhi.models as sreenidhi_mod
import src.Rohith.models as rohith_mod
import src.Sruthi.models as sruthi_mod

# Mapping of Member -> Assigned Models
MEMBER_MODELS = {
    "Vimal": ["logistic_regression", "knn"],
    "Sreenidhi": ["svm", "naive_bayes"],
    "Rohith": ["random_forest", "gradient_boosting"],
    "Sruthi": ["decision_tree", "xgboost"]
}

# Mapping of Dataset -> Member Module
DATASET_MEMBER_MAP = {
    "india": ("Vimal", vimal_mod),
    "italy": ("Sreenidhi", sreenidhi_mod),
    "beijing": ("Rohith", rohith_mod),
    "seoul": ("Sruthi", sruthi_mod)
}

# Mapping of Algorithm -> Member Module
ALGORITHM_TO_MODULE = {
    "logistic_regression": vimal_mod,
    "knn": vimal_mod,
    "svm": sreenidhi_mod,
    "svm (LinearSVC)": sreenidhi_mod,
    "svm_rbf": sreenidhi_mod,
    "naive_bayes": sreenidhi_mod,
    "random_forest": rohith_mod,
    "gradient_boosting": rohith_mod,
    "decision_tree": sruthi_mod,
    "xgboost": sruthi_mod
}


def get_base_model(
    algorithm: str,
    imbalance_method: str = "none",
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    if algorithm == "majority_baseline":
        return vimal_mod.get_base_model("majority_baseline", imbalance_method, random_state)

    if algorithm in ALGORITHM_TO_MODULE:
        return ALGORITHM_TO_MODULE[algorithm].get_base_model(algorithm, imbalance_method, random_state)

    raise ValueError(f"Unknown algorithm: '{algorithm}'. Supported: {list(ALGORITHM_TO_MODULE.keys())}")


def requires_scaling(algorithm: str) -> bool:
    if algorithm in ALGORITHM_TO_MODULE:
        return ALGORITHM_TO_MODULE[algorithm].requires_scaling(algorithm)
    return algorithm in ["logistic_regression", "knn", "svm", "linear_svc"]


def build_full_pipeline(
    preprocessor,
    algorithm: str,
    imbalance_method: str = "none",
    random_state: int = 42,
    min_class_samples: int = 6
) -> Tuple[ImbPipeline, Dict[str, Any]]:
    if algorithm == "majority_baseline":
        return vimal_mod.build_full_pipeline(preprocessor, algorithm, imbalance_method, random_state, min_class_samples)

    if algorithm in ALGORITHM_TO_MODULE:
        return ALGORITHM_TO_MODULE[algorithm].build_full_pipeline(
            preprocessor, algorithm, imbalance_method, random_state, min_class_samples
        )

    raise ValueError(f"Unknown algorithm: '{algorithm}'")
