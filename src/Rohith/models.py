"""
Rohith's Module - Member 3 (Beijing Dataset: PRSA_data_2010.1.1-2014.12.31.csv)
Clubbed Models:
1. Random Forest (RandomForestClassifier)
2. Gradient Boosting (GradientBoostingClassifier)
"""

from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

MEMBER_NAME = "Rohith"
DATASET_KEY = "beijing"
ALGORITHMS = ["random_forest", "gradient_boosting"]


def get_base_model(
    algorithm: str,
    imbalance_method: str = "none",
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    has_cw = (imbalance_method == "class_weight")

    if algorithm == "random_forest":
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

    elif algorithm == "gradient_boosting":
        model = GradientBoostingClassifier(
            random_state=random_state
        )
        param_grid = {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1]
        }

    elif algorithm == "majority_baseline":
        model = DummyClassifier(strategy="most_frequent")
        param_grid = {}

    else:
        raise ValueError(f"Algorithm '{algorithm}' is not assigned to {MEMBER_NAME}. Assigned: {ALGORITHMS}")

    return model, param_grid


def requires_scaling(algorithm: str) -> bool:
    return False


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
