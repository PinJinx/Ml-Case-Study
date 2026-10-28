"""
Model configurations for Seoul Dataset (Sruthi):
1. Decision Tree
2. XGBoost
"""

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier


def get_models(random_state: int = 42):
    """Returns the dictionary of assigned classifiers for this benchmark."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            random_state=random_state
        ),
        "xgboost": XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
    }
