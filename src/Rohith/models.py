"""
Model configurations for Beijing Dataset (Rohith):
1. Random Forest
2. Gradient Boosting
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier


def get_models(random_state: int = 42):
    """Returns the dictionary of assigned classifiers for this benchmark."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        )
    }
