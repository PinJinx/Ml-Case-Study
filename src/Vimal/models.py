"""
Model configurations for India Dataset (Vimal):
1. Logistic Regression
2. K-Nearest Neighbours (KNN)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


def get_models(random_state: int = 42):
    """Returns the dictionary of assigned classifiers for this benchmark."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs"
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        )
    }
