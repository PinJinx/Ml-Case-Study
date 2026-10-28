"""
Model configurations for Italy Dataset (Sreenidhi):
1. Support Vector Machine (LinearSVC)
2. Gaussian Naive Bayes
"""

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier


def get_models(random_state: int = 42):
    """Returns the dictionary of assigned classifiers for this benchmark."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "svm": LinearSVC(
            dual=False,
            random_state=random_state,
            max_iter=2000
        ),
        "naive_bayes": GaussianNB(
            var_smoothing=1e-5
        )
    }
