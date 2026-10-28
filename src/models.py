"""
Unified collection of all 8 machine learning classifiers paired across the 4 benchmark datasets.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier

# Mapping of dataset to member and assigned algorithms
DATASET_ALGORITHMS = {
    "india": ["logistic_regression", "knn"],
    "italy": ["svm", "naive_bayes"],
    "beijing": ["random_forest", "gradient_boosting"],
    "seoul": ["decision_tree", "xgboost"]
}


def get_all_models(random_state: int = 42):
    """Returns all 8 models and the majority baseline in a single dictionary."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state, solver="lbfgs"),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "svm": LinearSVC(dual=False, random_state=random_state, max_iter=2000),
        "naive_bayes": GaussianNB(var_smoothing=1e-5),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=random_state),
        "xgboost": XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, eval_metric="mlogloss", random_state=random_state, n_jobs=-1, verbosity=0)
    }


def requires_scaling(model_name: str) -> bool:
    """Returns True if the algorithm requires feature standardization."""
    return model_name in ["logistic_regression", "knn", "svm"]
