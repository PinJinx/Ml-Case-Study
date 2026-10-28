"""
Evaluation metrics and helpers for model performance analysis.
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)
from src.labeling import compute_off_by_two_or_more, ID_TO_CATEGORY


def evaluate_predictions(y_true, y_pred, present_classes=None):
    """Calculates standard classification metrics."""
    if present_classes is None:
        present_classes = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=present_classes, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    off_by_2 = float(compute_off_by_two_or_more(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=present_classes, normalize="true")

    class_names = [ID_TO_CATEGORY.get(c, f"Class_{c}") for c in present_classes]

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "off_by_2_plus": off_by_2,
        "confusion_matrix": cm,
        "class_names": class_names
    }


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Fits model on training data and computes test metrics with timing."""
    start_time = time.time()
    model.fit(X_train, y_train)
    train_seconds = float(time.time() - start_time)

    y_pred = model.predict(X_test)
    present_classes = sorted(list(set(y_train.unique()) | set(y_test.unique())))
    metrics = evaluate_predictions(y_test.values, y_pred, present_classes)
    metrics["train_seconds"] = train_seconds
    metrics["model"] = model

    return metrics
