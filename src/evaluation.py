import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from src.labeling import compute_off_by_two_or_more, AQI_CATEGORIES, ID_TO_CATEGORY


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    present_classes: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics on test predictions.
    """
    if present_classes is None:
        present_classes = sorted(list(set(y_true) | set(y_pred)))

    macro_f1 = float(f1_score(y_true, y_pred, labels=present_classes, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    off_by_2_plus = float(compute_off_by_two_or_more(y_true, y_pred))

    # Per-class metrics
    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=present_classes,
        zero_division=0
    )

    per_class = {}
    for idx, c_id in enumerate(present_classes):
        cat_name = ID_TO_CATEGORY.get(c_id, f"Class_{c_id}")
        per_class[cat_name] = {
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f[idx]),
            "support": int(s[idx])
        }

    # Normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_classes, normalize="true")

    return {
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "accuracy": acc,
        "off_by_2_plus": off_by_2_plus,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "present_classes": present_classes,
        "class_names": [ID_TO_CATEGORY.get(c, f"Class_{c}") for c in present_classes]
    }


def train_and_evaluate_model(
    pipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = 5
) -> Dict[str, Any]:
    """
    Tune hyperparameters using TimeSeriesSplit on X_train/y_train,
    then evaluate best estimator on held-out X_test/y_test.
    Measures training runtime, CV mean/std, and final test metrics.
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    start_time = time.time()

    if param_grid:
        try:
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=tscv,
                scoring="f1_macro",
                n_jobs=-1,
                refit=True
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_mean = float(grid_search.best_score_)
            cv_std = float(grid_search.cv_results_["std_test_score"][grid_search.best_index_])
        except Exception as e:
            # Handle temporal sparsity where early folds lack minimum samples for SMOTE
            # Fit pipeline on the 80% training set directly
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {"note": "Fitted on full chronological train set due to fold sparsity"}
            cv_mean = float("nan")
            cv_std = float("nan")
    else:
        # No tuning required (e.g., baseline dummy classifier)
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = {}
        # Compute CV score manually
        scores = []
        try:
            for train_idx, val_idx in tscv.split(X_train):
                pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                val_preds = pipeline.predict(X_train.iloc[val_idx])
                scores.append(f1_score(y_train.iloc[val_idx], val_preds, average="macro", zero_division=0))
            cv_mean = float(np.mean(scores))
            cv_std = float(np.std(scores))
        except Exception:
            cv_mean = float("nan")
            cv_std = float("nan")
        # Re-fit on full training set
        best_model.fit(X_train, y_train)

    train_seconds = float(time.time() - start_time)

    # Held-out test evaluation
    y_pred = best_model.predict(X_test)

    # Present classes across train and test
    present_classes = sorted(list(set(y_train.unique()) | set(y_test.unique())))
    test_metrics = evaluate_predictions(y_test.values, y_pred, present_classes=present_classes)

    return {
        "best_params": best_params,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "train_seconds": train_seconds,
        "test_metrics": test_metrics,
        "best_model": best_model,
        "y_pred": y_pred.tolist()
    }
