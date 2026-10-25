"""
preprocessing.py
Chronological splitting, imputation, encoding, and transformer pipelines.
Ensures zero data leakage: all transformations fitted strictly on training folds.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.labeling import CATEGORY_TO_ID, AQI_CATEGORIES


def chronological_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.8,
    date_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split time-series data chronologically:
    - First 80% for training.
    - Last 20% for held-out testing.
    - No random shuffling.
    """
    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * train_ratio)

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    test_df = df.iloc[n_train:].copy().reset_index(drop=True)

    y_train_raw = train_df[target_col]
    y_test_raw = test_df[target_col]

    # Map target strings to integer IDs
    y_train = y_train_raw.map(CATEGORY_TO_ID).astype(int)
    y_test = y_test_raw.map(CATEGORY_TO_ID).astype(int)

    # Features: drop target column and any date columns from predictor matrix
    drop_cols = [target_col]
    for c in ["Date", "DateTime", "Measurement date"]:
        if c in df.columns:
            drop_cols.append(c)

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    return X_train, X_test, y_train, y_test


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale: bool = True,
    impute_strategy: str = "median"
) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for preprocessing:
    - Imputation: Median (or mean/KNN) for numeric columns, most_frequent for categorical.
    - Scaling: StandardScaler applied if scale=True (for LR, KNN, SVM).
    - Encoding: OneHotEncoder (handle_unknown='ignore') for categorical columns.
    """
    # Numeric pipeline
    num_steps = []
    if impute_strategy == "knn":
        num_steps.append(("imputer", KNNImputer(n_neighbors=5)))
    else:
        num_steps.append(("imputer", SimpleImputer(strategy="median")))

    if scale:
        num_steps.append(("scaler", StandardScaler()))

    num_pipeline = Pipeline(steps=num_steps)

    # Categorical pipeline
    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
    cat_pipeline = Pipeline(steps=cat_steps)

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor
