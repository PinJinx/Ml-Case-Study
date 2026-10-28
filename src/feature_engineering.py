"""
feature_engineering.py
Domain-specific temporal and environmental feature engineering functions.
"""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    if date_col not in df.columns:
        if "DateTime" in df.columns:
            date_col = "DateTime"
        else:
            return df

    dates = pd.to_datetime(df[date_col])
    df["month"] = dates.dt.month
    df["day_of_week"] = dates.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Cyclical encoding for periodicity
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # Meteorological season
    # 12, 1, 2: Winter; 3, 4, 5: Spring; 6, 7, 8: Summer; 9, 10, 11: Autumn
    def get_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Spring"
        elif m in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df["season"] = df["month"].apply(get_season)

    return df
