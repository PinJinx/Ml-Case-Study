from typing import Union, List
import numpy as np
import pandas as pd

# The 6 standard CPCB AQI categories
AQI_CATEGORIES = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe"
]

CATEGORY_TO_ID = {cat: idx for idx, cat in enumerate(AQI_CATEGORIES)}
ID_TO_CATEGORY = {idx: cat for idx, cat in enumerate(AQI_CATEGORIES)}

# CPCB 24-hr Concentration Breakpoints (µg/m³, except CO in mg/m³)
# Format: (low_inclusive, high_inclusive, category)
PM25_BREAKPOINTS = [
    (0.0, 30.0, "Good"),
    (30.0, 60.0, "Satisfactory"),
    (60.0, 90.0, "Moderate"),
    (90.0, 120.0, "Poor"),
    (120.0, 250.0, "Very Poor"),
    (250.0, float("inf"), "Severe")
]

CO_BREAKPOINTS = [
    (0.0, 1.0, "Good"),
    (1.0, 2.0, "Satisfactory"),
    (2.0, 10.0, "Moderate"),
    (10.0, 17.0, "Poor"),
    (17.0, 34.0, "Very Poor"),
    (34.0, float("inf"), "Severe")
]

NO2_BREAKPOINTS = [
    (0.0, 40.0, "Good"),
    (40.0, 80.0, "Satisfactory"),
    (80.0, 180.0, "Moderate"),
    (180.0, 280.0, "Poor"),
    (280.0, 400.0, "Very Poor"),
    (400.0, float("inf"), "Severe")
]


def classify_scalar(value: float, breakpoints: List[tuple]) -> Union[str, float]:
    if pd.isna(value) or value < 0:
        return np.nan
    for low, high, category in breakpoints:
        if low == 0.0:
            if low <= value <= high:
                return category
        else:
            if low < value <= high:
                return category
    return "Severe"


def classify_pm25(value: float) -> Union[str, float]:
    return classify_scalar(value, PM25_BREAKPOINTS)


def classify_co(value: float) -> Union[str, float]:
    return classify_scalar(value, CO_BREAKPOINTS)


def classify_no2(value: float) -> Union[str, float]:
    return classify_scalar(value, NO2_BREAKPOINTS)


def classify_worse_of_co_and_no2(co_val: float, no2_val: float) -> Union[str, float]:
    cat_co = classify_co(co_val)
    cat_no2 = classify_no2(no2_val)

    if pd.isna(cat_co) and pd.isna(cat_no2):
        return np.nan
    if pd.isna(cat_co):
        return cat_no2
    if pd.isna(cat_no2):
        return cat_co

    idx_co = CATEGORY_TO_ID[cat_co]
    idx_no2 = CATEGORY_TO_ID[cat_no2]
    worse_idx = max(idx_co, idx_no2)
    return ID_TO_CATEGORY[worse_idx]


def compute_off_by_two_or_more(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diffs = np.abs(np.array(y_true) - np.array(y_pred))
    return float(np.mean(diffs >= 2) * 100.0)
