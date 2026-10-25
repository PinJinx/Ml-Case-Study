"""
data_loader.py
Modular loaders for all four Air Quality datasets:
1. India (city_day.csv)
2. Italy (AirQualityUCI.csv)
3. Beijing (PRSA_data_2010.1.1-2014.12.31.csv)
4. Seoul (Measurement_summary.csv)
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from src.labeling import (
    classify_pm25,
    classify_worse_of_co_and_no2,
    AQI_CATEGORIES,
    CATEGORY_TO_ID
)


def load_india_dataset(filepath: str) -> pd.DataFrame:
    """
    Load India city_day.csv dataset.
    - Uses existing AQI_Bucket as target.
    - Excludes rows without a valid target.
    - Parses Date and sorts chronologically.
    - Removes AQI column (target leakage).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"India dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Exclude rows without valid AQI_Bucket
    df = df.dropna(subset=["AQI_Bucket"]).copy()
    # Filter to only valid CPCB categories
    df = df[df["AQI_Bucket"].isin(AQI_CATEGORIES)].copy()

    # Drop direct target leakage
    if "AQI" in df.columns:
        df = df.drop(columns=["AQI"])

    return df


def load_italy_dataset(filepath: str) -> pd.DataFrame:
    """
    Load Italy UCI AirQuality dataset.
    - Semicolon delimited, comma as decimal separator.
    - Sentinel values (-200) replaced by NaN.
    - Drop trailing empty columns.
    - Hourly data kept with rolling 24h window (since total days ~391 < 500 rows threshold).
    - Target constructed as worse of CO(GT) and NO2(GT) CPCB bands.
    - CO(GT) and NO2(GT) dropped from predictors to prevent target leakage.
    - NMHC(GT) analyzed (has >90% missing values) and dropped.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Italy dataset not found at {filepath}")

    df = pd.read_csv(filepath, sep=";", decimal=",")

    # Drop unnamed/empty trailing columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()
    # Remove rows where Date is NaN
    df = df.dropna(subset=["Date"]).copy()

    # Convert sentinel values -200 to NaN
    for col in df.columns:
        if col not in ["Date", "Time"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace(-200.0, np.nan)

    # Combine Date and Time into DateTime
    # Format in dataset is DD/MM/YYYY and HH.MM.SS
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce"
    )
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)

    # Construct target using worse of CO(GT) and NO2(GT)
    # CO(GT) is in mg/m3, NO2(GT) is in µg/m3
    targets = []
    for _, row in df.iterrows():
        cat = classify_worse_of_co_and_no2(row.get("CO(GT)"), row.get("NO2(GT)"))
        targets.append(cat)
    df["AQI_Category"] = targets

    # Filter rows with valid target
    df = df.dropna(subset=["AQI_Category"]).copy()
    df = df[df["AQI_Category"].isin(AQI_CATEGORIES)].copy()

    # Drop leakage columns: CO(GT), NO2(GT) and highly missing NMHC(GT) (>90% missing)
    cols_to_drop = ["Date", "Time", "CO(GT)", "NO2(GT)", "NMHC(GT)"]
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drop)

    return df


def load_beijing_dataset(filepath: str, aggregate_daily: bool = True) -> pd.DataFrame:
    """
    Load Beijing UCI PM2.5 dataset.
    - Exclude 'No' index column and 'pm2.5' from predictors.
    - Target constructed from PM2.5 concentration bands.
    - Encode wind direction (cbwd).
    - Aggregate to daily means if specified.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Beijing dataset not found at {filepath}")

    df = pd.read_csv(filepath)

    # Combine year, month, day, hour into datetime
    df["DateTime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("DateTime").reset_index(drop=True)

    # Construct target category from pm2.5
    df["AQI_Category"] = df["pm2.5"].apply(classify_pm25)
    df = df.dropna(subset=["AQI_Category"]).copy()
    df = df[df["AQI_Category"].isin(AQI_CATEGORIES)].copy()

    if aggregate_daily:
        # Group by Date
        df["Date"] = df["DateTime"].dt.date
        
        # Numeric columns mean
        numeric_cols = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
        daily_num = df.groupby("Date")[numeric_cols].mean()
        
        # Most frequent wind direction per day
        daily_wind = df.groupby("Date")["cbwd"].agg(lambda x: x.mode()[0] if not x.empty else "cv")
        
        # Target: daily mean PM2.5 or worst category of the day
        daily_pm25 = df.groupby("Date")["pm2.5"].mean()
        daily_cat = daily_pm25.apply(classify_pm25)
        
        daily_df = daily_num.copy()
        daily_df["cbwd"] = daily_wind
        daily_df["AQI_Category"] = daily_cat
        dates = pd.to_datetime(daily_df.index)
        daily_df = daily_df.reset_index(drop=True)
        daily_df["Date"] = dates
        daily_df = daily_df.sort_values("Date").reset_index(drop=True)
        daily_df = daily_df.dropna(subset=["AQI_Category"]).copy()
        return daily_df
    else:
        # Drop index and target-leakage
        df = df.drop(columns=["No", "pm2.5"])
        return df


def load_seoul_dataset(filepath: str, sample_stations: bool = True) -> pd.DataFrame:
    """
    Load Seoul Air Pollution dataset (Measurement_summary.csv).
    - Columns: Measurement date, Station code, Address, Latitude, Longitude, SO2, NO2, O3, CO, PM10, PM2.5
    - Handle negative values (sensor error/calibration) by replacing with NaN.
    - Construct target category from PM2.5 concentration bands.
    - Exclude PM2.5 and Address from predictors.
    - Aggregate to daily means per station for robustness and training efficiency.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Seoul dataset not found at {filepath}")

    # Read essential columns
    cols = ["Measurement date", "Station code", "Latitude", "Longitude", "SO2", "NO2", "O3", "CO", "PM10", "PM2.5"]
    df = pd.read_csv(filepath, usecols=cols)

    # Clean negative values in pollutant measurements
    pollutant_cols = ["SO2", "NO2", "O3", "CO", "PM10", "PM2.5"]
    for col in pollutant_cols:
        df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)

    df["DateTime"] = pd.to_datetime(df["Measurement date"])
    df["Date"] = df["DateTime"].dt.date

    # Aggregate by Date and Station code to daily averages
    agg_dict = {
        "SO2": "mean",
        "NO2": "mean",
        "O3": "mean",
        "CO": "mean",
        "PM10": "mean",
        "PM2.5": "mean",
        "Latitude": "first",
        "Longitude": "first"
    }
    daily_df = df.groupby(["Date", "Station code"], as_index=False).agg(agg_dict)
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    daily_df = daily_df.sort_values(["Date", "Station code"]).reset_index(drop=True)

    # Construct target category from daily mean PM2.5
    daily_df["AQI_Category"] = daily_df["PM2.5"].apply(classify_pm25)
    daily_df = daily_df.dropna(subset=["AQI_Category"]).copy()
    daily_df = daily_df[daily_df["AQI_Category"].isin(AQI_CATEGORIES)].copy()

    # Drop PM2.5 (target leakage)
    daily_df = daily_df.drop(columns=["PM2.5"])

    return daily_df
