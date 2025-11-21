"""
data_processing.py
Utilities for loading, cleaning, batching, and preparing categorical data.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np


def load_csv(path: str, encoding: Optional[str] = "utf-8") -> pd.DataFrame:
    """Load CSV into DataFrame with basic checks."""
    df = pd.read_csv(path, encoding=encoding)
    if df.empty:
        raise ValueError(f"Loaded dataframe from {path} is empty.")
    return df


def clean_categorical_columns(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Basic cleaning for categorical columns:
    - strip whitespace
    - fill NaN with 'Missing'
    - optionally lowercase (preserve original? depends on project)
    """
    df = df.copy()
    for c in categorical_cols:
        if c not in df.columns:
            raise KeyError(f"Column {c} not in dataframe.")
        # Convert to string, strip, replace empty with Missing
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"": "Missing", "nan": "Missing", "None": "Missing"})
        df[c] = df[c].fillna("Missing")
    return df


def define_batches_by_column(df: pd.DataFrame, batch_col: str) -> List[pd.DataFrame]:
    """
    Split dataframe into list of batches using unique values in batch_col.
    Useful when batches are e.g., month, region, file_id.
    """
    batches = []
    if batch_col not in df.columns:
        raise KeyError(f"batch_col {batch_col} not found in dataframe.")
    for val, group in df.groupby(batch_col):
        batches.append(group.reset_index(drop=True))
    return batches


def sliding_window_batches(df: pd.DataFrame, time_col: str, window_size: int, step: int = 1, sort_ascending: bool = True) -> List[pd.DataFrame]:
    """
    Create time-windowed batches given a time column (must be datetime-like).
    window_size and step in number of rows. Useful for streaming-like data.
    """
    df = df.sort_values(by=time_col, ascending=sort_ascending).reset_index(drop=True)
    batches = []
    n = len(df)
    start = 0
    while start < n:
        end = min(start + window_size, n)
        batches.append(df.iloc[start:end].reset_index(drop=True))
        start += step
        if end == n:
            break
    return batches


def generate_synthetic_dataset(n_rows: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset with categorical columns and a batch column.
    Useful for demo and testing.
    """
    rng = np.random.default_rng(random_state)
    regions = ["North", "South", "East", "West"]
    property_type = ["Apartment", "Independent", "Villa", "Studio"]
    satisfaction = ["Positive", "Neutral", "Negative"]
    district = [f"D{i}" for i in range(1, 11)]
    batch = [f"Batch_{i}" for i in rng.integers(1, 6, size=n_rows)]

    df = pd.DataFrame({
        "Region": rng.choice(regions, size=n_rows, p=[0.25, 0.25, 0.25, 0.25]),
        "PropertyType": rng.choice(property_type, size=n_rows),
        "Satisfaction": rng.choice(satisfaction, size=n_rows, p=[0.5, 0.2, 0.3]),
        "District": rng.choice(district, size=n_rows),
        "BatchID": batch,
        "Price": rng.normal(50_00_000, 20_00_000, size=n_rows).round(0)  # numeric example
    })
    # inject some missing
    mask = rng.random(n_rows) < 0.02
    df.loc[mask, "PropertyType"] = np.nan
    return df
