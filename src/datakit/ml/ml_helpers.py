"""Machine Learning dataset preparation helpers for DataKit.

Provides non-destructive target label encoding, cross-validation split generators,
and target class imbalance analysis.
"""
from __future__ import annotations

from typing import Any, Generator

import numpy as np
import pandas as pd

from datakit.core.exceptions import ColumnNotFoundError, DataKitError


def encode_target_labels(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.Series, dict[Any, int]]:
    """Encode a categorical target column into integer labels with explicit reverse mapping.

    Purpose:
        Safely converts string or categorical target variables into 0-indexed integer labels
        for machine learning classification algorithms while preserving exact label mappings.

    Params:
        df (pd.DataFrame): Input DataFrame.
        target (str): Target column name to encode.

    Returns:
        tuple[pd.Series, dict[Any, int]]:
            - Encoded Series with integer values (0, 1, 2, ...).
            - Dictionary mapping original category labels to assigned integers.

    Mutates: No (returns new Series and mapping dictionary).
    Chainable: No.
    Version Added: v0.2.0

    Errors:
        ColumnNotFoundError: If target column is missing from DataFrame.
    """
    if target not in df.columns:
        raise ColumnNotFoundError(target, list(df.columns))

    series = df[target]
    unique_vals = series.dropna().unique()
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    encoded = series.map(mapping)

    return encoded, mapping


def check_imbalance(
    df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Analyze target class distribution and compute class imbalance ratios.

    Purpose:
        Diagnoses class imbalance in categorical targets to determine whether resampling,
        stratified splitting, or class weighting is required.

    Params:
        df (pd.DataFrame): Input DataFrame.
        target (str): Classification target column name.

    Returns:
        pd.DataFrame: Summary table containing class counts, percentages, and ratio relative to majority class.

    Mutates: No (returns new summary DataFrame).
    Chainable: No.
    Version Added: v0.2.0

    Errors:
        ColumnNotFoundError: If target column is missing from DataFrame.
    """
    if target not in df.columns:
        raise ColumnNotFoundError(target, list(df.columns))

    counts = df[target].value_counts(dropna=False)
    pcts = (counts / len(df) * 100).round(2)
    max_count = counts.max()
    ratios = (max_count / counts).round(2)

    summary_df = pd.DataFrame(
        {
            "count": counts,
            "percentage": pcts,
            "imbalance_ratio": ratios,
        }
    )
    summary_df.index.name = "class"
    return summary_df


def create_cv_splits(
    df: pd.DataFrame,
    target: str | None = None,
    n_splits: int = 5,
    stratified: bool = True,
    random_state: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate K-Fold or Stratified K-Fold cross-validation index splits.

    Purpose:
        Produces index pairs for reproducible cross-validation evaluation without global state bleed.

    Params:
        df (pd.DataFrame): Input DataFrame.
        target (str | None): Target column for stratified splitting, or None for standard K-Fold.
        n_splits (int): Number of folds (default: 5).
        stratified (bool): If True and target is provided, uses Stratified K-Fold.
        random_state (int | None): Random seed for reproducible fold generation.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: List of (train_indices, validation_indices) tuples.

    Mutates: No (returns new index arrays).
    Chainable: No.
    Version Added: v0.2.0

    Errors:
        ImportError: If scikit-learn is not installed.
        ColumnNotFoundError: If target is specified but missing.
    """
    try:
        from sklearn.model_selection import KFold, StratifiedKFold
    except ImportError as e:
        raise ImportError(
            "cv_splits() requires scikit-learn. Install it with: pip install 'datakit[ml]'"
        ) from e

    if target:
        if target not in df.columns:
            raise ColumnNotFoundError(target, list(df.columns))
        if stratified:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return list(skf.split(df, df[target]))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(df))
