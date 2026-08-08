"""Outlier detection module for DataKit.

Provides univariate outlier detection using IQR and Z-Score methods with explainable results.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from datakit.config import config
from datakit.core.exceptions import IncompatibleColumnTypeError
from datakit.core.results import OutlierResult


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float | None = None,
) -> OutlierResult:
    """Detect outliers per numeric column using standard statistical methods.

    Args:
        df: Input DataFrame.
        columns: List of column names to analyze, or None for all numeric columns.
        method: Outlier detection method ("iqr" or "zscore").
        threshold: Multiplier for IQR (default: 1.5) or z-score cutoff (default: 3.0).

    Returns:
        OutlierResult object.

    Raises:
        IncompatibleColumnTypeError: If non-numeric column is explicitly requested.
        ValueError: If unsupported method is requested.
    """
    if method not in ("iqr", "zscore"):
        raise ValueError(f"Unsupported outlier detection method '{method}'. Supported: 'iqr', 'zscore'.")

    if threshold is None:
        threshold = (
            config.get("outlier_iqr_multiplier")
            if method == "iqr"
            else config.get("outlier_zscore_threshold")
        )

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    if columns is not None:
        target_cols = []
        for col in columns:
            if col not in df.columns:
                from datakit.core.exceptions import ColumnNotFoundError
                raise ColumnNotFoundError(col, list(df.columns))
            if not pd.api.types.is_numeric_dtype(df[col].dtype):
                raise IncompatibleColumnTypeError(
                    column=str(col),
                    actual_dtype=str(df[col].dtype),
                    expected_dtypes=["int64", "float64", "numeric"],
                )
            target_cols.append(str(col))
    else:
        target_cols = [str(c) for c in numeric_cols]

    counts: dict[str, int] = {}
    indices: dict[str, list[int]] = {}

    for col in target_cols:
        series = df[col].dropna()
        if len(series) == 0:
            counts[col] = 0
            indices[col] = []
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        else:  # zscore
            mean = series.mean()
            std = series.std(ddof=0)
            if std == 0:
                outlier_mask = pd.Series(False, index=df.index)
            else:
                z_scores = (df[col] - mean) / std
                outlier_mask = z_scores.abs() > threshold

        # Get original DataFrame row indices where outlier_mask is True
        outlier_indices = df.index[outlier_mask.fillna(False)].tolist()
        counts[col] = len(outlier_indices)
        indices[col] = [int(idx) for idx in outlier_indices]

    return OutlierResult(
        counts=counts,
        indices=indices,
        method=method,
        threshold=threshold,
    )
