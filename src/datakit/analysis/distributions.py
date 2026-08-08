"""Distributions shape analysis module for DataKit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from datakit.core.exceptions import ColumnNotFoundError, IncompatibleColumnTypeError
from datakit.core.results import DistributionResult


def analyze_distributions(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> DistributionResult:
    """Summarize distribution shapes (skewness, kurtosis, normality heuristic) per numeric column.

    Args:
        df: Input DataFrame.
        columns: List of column names to analyze, or None for all numeric columns.

    Returns:
        DistributionResult object.

    Raises:
        IncompatibleColumnTypeError: If non-numeric column is explicitly requested.
    """
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    if columns is not None:
        target_cols = []
        for col in columns:
            if col not in df.columns:
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

    rows: list[dict] = []
    for col in target_cols:
        series = df[col].dropna()
        if len(series) < 3:
            skew_val = 0.0
            kurt_val = 0.0
            likely_norm = False
        else:
            skew_val = float(series.skew())
            kurt_val = float(series.kurt())
            # Heuristic: moderately symmetric (|skew| <= 0.5) and meso-kurtic (|kurt| <= 1.0)
            likely_norm = bool(abs(skew_val) <= 0.5 and abs(kurt_val) <= 1.0)

        rows.append(
            {
                "column": col,
                "skew": round(skew_val, 4),
                "kurtosis": round(kurt_val, 4),
                "likely_normal": likely_norm,
            }
        )

    if rows:
        stats_df = pd.DataFrame(rows).set_index("column")
    else:
        stats_df = pd.DataFrame(columns=["skew", "kurtosis", "likely_normal"])

    return DistributionResult(stats=stats_df)
