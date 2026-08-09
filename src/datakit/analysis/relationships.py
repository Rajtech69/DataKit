"""Relationships and correlation analysis module for DataKit.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from datakit.config import config
from datakit.core.exceptions import InsufficientDataError
from datakit.core.results import RelationshipResult


def analyze_relationships(
    df: pd.DataFrame,
    method: Literal["pearson", "spearman"] = "pearson",
    threshold: float = 0.7,
) -> RelationshipResult:
    """Analyze correlations and strong pairwise relationships across numeric columns.

    Args:
        df: Input DataFrame.
        method: Correlation method ("pearson" or "spearman").
        threshold: Absolute correlation threshold (0.0 to 1.0) to flag as strong pair.

    Returns:
        RelationshipResult object.

    Raises:
        InsufficientDataError: If fewer than 2 numeric columns exist in DataFrame.
        ValueError: If unsupported correlation method requested.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unsupported correlation method '{method}'. Supported: 'pearson', 'spearman'.")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise InsufficientDataError("Relationships analysis requires at least 2 numeric columns.")

    corr_matrix = numeric_df.corr(method=method)

    strong_pairs: list[tuple[str, str, float]] = []
    cols = list(corr_matrix.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = str(cols[i])
            col2 = str(cols[j])
            val = float(corr_matrix.iloc[i, j])

            if not np.isnan(val) and abs(val) >= threshold:
                strong_pairs.append((col1, col2, val))

    # Sort strong pairs by absolute correlation value descending
    strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return RelationshipResult(
        matrix=corr_matrix,
        strong_pairs=strong_pairs,
        method=method,
    )


def get_target_correlations(
    df: pd.DataFrame,
    target: str,
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.Series:
    """Calculate and return sorted correlations of numeric features against a target column.

    Args:
        df: Input DataFrame.
        target: Target column name.
        method: Correlation method ("pearson" or "spearman").

    Returns:
        pd.Series of correlation values sorted by absolute magnitude descending.
    """
    from datakit.core.exceptions import ColumnNotFoundError

    if target not in df.columns:
        raise ColumnNotFoundError(target, list(df.columns))

    numeric_df = df.select_dtypes(include=[np.number])
    if target not in numeric_df.columns:
        raise InsufficientDataError(f"Target column '{target}' is not numeric.")

    corrs = numeric_df.corr(method=method)[target].drop(target)
    sorted_corrs = corrs.iloc[corrs.abs().argsort()[::-1]]
    return sorted_corrs
