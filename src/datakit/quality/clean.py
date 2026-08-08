"""Data cleaning module for DataKit.

Provides guided, report-first, non-mutating data cleaning operations with
mandatory confirmation for destructive actions.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd

from datakit.config import config
from datakit.core.exceptions import ConfirmationRequiredError
from datakit.core.infer import infer_suspicious_dtypes
from datakit.core.results import CleanReport
from datakit.core.warnings import DataLossWarning


def clean_dataframe(
    df: pd.DataFrame,
    missing: Literal["report", "drop", "impute_mean", "impute_median", "impute_mode"] = "report",
    duplicates: Literal["report", "drop"] = "report",
    dtypes: Literal["report", "coerce"] = "report",
    confirm: bool = False,
) -> tuple[pd.DataFrame, CleanReport]:
    """Clean a DataFrame with mandatory confirmation for destructive strategies.

    Args:
        df: Input DataFrame (never mutated).
        missing: Strategy for missing values ("report", "drop", "impute_mean", "impute_median", "impute_mode").
        duplicates: Strategy for duplicate rows ("report", "drop").
        dtypes: Strategy for object/string columns containing numeric values ("report", "coerce").
        confirm: Must be True to apply any non-report strategy.

    Returns:
        Tuple of (new_cleaned_dataframe, clean_report).

    Raises:
        ConfirmationRequiredError: If non-report strategy specified without confirm=True.
    """
    is_destructive = (missing != "report") or (duplicates != "report") or (dtypes != "report")
    if is_destructive and not confirm:
        strategies = []
        if missing != "report":
            strategies.append(f"missing='{missing}'")
        if duplicates != "report":
            strategies.append(f"duplicates='{duplicates}'")
        if dtypes != "report":
            strategies.append(f"dtypes='{dtypes}'")
        op_str = ", ".join(strategies)
        raise ConfirmationRequiredError(f"clean({op_str})")

    # Start with a copy
    new_df = df.copy()
    original_shape = (int(df.shape[0]), int(df.shape[1]))
    rows_dropped = 0
    values_imputed: dict[str, int] = {}
    dtype_changes: list[tuple[str, str, str]] = []

    # 1. Dtypes Coercion
    if dtypes == "coerce":
        suspicious = infer_suspicious_dtypes(new_df)
        for col, target_type in suspicious:
            old_dtype_str = str(new_df[col].dtype)
            if target_type == "numeric":
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
                dtype_changes.append((col, old_dtype_str, str(new_df[col].dtype)))
            elif target_type == "datetime":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    new_df[col] = pd.to_datetime(new_df[col], errors="coerce")
                dtype_changes.append((col, old_dtype_str, str(new_df[col].dtype)))

    # 2. Duplicate Removal
    if duplicates == "drop":
        n_before = len(new_df)
        new_df = new_df.drop_duplicates().reset_index(drop=True)
        dropped_dups = n_before - len(new_df)
        rows_dropped += dropped_dups

    # 3. Missing Value Handling
    if missing == "drop":
        n_before = len(new_df)
        new_df = new_df.dropna().reset_index(drop=True)
        dropped_nulls = n_before - len(new_df)
        rows_dropped += dropped_nulls
    elif missing.startswith("impute_"):
        mode_type = missing.replace("impute_", "")
        for col in new_df.columns:
            null_count = int(new_df[col].isnull().sum())
            if null_count == 0:
                continue

            series = new_df[col]
            impute_val = None

            if mode_type == "mean" and pd.api.types.is_numeric_dtype(series.dtype):
                impute_val = series.mean()
            elif mode_type == "median" and pd.api.types.is_numeric_dtype(series.dtype):
                impute_val = series.median()
            elif mode_type == "mode":
                modes = series.mode()
                if not modes.empty:
                    impute_val = modes.iloc[0]

            if impute_val is not None:
                new_df[col] = series.fillna(impute_val)
                values_imputed[str(col)] = null_count

    # Check data loss warning
    new_shape = (int(new_df.shape[0]), int(new_df.shape[1]))
    if original_shape[0] > 0:
        loss_pct = rows_dropped / original_shape[0]
        loss_thresh = config.get("data_loss_warning_threshold")
        if loss_pct >= loss_thresh:
            warnings.warn(
                f"Cleaning operation dropped {rows_dropped} rows ({loss_pct*100:.1f}% of dataset), "
                f"which exceeds the warning threshold ({loss_thresh*100:.0f}%).",
                DataLossWarning,
                stacklevel=2,
            )

    report = CleanReport(
        rows_dropped=rows_dropped,
        values_imputed=values_imputed,
        dtype_changes=dtype_changes,
        original_shape=original_shape,
        new_shape=new_shape,
    )

    return new_df, report
