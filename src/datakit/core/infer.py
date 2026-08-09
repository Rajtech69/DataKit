"""Shared deterministic inference engine for DataKit.

All inference logic (type detection, ID-like column detection, target candidate
detection, suspicious dtype detection) is centralized here so it is consistent across
audit(), inspect(), eda(), and plot.* modules.
"""
from __future__ import annotations

import re
import warnings
from typing import Literal

import numpy as np
import pandas as pd

from datakit.core.results import CompareResult

from datakit.config import config

ColumnType = Literal["numeric", "categorical", "datetime", "boolean"]


def infer_column_types(
    df: pd.DataFrame,
    cardinality_threshold: int | None = None,
) -> dict[str, ColumnType]:
    """Infer semantic column types (numeric, categorical, datetime, boolean).

    Args:
        df: Input DataFrame.
        cardinality_threshold: Max unique values for numeric columns to be
            treated as categorical. Defaults to config setting.

    Returns:
        Mapping of column name to inferred type string.
    """
    if cardinality_threshold is None:
        cardinality_threshold = config.get("high_cardinality_threshold")

    types: dict[str, ColumnType] = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        dtype = series.dtype

        if pd.api.types.is_bool_dtype(dtype):
            types[col] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(dtype):
            # Check if low-cardinality integer-like column (e.g., 0/1/2 codes)
            n_unique = series.nunique(dropna=True)
            if (
                n_rows > 10
                and n_unique <= min(cardinality_threshold, 10)
                and pd.api.types.is_integer_dtype(dtype)
            ):
                types[col] = "categorical"
            else:
                types[col] = "numeric"
        elif pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            types[col] = "categorical"
        else:
            types[col] = "categorical"

    return types


def infer_id_like_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that appear to be row identifiers or primary keys.

    Requires BOTH:
    1. Uniqueness ratio > id_like_uniqueness_threshold (default 0.95)
    2. Name matches identifier patterns (id, _id, uuid, index, etc.)
    """
    if len(df) == 0:
        return []

    uniqueness_threshold = config.get("id_like_uniqueness_threshold")
    id_cols: list[str] = []
    id_pattern = re.compile(r"(^|_)(id|uuid|guid|pk|index)($|_)", re.IGNORECASE)

    for col in df.columns:
        col_str = str(col)
        n_unique = df[col].nunique(dropna=True)
        uniqueness_ratio = n_unique / len(df)

        if uniqueness_ratio >= uniqueness_threshold:
            if id_pattern.search(col_str) or col_str.lower() in ("id", "index", "code"):
                id_cols.append(col_str)

    return id_cols


def infer_target_candidates(df: pd.DataFrame) -> list[str]:
    """Identify columns that are likely machine learning target variables.

    Heuristics:
    1. Common target column names (target, label, y, outcome, class, charges, price, response)
    2. Position (last column is often target)
    3. Excludes ID-like columns
    """
    if df.empty:
        return []

    id_cols = set(infer_id_like_columns(df))
    candidates: list[str] = []

    target_keywords = re.compile(
        r"(^|_)(target|label|y|outcome|class|response|status|price|charges|salary|score)($|_)",
        re.IGNORECASE,
    )

    # 1. Keyword matching
    for col in df.columns:
        col_str = str(col)
        if col_str in id_cols:
            continue
        if target_keywords.search(col_str):
            candidates.append(col_str)

    # 2. If no candidate found, suggest the last non-ID column
    if not candidates:
        for col in reversed(list(df.columns)):
            col_str = str(col)
            if col_str not in id_cols:
                candidates.append(col_str)
                break

    return candidates


def infer_suspicious_dtypes(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Detect object columns that contain numeric or datetime data.

    Returns:
        List of tuples: (column_name, suggested_dtype_str)
    """
    suspicious: list[tuple[str, str]] = []

    for col in df.columns:
        series = df[col]
        # Skip already typed numeric, datetime, bool
        if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_datetime64_any_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
            continue

        non_null = series.dropna()
        if len(non_null) == 0:
            continue

        # Try converting to numeric
        converted_num = pd.to_numeric(non_null, errors="coerce")
        valid_num_count = converted_num.notnull().sum()
        if valid_num_count / len(non_null) >= 0.8:
            # High proportion of valid numbers stored as object/string
            suspicious.append((str(col), "numeric"))
            continue

        # Try converting to datetime
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted_dt = pd.to_datetime(non_null, errors="coerce")
            valid_dt_count = converted_dt.notnull().sum()
            if valid_dt_count / len(non_null) >= 0.8:
                suspicious.append((str(col), "datetime"))
        except Exception:
            pass

    return suspicious


def compare_datasets(df_a: pd.DataFrame, df_b: pd.DataFrame) -> CompareResult:
    """Compare two DataFrames structurally, analyzing shape, added/removed columns, and dtype changes.

    Args:
        df_a: First DataFrame (e.g. baseline or original).
        df_b: Second DataFrame (e.g. transformed or target).

    Returns:
        CompareResult object.
    """
    shape_a = (df_a.shape[0], df_a.shape[1])
    shape_b = (df_b.shape[0], df_b.shape[1])

    cols_a = [str(c) for c in df_a.columns]
    cols_b = [str(c) for c in df_b.columns]

    added = [c for c in cols_b if c not in cols_a]
    removed = [c for c in cols_a if c not in cols_b]
    common = [c for c in cols_a if c in cols_b]

    dtype_changes: list[tuple[str, str, str]] = []
    for col in common:
        type_a = str(df_a[col].dtype)
        type_b = str(df_b[col].dtype)
        if type_a != type_b:
            dtype_changes.append((col, type_a, type_b))

    return CompareResult(
        shape_a=shape_a,
        shape_b=shape_b,
        added_columns=added,
        removed_columns=removed,
        common_columns=common,
        dtype_changes=dtype_changes,
    )
