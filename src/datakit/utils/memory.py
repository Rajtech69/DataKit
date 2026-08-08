"""Memory profiling and downcasting suggestion module for DataKit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from datakit.core.results import MemoryResult


def profile_memory(df: pd.DataFrame) -> MemoryResult:
    """Analyze memory consumption per column and provide downcasting recommendations.

    Args:
        df: Input DataFrame.

    Returns:
        MemoryResult object.
    """
    by_col_bytes = df.memory_usage(deep=True)
    # Exclude Index if present in memory_usage
    if "Index" in by_col_bytes.index:
        by_col_bytes = by_col_bytes.drop("Index")

    by_col_mb = by_col_bytes / (1024 * 1024)
    total_mb = float(by_col_bytes.sum() / (1024 * 1024))

    suggestions: list[str] = []
    n_rows = len(df)

    if n_rows > 0:
        for col in df.columns:
            series = df[col]
            dtype = series.dtype

            if pd.api.types.is_float_dtype(dtype) and dtype != np.float32:
                suggestions.append(
                    f"Column '{col}' is {dtype}. Downcast to float32 to reduce memory by ~50%."
                )
            elif pd.api.types.is_integer_dtype(dtype) and dtype != np.int8:
                c_min = series.min()
                c_max = series.max()
                if c_min >= -128 and c_max <= 127:
                    suggested = "int8"
                elif c_min >= -32768 and c_max <= 32767:
                    suggested = "int16"
                elif c_min >= -2147483648 and c_max <= 2147483647:
                    suggested = "int32"
                else:
                    suggested = None

                if suggested and str(dtype) != suggested:
                    suggestions.append(
                        f"Column '{col}' values range [{c_min}, {c_max}]. Downcast from {dtype} to {suggested}."
                    )
            elif pd.api.types.is_object_dtype(dtype):
                n_unique = series.nunique(dropna=True)
                if n_unique / n_rows < 0.5:
                    suggestions.append(
                        f"Column '{col}' has low cardinality ({n_unique} unique values / {n_rows} rows). Convert object to category."
                    )

    return MemoryResult(
        by_column=by_col_mb,
        total_mb=total_mb,
        suggestions=suggestions,
    )
