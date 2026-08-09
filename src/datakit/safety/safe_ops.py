"""Broadcasting-aware safety module for NumPy and Pandas operations.

Prevents or warns on dangerous implicit broadcasting (e.g. subtracting (n,)
from (n,1) to produce a (n,n) outer product), index misalignment, and memory view leaks.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from datakit.core.exceptions import ShapeMismatchError
from datakit.core.results import AlignCheckResult, ShapeCheckResult
from datakit.core.warnings import ImplicitBroadcastWarning, IndexAlignmentWarning


def _get_shape(x: Any) -> tuple[int, ...]:
    if hasattr(x, "shape"):
        return tuple(x.shape)
    arr = np.asarray(x)
    return tuple(arr.shape)


def _check_broadcast_and_warn(
    a: Any, b: Any, op_name: str, strict: bool = False
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    shape_a = _get_shape(a)
    shape_b = _get_shape(b)

    # Calculate actual result shape
    dummy_a = np.empty(shape_a, dtype=np.int8)
    dummy_b = np.empty(shape_b, dtype=np.int8)

    try:
        res_dummy = np.broadcast_shapes(shape_a, shape_b)
        result_shape = tuple(res_dummy)
    except ValueError as e:
        raise ShapeMismatchError(shape_a, shape_b, ()) from e

    # Implicit broadcasting occurs when ranks differ or when a 1-D array is broadcast with a 2-D column/row vector
    is_rank_mismatch = len(shape_a) != len(shape_b)
    is_outer_product_risk = (
        (len(shape_a) == 1 and len(shape_b) == 2 and shape_b[1] == 1)
        or (len(shape_b) == 1 and len(shape_a) == 2 and shape_a[1] == 1)
    )

    if is_rank_mismatch or is_outer_product_risk:
        msg = (
            f"Implicit broadcasting in {op_name}: array of shape {shape_a} and array of shape {shape_b} "
            f"produced result of shape {result_shape}. NumPy interpreted rank mismatch as an implicit expansion. "
            f"Ensure equal-rank arrays or explicit reshaping (e.g., a[:, None]) if this was intended."
        )
        if strict:
            raise ShapeMismatchError(shape_a, shape_b, result_shape)
        warnings.warn(msg, ImplicitBroadcastWarning, stacklevel=3)

    return shape_a, shape_b, result_shape


def subtract(a: Any, b: Any, strict: bool = False) -> Any:
    """Safe subtraction that detects implicit rank-mismatched broadcasting.

    Args:
        a: Array-like or numeric.
        b: Array-like or numeric.
        strict: If True, raises ShapeMismatchError on implicit broadcast.
            If False (default), issues ImplicitBroadcastWarning.

    Returns:
        Result of a - b.
    """
    _check_broadcast_and_warn(a, b, "subtract", strict=strict)
    return a - b


def add(a: Any, b: Any, strict: bool = False) -> Any:
    """Safe addition that detects implicit rank-mismatched broadcasting."""
    _check_broadcast_and_warn(a, b, "add", strict=strict)
    return a + b


def multiply(a: Any, b: Any, strict: bool = False) -> Any:
    """Safe multiplication that detects implicit rank-mismatched broadcasting."""
    _check_broadcast_and_warn(a, b, "multiply", strict=strict)
    return a * b


def divide(a: Any, b: Any, strict: bool = False) -> Any:
    """Safe division that detects implicit rank-mismatched broadcasting."""
    _check_broadcast_and_warn(a, b, "divide", strict=strict)
    return a / b


def check_shapes(a: Any, b: Any) -> ShapeCheckResult:
    """Inspect shape compatibility and explain how NumPy/Pandas would interpret them.

    Args:
        a: First input array-like.
        b: Second input array-like.

    Returns:
        ShapeCheckResult object.
    """
    shape_a = _get_shape(a)
    shape_b = _get_shape(b)

    try:
        res_shape = tuple(np.broadcast_shapes(shape_a, shape_b))
        compatible = True
    except ValueError:
        return ShapeCheckResult(
            shape_a=shape_a,
            shape_b=shape_b,
            compatible=False,
            is_implicit_broadcast=False,
            result_shape=None,
            explanation=f"Shapes {shape_a} and {shape_b} are incompatible for broadcasting.",
        )

    is_implicit = len(shape_a) != len(shape_b) or (
        (len(shape_a) == 1 and len(shape_b) == 2 and shape_b[1] == 1)
        or (len(shape_b) == 1 and len(shape_a) == 2 and shape_a[1] == 1)
    )

    if is_implicit:
        explanation = (
            f"Shapes {shape_a} and {shape_b} are compatible via IMPLICIT broadcasting, "
            f"producing shape {res_shape}. This reinterprets rank differences as outer dimensions."
        )
    else:
        explanation = f"Shapes {shape_a} and {shape_b} are compatible, producing shape {res_shape}."

    return ShapeCheckResult(
        shape_a=shape_a,
        shape_b=shape_b,
        compatible=compatible,
        is_implicit_broadcast=is_implicit,
        result_shape=res_shape,
        explanation=explanation,
    )


def is_view(x: Any) -> bool:
    """Check whether a NumPy array or Pandas object shares memory with a parent object.

    Args:
        x: NumPy ndarray, Pandas Series, or Pandas DataFrame.

    Returns:
        True if x shares memory with another array or has a base array, False otherwise.
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x_arr = x.values
    else:
        x_arr = np.asarray(x)

    base = getattr(x_arr, "base", None)
    return base is not None


def align_check(df1: pd.DataFrame | pd.Series, df2: pd.DataFrame | pd.Series, how: str = "outer") -> AlignCheckResult:
    """Check index overlap between two Pandas objects before concat or join.

    Args:
        df1: First DataFrame or Series.
        df2: Second DataFrame or Series.
        how: Alignment comparison mode ("outer" or "inner").

    Returns:
        AlignCheckResult object.

    Raises:
        IndexAlignmentWarning: If index sets are not identical.
    """
    idx1 = set(df1.index)
    idx2 = set(df2.index)

    overlapping = sorted(list(idx1 & idx2), key=str)
    non_overlap1 = sorted(list(idx1 - idx2), key=str)
    non_overlap2 = sorted(list(idx2 - idx1), key=str)

    total_unique = len(idx1 | idx2)
    match_pct = (len(overlapping) / total_unique * 100) if total_unique > 0 else 100.0

    if non_overlap1 or non_overlap2:
        explanation = (
            f"Index sets do not match perfectly ({match_pct:.1f}% overlap). "
            f"{len(non_overlap1)} labels in df1 not in df2; {len(non_overlap2)} labels in df2 not in df1. "
            f"Operations like pd.concat(axis=1) will introduce NaNs for non-overlapping labels."
        )
        warnings.warn(explanation, IndexAlignmentWarning, stacklevel=2)
    else:
        explanation = f"Index sets match perfectly ({len(overlapping)} identical labels)."

    return AlignCheckResult(
        overlapping_labels=overlapping,
        non_overlapping_df1=non_overlap1,
        non_overlapping_df2=non_overlap2,
        match_pct=match_pct,
        explanation=explanation,
    )


def reshape_column(series_or_array: Any) -> np.ndarray:
    """Explicit wrapper converting 1D Series or array into 2D column vector of shape (n, 1).

    Purpose:
        Fixes rank mismatches flagged by ImplicitBroadcastWarning by converting 1D shape (n,) to 2D (n, 1).

    Params:
        series_or_array (Any): 1D Pandas Series, list, or NumPy array.

    Returns:
        np.ndarray: 2D column vector of shape (n, 1).

    Mutates: No (returns new 2D array view/copy).
    Chainable: No.
    Version Added: v0.1.0
    """
    arr = np.asarray(series_or_array)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        return arr
    else:
        raise ValueError(f"reshape_column expects 1D array or (n, 1) vector. Got array with shape {arr.shape}.")
