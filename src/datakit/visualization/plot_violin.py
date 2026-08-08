"""Violin plot module for DataKit.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from datakit.core.exceptions import ColumnNotFoundError, IncompatibleColumnTypeError
from datakit.core.results import PlotResult
from datakit.visualization.common import apply_common_layout
from datakit.visualization.config import ConfigResolver


def plot_violin(
    df: pd.DataFrame,
    column: str,
    by: str | None = None,
    split: bool = False,
    ax: Any = None,
    instance_style: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PlotResult:
    if column not in df.columns:
        raise ColumnNotFoundError(column, list(df.columns))

    if by and by not in df.columns:
        raise ColumnNotFoundError(by, list(df.columns))

    if not pd.api.types.is_numeric_dtype(df[column].dtype):
        raise IncompatibleColumnTypeError(
            column=column,
            actual_dtype=str(df[column].dtype),
            expected_dtypes=["int64", "float64", "numeric"],
        )

    figsize = ConfigResolver.resolve("figsize", kwargs.get("figsize"), instance_style)
    dpi = ConfigResolver.resolve("dpi", kwargs.get("dpi"), instance_style)

    if ax is None:
        fig, target_ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        target_ax = ax
        fig = target_ax.get_figure()

    sns.violinplot(
        data=df,
        x=by,
        y=column,
        split=split,
        ax=target_ax,
    )

    if "title" not in kwargs and "title" not in (instance_style or {}):
        kwargs["title"] = f"Violin Plot of {column}" + (f" by {by}" if by else "")

    apply_common_layout(fig, target_ax, instance_style, **kwargs)

    call_info = f"sns.violinplot(data=df, column='{column}', by={by!r})"
    return PlotResult(fig=fig, ax=target_ax, call_info=call_info)
