"""Kernel Density Estimate (KDE) plot module for DataKit.
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


def plot_kde(
    df: pd.DataFrame,
    column: str,
    hue: str | None = None,
    ax: Any = None,
    instance_style: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PlotResult:
    """Plot a standalone Kernel Density Estimate (KDE) plot for a numeric column.

    Purpose:
        Visualizes the smooth probability density function of a numeric feature without histogram bars.

    Params:
        df (pd.DataFrame): Source DataFrame.
        column (str): Numeric column name to plot.
        hue (str | None): Categorical column name for color grouping.
        ax (Any | None): Optional Matplotlib Axes to draw into.
        instance_style (dict | None): Instance-level plot configuration defaults.
        **kwargs: Layout overrides (title, xlabel, ylabel, xlim, ylim, grid, legend).

    Returns:
        PlotResult: Dataclass holding fig, ax, and call_info.

    Mutates: No (returns PlotResult wrapping new or injected Axes).
    Chainable: No.
    Version Added: v0.1.0

    Errors:
        ColumnNotFoundError: If column or hue is not in DataFrame.
        IncompatibleColumnTypeError: If column is not numeric.
    """
    if column not in df.columns:
        raise ColumnNotFoundError(column, list(df.columns))

    if hue and hue not in df.columns:
        raise ColumnNotFoundError(hue, list(df.columns))

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

    sns.kdeplot(
        data=df,
        x=column,
        hue=hue,
        ax=target_ax,
    )

    if "title" not in kwargs and "title" not in (instance_style or {}):
        kwargs["title"] = f"Density Plot of {column}"

    apply_common_layout(fig, target_ax, instance_style, **kwargs)

    call_info = f"sns.kdeplot(data=df, x='{column}', hue={hue!r})"
    return PlotResult(fig=fig, ax=target_ax, call_info=call_info)
