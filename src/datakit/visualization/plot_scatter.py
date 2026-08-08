"""Scatter plot module for DataKit.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datakit.core.exceptions import ColumnNotFoundError, IncompatibleColumnTypeError
from datakit.core.results import PlotResult
from datakit.visualization.common import apply_common_layout
from datakit.visualization.config import ConfigResolver


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    size: str | None = None,
    alpha: float | None = None,
    trend: bool = False,
    ax: Any = None,
    instance_style: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PlotResult:
    for col in (x, y):
        if col not in df.columns:
            raise ColumnNotFoundError(col, list(df.columns))

    if not pd.api.types.is_numeric_dtype(df[x].dtype):
        raise IncompatibleColumnTypeError(
            column=x,
            actual_dtype=str(df[x].dtype),
            expected_dtypes=["int64", "float64", "numeric"],
        )

    if not pd.api.types.is_numeric_dtype(df[y].dtype):
        raise IncompatibleColumnTypeError(
            column=y,
            actual_dtype=str(df[y].dtype),
            expected_dtypes=["int64", "float64", "numeric"],
        )

    figsize = ConfigResolver.resolve("figsize", kwargs.get("figsize"), instance_style)
    dpi = ConfigResolver.resolve("dpi", kwargs.get("dpi"), instance_style)

    if ax is None:
        fig, target_ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        target_ax = ax
        fig = target_ax.get_figure()

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        size=size,
        alpha=alpha,
        ax=target_ax,
    )

    if trend:
        # Add simple OLS linear trendline via polyfit
        clean_data = df[[x, y]].dropna()
        if len(clean_data) > 1:
            x_vals = clean_data[x].values
            y_vals = clean_data[y].values
            m, b = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = m * x_line + b
            target_ax.plot(x_line, y_line, color="red", linestyle="--", label="Trendline")

    if "title" not in kwargs and "title" not in (instance_style or {}):
        kwargs["title"] = f"{y} vs {x}"
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = x
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = y

    apply_common_layout(fig, target_ax, instance_style, **kwargs)

    call_info = f"sns.scatterplot(data=df, x='{x}', y='{y}', hue={hue!r}, trend={trend})"
    return PlotResult(fig=fig, ax=target_ax, call_info=call_info)
