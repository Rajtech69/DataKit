"""Histogram plot module for DataKit.
"""
from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datakit.core.exceptions import ColumnNotFoundError, IncompatibleColumnTypeError
from datakit.core.results import PlotResult
from datakit.visualization.common import apply_common_layout
from datakit.visualization.config import ConfigResolver


def plot_hist(
    df: pd.DataFrame,
    column: str,
    bins: int | list[float] | None = None,
    kde: bool = False,
    hue: str | None = None,
    stat: str = "count",
    ax: Any = None,
    instance_style: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PlotResult:
    if column not in df.columns:
        raise ColumnNotFoundError(column, list(df.columns))

    if not pd.api.types.is_numeric_dtype(df[column].dtype):
        raise IncompatibleColumnTypeError(
            column=column,
            actual_dtype=str(df[column].dtype),
            expected_dtypes=["int64", "float64", "numeric"],
        )

    # Bin count heuristic if bins unset
    if bins is None:
        n_valid = int(df[column].dropna().count())
        bins = min(50, max(5, int(math.sqrt(n_valid)))) if n_valid > 0 else 10

    figsize = ConfigResolver.resolve("figsize", kwargs.get("figsize"), instance_style)
    dpi = ConfigResolver.resolve("dpi", kwargs.get("dpi"), instance_style)

    if ax is None:
        fig, target_ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        target_ax = ax
        fig = target_ax.get_figure()

    sns.histplot(
        data=df,
        x=column,
        bins=bins,
        kde=kde,
        hue=hue,
        stat=stat,
        ax=target_ax,
    )

    if "title" not in kwargs and "title" not in (instance_style or {}):
        kwargs["title"] = f"Distribution of {column}"
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = column
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = stat.capitalize()

    apply_common_layout(fig, target_ax, instance_style, **kwargs)

    call_info = f"sns.histplot(data=df, x='{column}', bins={bins}, kde={kde}, hue={hue!r}, stat={stat!r})"
    return PlotResult(fig=fig, ax=target_ax, call_info=call_info)
