"""Count plot module for DataKit.
"""
from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from datakit.config import config
from datakit.core.exceptions import ColumnNotFoundError
from datakit.core.results import PlotResult
from datakit.core.warnings import HighCardinalityWarning
from datakit.visualization.common import apply_common_layout
from datakit.visualization.config import ConfigResolver


def plot_count(
    df: pd.DataFrame,
    column: str,
    hue: str | None = None,
    order: list[Any] | None = None,
    ax: Any = None,
    instance_style: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PlotResult:
    if column not in df.columns:
        raise ColumnNotFoundError(column, list(df.columns))

    n_unique = df[column].nunique(dropna=True)
    max_cats = config.get("count_plot_max_categories")
    if n_unique > max_cats:
        warnings.warn(
            f"Column '{column}' has {n_unique} categories, which exceeds max_categories ({max_cats}). "
            f"The plot may be unreadable; consider filtering to top N categories.",
            HighCardinalityWarning,
        )

    figsize = ConfigResolver.resolve("figsize", kwargs.get("figsize"), instance_style)
    dpi = ConfigResolver.resolve("dpi", kwargs.get("dpi"), instance_style)

    if ax is None:
        fig, target_ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        target_ax = ax
        fig = target_ax.get_figure()

    if order is None:
        order = df[column].value_counts().index.tolist()

    sns.countplot(
        data=df,
        x=column,
        hue=hue,
        order=order,
        ax=target_ax,
    )

    if "title" not in kwargs and "title" not in (instance_style or {}):
        kwargs["title"] = f"Category Counts for {column}"
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = column
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Count"

    apply_common_layout(fig, target_ax, instance_style, **kwargs)

    call_info = f"sns.countplot(data=df, x='{column}', hue={hue!r})"
    return PlotResult(fig=fig, ax=target_ax, call_info=call_info)
