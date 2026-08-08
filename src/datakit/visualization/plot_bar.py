"""Bar plot module for DataKit.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from datakit.core.exceptions import ColumnNotFoundError
from datakit.core.results import PlotResult
from datakit.visualization.common import apply_common_layout
from datakit.visualization.config import ConfigResolver


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str | None = None,
    agg: str = "count",
    hue: str | None = None,
    orient: str = "v",
    ax: Any = None,
    instance_style: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PlotResult:
    if x not in df.columns:
        raise ColumnNotFoundError(x, list(df.columns))

    if y and y not in df.columns:
        raise ColumnNotFoundError(y, list(df.columns))

    figsize = ConfigResolver.resolve("figsize", kwargs.get("figsize"), instance_style)
    dpi = ConfigResolver.resolve("dpi", kwargs.get("dpi"), instance_style)

    if ax is None:
        fig, target_ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        target_ax = ax
        fig = target_ax.get_figure()

    if y is None or agg == "count":
        sns.countplot(
            data=df,
            x=x if orient == "v" else None,
            y=x if orient == "h" else None,
            hue=hue,
            ax=target_ax,
        )
    else:
        estimator = "mean" if agg == "mean" else "sum" if agg == "sum" else agg
        sns.barplot(
            data=df,
            x=x if orient == "v" else y,
            y=y if orient == "v" else x,
            estimator=estimator,
            hue=hue,
            ax=target_ax,
        )

    if "title" not in kwargs and "title" not in (instance_style or {}):
        kwargs["title"] = f"Bar Plot of {x}" + (f" ({y} {agg})" if y else "")

    apply_common_layout(fig, target_ax, instance_style, **kwargs)

    call_info = f"sns.barplot/countplot(data=df, x='{x}', y={y!r}, agg={agg!r})"
    return PlotResult(fig=fig, ax=target_ax, call_info=call_info)
