"""Line plot module for DataKit.
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


def plot_line(
    df: pd.DataFrame,
    x: str,
    y: str | list[str],
    hue: str | None = None,
    ax: Any = None,
    instance_style: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PlotResult:
    if x not in df.columns:
        raise ColumnNotFoundError(x, list(df.columns))

    if isinstance(y, str):
        if y not in df.columns:
            raise ColumnNotFoundError(y, list(df.columns))
        y_cols = [y]
    else:
        for col in y:
            if col not in df.columns:
                raise ColumnNotFoundError(col, list(df.columns))
        y_cols = y

    figsize = ConfigResolver.resolve("figsize", kwargs.get("figsize"), instance_style)
    dpi = ConfigResolver.resolve("dpi", kwargs.get("dpi"), instance_style)

    if ax is None:
        fig, target_ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        target_ax = ax
        fig = target_ax.get_figure()

    if len(y_cols) == 1:
        sns.lineplot(
            data=df,
            x=x,
            y=y_cols[0],
            hue=hue,
            ax=target_ax,
        )
    else:
        # Multi-series line plot
        melted = df.melt(id_vars=[x], value_vars=y_cols, var_name="_series", value_name="_value")
        sns.lineplot(
            data=melted,
            x=x,
            y="_value",
            hue="_series",
            ax=target_ax,
        )

    if "title" not in kwargs and "title" not in (instance_style or {}):
        kwargs["title"] = f"Line Plot of {', '.join(y_cols)} vs {x}"
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = x

    apply_common_layout(fig, target_ax, instance_style, **kwargs)

    call_info = f"sns.lineplot(data=df, x='{x}', y={y!r})"
    return PlotResult(fig=fig, ax=target_ax, call_info=call_info)
