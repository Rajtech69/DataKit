"""Heatmap plot module for DataKit.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datakit.core.exceptions import ColumnNotFoundError, InsufficientDataError
from datakit.core.results import PlotResult
from datakit.visualization.common import apply_common_layout
from datakit.visualization.config import ConfigResolver


def plot_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    annot: bool = True,
    cmap: str = "coolwarm",
    mask_upper: bool = False,
    ax: Any = None,
    instance_style: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PlotResult:
    numeric_df = df.select_dtypes(include=[np.number])

    if columns is not None:
        for col in columns:
            if col not in df.columns:
                raise ColumnNotFoundError(col, list(df.columns))
        numeric_df = numeric_df[[c for c in columns if c in numeric_df.columns]]

    if numeric_df.shape[1] < 2:
        raise InsufficientDataError("Heatmap requires at least 2 numeric columns.")

    corr_matrix = numeric_df.corr(method=method)

    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    figsize = ConfigResolver.resolve("figsize", kwargs.get("figsize"), instance_style)
    dpi = ConfigResolver.resolve("dpi", kwargs.get("dpi"), instance_style)

    if ax is None:
        fig, target_ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        target_ax = ax
        fig = target_ax.get_figure()

    sns.heatmap(
        data=corr_matrix,
        annot=annot,
        cmap=cmap,
        mask=mask,
        vmin=-1.0,
        vmax=1.0,
        fmt=".2f",
        ax=target_ax,
    )

    if "title" not in kwargs and "title" not in (instance_style or {}):
        kwargs["title"] = f"Correlation Heatmap ({method.capitalize()})"

    apply_common_layout(fig, target_ax, instance_style, **kwargs)

    call_info = f"sns.heatmap(data=corr_matrix, method={method!r}, annot={annot})"
    return PlotResult(fig=fig, ax=target_ax, call_info=call_info)
