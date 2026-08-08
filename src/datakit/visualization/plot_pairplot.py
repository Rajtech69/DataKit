"""Pairplot module for DataKit.
"""
from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datakit.config import config
from datakit.core.exceptions import ColumnNotFoundError, InsufficientDataError
from datakit.core.results import PlotResult
from datakit.core.warnings import LargeDatasetWarning
from datakit.visualization.config import ConfigResolver


def plot_pairplot(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    hue: str | None = None,
    kind: str = "scatter",
    diag_kind: str = "auto",
    instance_style: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PlotResult:
    numeric_df = df.select_dtypes(include=[np.number])

    if columns is not None:
        for col in columns:
            if col not in df.columns:
                raise ColumnNotFoundError(col, list(df.columns))
        numeric_df = numeric_df[[c for c in columns if c in numeric_df.columns]]

    max_cols = config.get("pairplot_max_columns")
    if numeric_df.shape[1] > max_cols:
        warnings.warn(
            f"Pairplot requested for {numeric_df.shape[1]} columns, which exceeds pairplot_max_columns ({max_cols}). "
            f"Truncating to first {max_cols} numeric columns to prevent performance freeze.",
            LargeDatasetWarning,
            stacklevel=2,
        )
        numeric_df = numeric_df.iloc[:, :max_cols]

    if numeric_df.shape[1] < 2:
        raise InsufficientDataError("Pairplot requires at least 2 numeric columns.")

    plot_df = numeric_df.copy()
    if hue and hue in df.columns and hue not in plot_df.columns:
        plot_df[hue] = df[hue]

    g = sns.pairplot(
        data=plot_df,
        hue=hue,
        kind=kind,
        diag_kind=diag_kind,
    )

    call_info = f"sns.pairplot(data=df, columns={list(numeric_df.columns)}, hue={hue!r})"
    return PlotResult(fig=g.fig, ax=g.axes, call_info=call_info)
