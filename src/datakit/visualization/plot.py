"""PlotNamespace class exposing data.plot.* methods.

Every method creates an OO-API plot, never touches global pyplot state, and returns
a PlotResult object with .fig and .ax escape hatches.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from datakit.core.datakit import DataKit
    from datakit.core.results import PlotResult


class PlotNamespace:
    """Visualization namespace bound to a DataKit instance (data.plot.*).

    Examples:
        >>> data = DataKit("insurance.csv")
        >>> data.plot.hist("charges")
        >>> data.plot.scatter("age", "charges", hue="smoker", trend=True)
        >>> data.plot.set_style(figsize=(12, 6), theme="dark")
    """

    def __init__(self, datakit: DataKit) -> None:
        self._datakit = datakit
        self._instance_style: dict[str, Any] = {}

    def set_style(self, **kwargs: Any) -> None:
        """Set instance-level default visualization parameters.

        Examples:
            >>> data.plot.set_style(figsize=(12, 6), dpi=150, palette="viridis")
        """
        self._instance_style.update(kwargs)

    def hist(
        self,
        column: str,
        bins: int | list[float] | None = None,
        kde: bool = False,
        hue: str | None = None,
        stat: str = "count",
        ax: Any = None,
        **kwargs: Any,
    ) -> PlotResult:
        """Plot a histogram of a numeric column."""
        from datakit.visualization.plot_hist import plot_hist

        return plot_hist(
            self._datakit.df,
            column=column,
            bins=bins,
            kde=kde,
            hue=hue,
            stat=stat,
            ax=ax,
            instance_style=self._instance_style,
            **kwargs,
        )

    def box(
        self,
        column: str,
        by: str | None = None,
        orient: str = "v",
        showfliers: bool = True,
        ax: Any = None,
        **kwargs: Any,
    ) -> PlotResult:
        """Plot a box plot of a numeric column, optionally grouped by a category."""
        from datakit.visualization.plot_box import plot_box

        return plot_box(
            self._datakit.df,
            column=column,
            by=by,
            orient=orient,
            showfliers=showfliers,
            ax=ax,
            instance_style=self._instance_style,
            **kwargs,
        )

    def scatter(
        self,
        x: str,
        y: str,
        hue: str | None = None,
        size: str | None = None,
        alpha: float | None = None,
        trend: bool = False,
        ax: Any = None,
        **kwargs: Any,
    ) -> PlotResult:
        """Plot a scatter plot of two numeric columns, with optional trendline."""
        from datakit.visualization.plot_scatter import plot_scatter

        return plot_scatter(
            self._datakit.df,
            x=x,
            y=y,
            hue=hue,
            size=size,
            alpha=alpha,
            trend=trend,
            ax=ax,
            instance_style=self._instance_style,
            **kwargs,
        )

    def bar(
        self,
        x: str,
        y: str | None = None,
        agg: str = "count",
        hue: str | None = None,
        orient: str = "v",
        ax: Any = None,
        **kwargs: Any,
    ) -> PlotResult:
        """Plot a bar chart of categorical frequencies or aggregated numerical values."""
        from datakit.visualization.plot_bar import plot_bar

        return plot_bar(
            self._datakit.df,
            x=x,
            y=y,
            agg=agg,
            hue=hue,
            orient=orient,
            ax=ax,
            instance_style=self._instance_style,
            **kwargs,
        )

    def count(
        self,
        column: str,
        hue: str | None = None,
        order: list[Any] | None = None,
        ax: Any = None,
        **kwargs: Any,
    ) -> PlotResult:
        """Plot category counts for a categorical column."""
        from datakit.visualization.plot_count import plot_count

        return plot_count(
            self._datakit.df,
            column=column,
            hue=hue,
            order=order,
            ax=ax,
            instance_style=self._instance_style,
            **kwargs,
        )

    def line(
        self,
        x: str,
        y: str | list[str],
        hue: str | None = None,
        ax: Any = None,
        **kwargs: Any,
    ) -> PlotResult:
        """Plot a line chart for single or multiple numerical series."""
        from datakit.visualization.plot_line import plot_line

        return plot_line(
            self._datakit.df,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            instance_style=self._instance_style,
            **kwargs,
        )

    def heatmap(
        self,
        columns: list[str] | None = None,
        method: str = "pearson",
        annot: bool = True,
        cmap: str = "coolwarm",
        mask_upper: bool = False,
        ax: Any = None,
        **kwargs: Any,
    ) -> PlotResult:
        """Plot a correlation heatmap of numeric columns."""
        from datakit.visualization.plot_heatmap import plot_heatmap

        return plot_heatmap(
            self._datakit.df,
            columns=columns,
            method=method,
            annot=annot,
            cmap=cmap,
            mask_upper=mask_upper,
            ax=ax,
            instance_style=self._instance_style,
            **kwargs,
        )

    def violin(
        self,
        column: str,
        by: str | None = None,
        split: bool = False,
        ax: Any = None,
        **kwargs: Any,
    ) -> PlotResult:
        """Plot a violin plot of a numeric column, optionally grouped by a category."""
        from datakit.visualization.plot_violin import plot_violin

        return plot_violin(
            self._datakit.df,
            column=column,
            by=by,
            split=split,
            ax=ax,
            instance_style=self._instance_style,
            **kwargs,
        )

    def pairplot(
        self,
        columns: list[str] | None = None,
        hue: str | None = None,
        kind: str = "scatter",
        diag_kind: str = "auto",
        **kwargs: Any,
    ) -> PlotResult:
        """Plot pairwise relationships across numeric columns."""
        from datakit.visualization.plot_pairplot import plot_pairplot

        return plot_pairplot(
            self._datakit.df,
            columns=columns,
            hue=hue,
            kind=kind,
            diag_kind=diag_kind,
            instance_style=self._instance_style,
            **kwargs,
        )
