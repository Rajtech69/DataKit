"""Common OO-style Matplotlib layout helper for DataKit plots.

Strict rule: Never touches global pyplot state. All operations are done directly
on Figure and Axes objects passed into these functions.
"""
from __future__ import annotations

from typing import Any
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure

from datakit.visualization.config import ConfigResolver


def apply_common_layout(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    instance_style: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Apply common layout parameters OO-style on figure and axes.

    Args:
        fig: Matplotlib Figure object.
        ax: Matplotlib Axes object.
        instance_style: Instance-level style dictionary from PlotNamespace.
        **kwargs: Call-level plot keyword arguments.
    """
    title = ConfigResolver.resolve("title", kwargs.get("title"), instance_style)
    subtitle = ConfigResolver.resolve("subtitle", kwargs.get("subtitle"), instance_style)
    xlabel = ConfigResolver.resolve("xlabel", kwargs.get("xlabel"), instance_style)
    ylabel = ConfigResolver.resolve("ylabel", kwargs.get("ylabel"), instance_style)
    xlim = ConfigResolver.resolve("xlim", kwargs.get("xlim"), instance_style)
    ylim = ConfigResolver.resolve("ylim", kwargs.get("ylim"), instance_style)
    grid = ConfigResolver.resolve("grid", kwargs.get("grid"), instance_style)
    legend = ConfigResolver.resolve("legend", kwargs.get("legend"), instance_style)
    tight_layout = ConfigResolver.resolve("tight_layout", kwargs.get("tight_layout"), instance_style)
    save_path = ConfigResolver.resolve("save", kwargs.get("save"), instance_style)
    transparent = ConfigResolver.resolve("transparent", kwargs.get("transparent"), instance_style)

    # 1. Labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # 2. Limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # 3. Title and Subtitle
    if title and subtitle:
        ax.set_title(f"{title}\n{subtitle}")
    elif title:
        ax.set_title(title)
    elif subtitle:
        ax.set_title(subtitle)

    # 4. Grid
    if isinstance(grid, dict):
        ax.grid(True, **grid)
    elif isinstance(grid, bool):
        ax.grid(grid)

    # 5. Legend handling (ergo fixes Seaborn legend pain point)
    if legend is False:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    elif isinstance(legend, dict):
        ax.legend(**legend)
    elif isinstance(legend, str):
        ax.legend(loc=legend)

    # 6. Tight layout
    if tight_layout:
        fig.tight_layout()

    # 7. Save file if requested
    if save_path:
        fig.savefig(save_path, transparent=transparent, bbox_inches="tight")
