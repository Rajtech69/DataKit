"""Visualization configuration and 4-level precedence resolution engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datakit.config import config


@dataclass
class VisualizationConfig:
    """Dataclass holding visualization parameters for DataKit plots."""

    figsize: tuple[float, float] | None = None
    dpi: int | None = None
    title: str | None = None
    subtitle: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    legend: bool | dict[str, Any] | str | None = None
    grid: bool | dict[str, Any] | None = None
    palette: str | list[str] | None = None
    theme: str | None = None
    tight_layout: bool | None = None
    save: str | None = None
    transparent: bool | None = None


class ConfigResolver:
    """Single source of truth for visualization parameter precedence resolution.

    Precedence order (highest to lowest):
    1. Plot-call keyword argument (e.g. data.plot.hist("age", theme="dark"))
    2. Instance-level style (e.g. data.plot.set_style(theme="dark"))
    3. dk.config global process-wide setting
    4. Library default fallback
    """

    _DEFAULTS: dict[str, Any] = {
        "figsize": (10, 6),
        "dpi": 100,
        "theme": "whitegrid",
        "palette": None,
        "grid": True,
        "tight_layout": True,
        "transparent": False,
        "legend": True,
    }

    @classmethod
    def resolve(
        cls,
        param_name: str,
        call_kwarg: Any = None,
        instance_style: dict[str, Any] | None = None,
        default_fallback: Any = None,
    ) -> Any:
        # 1. Plot call kwarg (if passed and not None)
        if call_kwarg is not None:
            return call_kwarg

        # 2. Instance-level style
        if instance_style and param_name in instance_style and instance_style[param_name] is not None:
            return instance_style[param_name]

        # 3. dk.config global process-wide
        try:
            val = config.get(param_name)
            if val is not None:
                return val
        except Exception:
            pass

        # 4. Class default or explicit fallback
        if default_fallback is not None:
            return default_fallback

        return cls._DEFAULTS.get(param_name, None)
