"""Tests for data.plot.* visualization module, config precedence, and escape hatches."""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import datakit as dk
from datakit import DataKit
from datakit.core.exceptions import ColumnNotFoundError, IncompatibleColumnTypeError
from datakit.core.results import PlotResult
from datakit.visualization.config import ConfigResolver


class TestPlotTypes:
    def test_plot_hist_basic(self, insurance_datakit):
        res = insurance_datakit.plot.hist("charges")
        assert isinstance(res, PlotResult)
        assert res.fig is not None
        assert res.ax is not None
        assert len(res.axes) == 1
        plt.close(res.fig)

    def test_plot_box_basic(self, insurance_datakit):
        res = insurance_datakit.plot.box("charges", by="smoker")
        assert isinstance(res, PlotResult)
        plt.close(res.fig)

    def test_plot_scatter_with_trend(self, insurance_datakit):
        res = insurance_datakit.plot.scatter("age", "charges", hue="smoker", trend=True)
        assert isinstance(res, PlotResult)
        # Verify trendline was drawn (lines on ax > 0)
        assert len(res.ax.lines) > 0
        plt.close(res.fig)

    def test_plot_bar_count_agg(self, insurance_datakit):
        res = insurance_datakit.plot.bar("region")
        assert isinstance(res, PlotResult)
        plt.close(res.fig)

    def test_plot_bar_mean_agg(self, insurance_datakit):
        res = insurance_datakit.plot.bar("region", y="charges", agg="mean")
        assert isinstance(res, PlotResult)
        plt.close(res.fig)

    def test_plot_count_basic(self, insurance_datakit):
        res = insurance_datakit.plot.count("sex")
        assert isinstance(res, PlotResult)
        plt.close(res.fig)

    def test_plot_line_single_and_multi(self):
        df = pd.DataFrame({"x": range(5), "y1": [1, 2, 3, 4, 5], "y2": [5, 4, 3, 2, 1]})
        datakit = DataKit(df)

        res1 = datakit.plot.line("x", "y1")
        assert isinstance(res1, PlotResult)
        plt.close(res1.fig)

        res2 = datakit.plot.line("x", ["y1", "y2"])
        assert isinstance(res2, PlotResult)
        plt.close(res2.fig)

    def test_plot_heatmap_basic(self, insurance_datakit):
        res = insurance_datakit.plot.heatmap()
        assert isinstance(res, PlotResult)
        plt.close(res.fig)


class TestEscapeHatches:
    def test_escape_hatch_level_1_config_level(self, insurance_datakit):
        """Tier 1: Config-level styling kwargs passed directly to plot."""
        res = insurance_datakit.plot.hist("charges", title="Custom Title", xlabel="Cost ($)")
        assert res.ax.get_title() == "Custom Title"
        assert res.ax.get_xlabel() == "Cost ($)"
        plt.close(res.fig)

    def test_escape_hatch_level_2_object_level(self, insurance_datakit):
        """Tier 2: Direct post-hoc manipulation of returned .fig and .ax."""
        res = insurance_datakit.plot.hist("charges")
        res.ax.set_title("Post-hoc Title")
        assert res.ax.get_title() == "Post-hoc Title"
        plt.close(res.fig)

    def test_escape_hatch_level_3_injection_level(self, insurance_datakit):
        """Tier 3: Injection of pre-existing Matplotlib Axes without creating a new Figure."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        orig_fig_num = fig.number

        res1 = insurance_datakit.plot.hist("charges", ax=axes[0])
        res2 = insurance_datakit.plot.box("charges", by="smoker", ax=axes[1])

        assert res1.ax is axes[0]
        assert res2.ax is axes[1]
        assert res1.fig.number == orig_fig_num
        assert res2.fig.number == orig_fig_num
        plt.close(fig)


class TestConfigPrecedence:
    def test_four_level_precedence(self, insurance_datakit):
        """Test PRD §19 4-level precedence resolution."""
        # Level 4: Default fallback
        val_default = ConfigResolver.resolve("dpi")
        assert val_default == 100

        # Level 3: Global config
        try:
            dk.config.set(dpi=150)
            val_global = ConfigResolver.resolve("dpi")
            assert val_global == 150

            # Level 2: Instance-level style
            insurance_datakit.plot.set_style(dpi=200)
            val_instance = ConfigResolver.resolve(
                "dpi",
                instance_style=insurance_datakit.plot._instance_style,
            )
            assert val_instance == 200

            # Level 1: Plot-call kwarg
            val_kwarg = ConfigResolver.resolve(
                "dpi",
                call_kwarg=300,
                instance_style=insurance_datakit.plot._instance_style,
            )
            assert val_kwarg == 300

        finally:
            dk.config.reset()
