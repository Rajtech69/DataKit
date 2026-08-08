"""Tests for data.eda() and data.visualize() methods."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from datakit import DataKit
from datakit.core.exceptions import ColumnNotFoundError
from datakit.core.results import EDAResult, PlotResult


class TestEDA:
    def test_eda_full_pipeline(self, insurance_datakit):
        res = insurance_datakit.eda(plots=True)
        assert isinstance(res, EDAResult)
        assert res.inspect is not None
        assert res.audit is not None
        assert res.distributions is not None
        assert res.outliers is not None
        assert res.relationships is not None
        assert isinstance(res.figures, list)
        assert len(res.figures) > 0
        # Clean up figure objects
        for fig_res in res.figures:
            plt.close(fig_res.fig)

    def test_eda_include_exclude_filtering(self, sample_datakit):
        res = sample_datakit.eda(include=["inspect", "audit"], plots=False)
        assert res.inspect is not None
        assert res.audit is not None
        assert res.distributions is None
        assert res.outliers is None
        assert res.relationships is None
        assert len(res.figures) == 0

    def test_eda_target_bias(self, insurance_datakit):
        res = insurance_datakit.eda(target="charges", plots=True)
        assert isinstance(res, EDAResult)
        for fig_res in res.figures:
            plt.close(fig_res.fig)

    def test_eda_sampling(self, insurance_datakit):
        res = insurance_datakit.eda(sample=10, plots=False)
        assert res.inspect.shape[0] == 10

    def test_eda_invalid_target_raises(self, sample_datakit):
        with pytest.raises(ColumnNotFoundError):
            sample_datakit.eda(target="nonexistent_target")

    def test_eda_summary(self, insurance_datakit):
        res = insurance_datakit.eda(plots=False)
        summary = res.summary()
        assert "EDA Synthesis Report" in summary
        assert "Shape:" in summary

    def test_visualize(self, sample_datakit):
        figs = sample_datakit.visualize()
        assert isinstance(figs, list)
        assert len(figs) > 0
        for fig_res in figs:
            plt.close(fig_res.fig)
