"""Tests for data.distributions() method and DistributionResult."""
import pandas as pd
import pytest

from datakit import DataKit
from datakit.core.exceptions import IncompatibleColumnTypeError
from datakit.core.results import DistributionResult


class TestDistributions:
    def test_distributions_basic(self, sample_datakit):
        res = sample_datakit.distributions()
        assert isinstance(res, DistributionResult)
        assert isinstance(res.stats, pd.DataFrame)
        assert list(res.stats.columns) == ["skew", "kurtosis", "likely_normal"]
        assert "age" in res.stats.index
        assert "salary" in res.stats.index

    def test_distributions_specific_columns(self, insurance_datakit):
        res = insurance_datakit.distributions(columns=["age", "bmi"])
        assert len(res.stats) == 2
        assert "age" in res.stats.index
        assert "bmi" in res.stats.index

    def test_distributions_invalid_column(self, insurance_datakit):
        with pytest.raises(IncompatibleColumnTypeError):
            insurance_datakit.distributions(columns=["sex"])

    def test_distributions_summary(self, sample_datakit):
        res = sample_datakit.distributions()
        summary = res.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
