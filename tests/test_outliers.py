"""Tests for data.outliers() method and OutlierResult."""
import pandas as pd
import pytest

from datakit import DataKit
from datakit.core.exceptions import IncompatibleColumnTypeError
from datakit.core.results import OutlierResult


class TestOutliers:
    def test_outliers_iqr_default(self, insurance_datakit):
        res = insurance_datakit.outliers()
        assert isinstance(res, OutlierResult)
        assert res.method == "iqr"
        assert res.threshold == 1.5
        assert "charges" in res.counts
        assert res.counts["charges"] >= 1  # 65000.0 is outlier in charges

    def test_outliers_zscore(self, insurance_datakit):
        res = insurance_datakit.outliers(method="zscore", threshold=2.0)
        assert res.method == "zscore"
        assert res.threshold == 2.0
        assert "charges" in res.counts

    def test_outliers_to_dataframe(self, insurance_datakit):
        res = insurance_datakit.outliers(columns=["charges"])
        df_flagged = res.to_dataframe(insurance_datakit.df, flag_column="_is_outlier")

        assert isinstance(df_flagged, pd.DataFrame)
        assert "_is_outlier" in df_flagged.columns
        assert df_flagged["_is_outlier"].sum() > 0
        # Original DataFrame not mutated
        assert "_is_outlier" not in insurance_datakit.df.columns

    def test_outliers_invalid_column_raises(self, insurance_datakit):
        with pytest.raises(IncompatibleColumnTypeError):
            insurance_datakit.outliers(columns=["sex"])  # String column

    def test_outliers_summary(self, insurance_datakit):
        res = insurance_datakit.outliers()
        summary = res.summary()
        assert "Outlier Detection Summary" in summary
        assert "charges" in summary
