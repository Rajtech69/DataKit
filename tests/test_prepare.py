"""Tests for data.prepare() method and PrepareResult."""
import pandas as pd
import pytest

from datakit import DataKit
from datakit.core.exceptions import ColumnNotFoundError, DataKitError
from datakit.core.results import PrepareResult
from datakit.core.warnings import PotentialLeakageWarning


class TestPrepare:
    def test_prepare_section_29_acceptance(self, insurance_datakit):
        """PRD §29 Acceptance Criteria for data.prepare():

        Returns a PrepareResult holding X_train, X_test, y_train, y_test,
        a real sklearn Pipeline instance, and verified index alignment.
        """
        # First clean missing values so sklearn pipeline can scale/transform
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.prepare(target="charges", random_state=42)

        assert isinstance(res, PrepareResult)
        assert isinstance(res.X_train, pd.DataFrame)
        assert isinstance(res.X_test, pd.DataFrame)
        assert isinstance(res.y_train, pd.Series)
        assert isinstance(res.y_test, pd.Series)

        # Real sklearn Pipeline verification
        from sklearn.pipeline import Pipeline
        assert isinstance(res.preprocessing_pipeline, Pipeline)

        # Index alignment assertion: X_train.index equals y_train.index
        pd.testing.assert_index_equal(res.X_train.index, res.y_train.index)
        pd.testing.assert_index_equal(res.X_test.index, res.y_test.index)

    def test_prepare_ordinal_encoding(self, sample_datakit):
        res = sample_datakit.prepare(target="salary", encode="ordinal", random_state=42)
        assert isinstance(res, PrepareResult)
        assert res.X_train.shape[0] > 0

    def test_prepare_missing_target_raises(self, sample_datakit):
        with pytest.raises(ColumnNotFoundError):
            sample_datakit.prepare(target="nonexistent_col")

    def test_prepare_leakage_warning_and_strict(self):
        df = pd.DataFrame({
            "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "leaked_feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "normal_feature": [5, 2, 8, 1, 9, 3, 7, 4, 6, 0],
        })
        dk = DataKit(df)

        with pytest.warns(PotentialLeakageWarning, match="potential target leakage"):
            dk.prepare(target="target", strict_leakage=False)

        with pytest.raises(DataKitError, match="potential target leakage"):
            dk.prepare(target="target", strict_leakage=True)
