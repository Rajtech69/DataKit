"""Tests for data.clean() method and CleanReport."""
import pandas as pd
import pytest

from datakit import DataKit
from datakit.core.exceptions import ConfirmationRequiredError
from datakit.core.results import AuditResult, CleanReport
from datakit.core.warnings import DataLossWarning


class TestCleanAcceptanceCriteria:
    def test_clean_section_29_acceptance(self, df_with_missing):
        """PRD §29 Acceptance Criteria for data.clean():

        Given a DataFrame with missing values, when clean(missing="impute_median") is called
        without confirm=True, then ConfirmationRequiredError is raised and the original DataFrame
        is provably unmutated. When called with confirm=True, then a new DataKit is returned,
        the original instance's underlying DataFrame is unchanged, and
        .last_clean_report.values_imputed accurately counts every value changed.
        """
        dk_orig = DataKit(df_with_missing)
        orig_copy = df_with_missing.copy()

        # 1. Unconfirmed destructive call raises ConfirmationRequiredError
        with pytest.raises(ConfirmationRequiredError, match="confirm=True"):
            dk_orig.clean(missing="impute_median")

        # Verify original DataFrame is provably unmutated
        pd.testing.assert_frame_equal(dk_orig.df, orig_copy)

        # 2. Confirmed destructive call succeeds
        dk_cleaned = dk_orig.clean(missing="impute_median", confirm=True)

        assert isinstance(dk_cleaned, DataKit)
        assert dk_cleaned is not dk_orig
        # Original remains unchanged
        pd.testing.assert_frame_equal(dk_orig.df, orig_copy)
        # Cleaned df has no missing values in numeric columns where median could be calculated
        assert dk_cleaned.df["col_10_missing"].isnull().sum() == 0
        assert dk_cleaned.df["col_40_missing"].isnull().sum() == 0

        # Check last_clean_report
        report = dk_cleaned.last_clean_report
        assert isinstance(report, CleanReport)
        assert report.values_imputed["col_10_missing"] == 1
        assert report.values_imputed["col_40_missing"] == 4


class TestCleanStrategies:
    def test_clean_report_only_default(self, df_with_missing):
        dk = DataKit(df_with_missing)
        # Default report-only mode does not require confirm=True
        cleaned = dk.clean()
        assert isinstance(cleaned, DataKit)
        assert cleaned.last_clean_report.rows_dropped == 0
        assert len(cleaned.last_clean_report.values_imputed) == 0

    def test_clean_drop_missing(self, df_with_missing):
        dk = DataKit(df_with_missing)
        with pytest.warns(DataLossWarning):
            cleaned = dk.clean(missing="drop", confirm=True)
        # col_90_missing has 9 nulls out of 10 rows
        assert len(cleaned.df) == 1
        assert cleaned.last_clean_report.rows_dropped == 9

    def test_clean_drop_duplicates(self, df_with_duplicates):
        dk = DataKit(df_with_duplicates)
        cleaned = dk.clean(duplicates="drop", confirm=True)
        assert len(cleaned.df) == 4  # 6 - 2 duplicates = 4
        assert cleaned.last_clean_report.rows_dropped == 2

    def test_clean_impute_mean(self, df_with_missing):
        dk = DataKit(df_with_missing)
        cleaned = dk.clean(missing="impute_mean", confirm=True)
        assert cleaned.df["col_10_missing"].isnull().sum() == 0

    def test_clean_impute_mode(self):
        df = pd.DataFrame({"cat": ["a", "a", "b", None]})
        dk = DataKit(df)
        cleaned = dk.clean(missing="impute_mode", confirm=True)
        assert cleaned.df["cat"].iloc[3] == "a"
        assert cleaned.last_clean_report.values_imputed["cat"] == 1

    def test_clean_coerce_dtypes(self, df_mixed_types):
        dk = DataKit(df_mixed_types)
        cleaned = dk.clean(dtypes="coerce", confirm=True)
        assert pd.api.types.is_numeric_dtype(cleaned.df["numeric_as_string"].dtype)
        assert len(cleaned.last_clean_report.dtype_changes) >= 1

    def test_clean_chaining_with_audit(self, df_with_missing):
        dk = DataKit(df_with_missing)
        audit_res = dk.clean(missing="impute_median", confirm=True).audit()
        assert isinstance(audit_res, AuditResult)

    def test_diff_summary(self, df_with_missing):
        dk = DataKit(df_with_missing)
        cleaned = dk.clean(missing="impute_median", confirm=True)
        diff = cleaned.last_clean_report.diff_summary()
        assert "Values imputed:" in diff
        assert "col_10_missing" in diff
