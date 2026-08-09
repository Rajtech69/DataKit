"""Tests for data.audit() and AuditResult."""
import pandas as pd
import pytest

from datakit import DataKit
from datakit.core.results import AuditResult, Issue
from datakit.core.warnings import ConstantColumnWarning, HighCardinalityWarning


class TestAuditAcceptanceCriteria:
    def test_audit_section_29_acceptance(self, insurance_datakit):
        """PRD §29 Acceptance Criteria for data.audit():

        Given a DataFrame with missing values, duplicate rows, and at least one constant column,
        when audit() is called, then .issues contains an entry for each of: missing-value
        columns above 0%, duplicate-row count, and constant column — each with non-empty
        .recommendation. Calling audit() twice returns identical results.
        """
        # Create a df with missing values, duplicates, and a constant column
        df = insurance_datakit.df.copy()
        df["constant_col"] = "same_value"
        dk = DataKit(df)

        res1 = dk.audit()

        assert isinstance(res1, AuditResult)
        assert len(res1.issues) > 0

        # Check missing values present
        missing_issues = [i for i in res1.issues if "missing values" in i.message]
        assert len(missing_issues) > 0

        # Check duplicate rows present
        dup_issues = [i for i in res1.issues if "duplicate" in i.message]
        assert len(dup_issues) > 0

        # Check constant column present
        const_issues = [i for i in res1.issues if i.column == "constant_col"]
        assert len(const_issues) > 0

        # Verify all recommendations non-empty
        for issue in res1.issues:
            assert issue.recommendation is not None
            assert len(issue.recommendation.strip()) > 0

        # Determinism check: calling twice returns identical results
        res2 = dk.audit()
        assert res1.summary == res2.summary
        assert len(res1.issues) == len(res2.issues)
        for i1, i2 in zip(res1.issues, res2.issues):
            assert i1.column == i2.column
            assert i1.severity == i2.severity
            assert i1.message == i2.message
            assert i1.recommendation == i2.recommendation


class TestAuditChecks:
    def test_audit_missing_check(self, df_with_missing):
        dk = DataKit(df_with_missing)
        res = dk.audit(checks=["missing"])
        cols_flagged = [i.column for i in res.issues]
        assert "col_10_missing" in cols_flagged
        assert "col_40_missing" in cols_flagged
        assert "col_90_missing" in cols_flagged
        assert "col_0_missing" not in cols_flagged

    def test_audit_duplicates_check(self, df_with_duplicates):
        dk = DataKit(df_with_duplicates)
        res = dk.audit(checks=["duplicates"])
        assert len(res.issues) == 1
        assert "duplicate" in res.issues[0].message

    def test_audit_constant_check(self, df_with_constant_column):
        dk = DataKit(df_with_constant_column)
        with pytest.warns(ConstantColumnWarning):
            res = dk.audit(checks=["constant"])
        cols = [i.column for i in res.issues]
        assert "constant" in cols

    def test_audit_suspicious_dtype(self, df_mixed_types):
        dk = DataKit(df_mixed_types)
        res = dk.audit(checks=["suspicious_dtype"])
        cols = [i.column for i in res.issues]
        assert "numeric_as_string" in cols

    def test_audit_id_like_check(self, df_with_id_column):
        dk = DataKit(df_with_id_column)
        res = dk.audit(checks=["id_like"])
        cols = [i.column for i in res.issues]
        assert "user_id" in cols

    def test_audit_unknown_check_raises(self, sample_datakit):
        with pytest.raises(ValueError, match="Unknown check"):
            sample_datakit.audit(checks=["nonexistent_check"])


class TestAuditResultProperties:
    def test_severity_filtering(self, df_with_missing):
        dk = DataKit(df_with_missing)
        # col_90_missing is critical, col_40_missing is critical (>=40%), col_10_missing is warning (>=10%)
        res_info = dk.audit(severity_threshold="info")
        res_warning = dk.audit(severity_threshold="warning")
        res_critical = dk.audit(severity_threshold="critical")

        assert len(res_info.issues) >= len(res_warning.issues)
        assert len(res_warning.issues) >= len(res_critical.issues)

    def test_audit_result_properties(self, df_with_missing):
        dk = DataKit(df_with_missing)
        res = dk.audit()
        assert isinstance(res.warnings, list)
        assert isinstance(res.critical, list)
        assert isinstance(res.recommendations, list)

    def test_to_dataframe(self, sample_datakit):
        res = sample_datakit.audit()
        df_res = res.to_dataframe()
        assert isinstance(df_res, pd.DataFrame)
        assert list(df_res.columns) == ["column", "severity", "message", "recommendation"]

    def test_repr_html(self, sample_datakit):
        res = sample_datakit.audit()
        html = res._repr_html_()
        assert "AuditResult" in html

    def test_duplicates_view(self, insurance_datakit):
        dups = insurance_datakit.duplicates()
        assert isinstance(dups, pd.DataFrame)
        assert len(dups) > 0
