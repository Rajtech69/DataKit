"""Tests for shared deterministic inference engine."""
import numpy as np
import pandas as pd
import pytest

from datakit.core.infer import (
    infer_column_types,
    infer_id_like_columns,
    infer_suspicious_dtypes,
    infer_target_candidates,
)


class TestInferColumnTypes:
    def test_infer_types_basic(self, sample_df):
        types = infer_column_types(sample_df)
        assert types["age"] == "numeric"
        assert types["salary"] == "numeric"
        assert types["department"] == "categorical"
        assert types["active"] == "boolean"

    def test_infer_datetime_type(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=5),
                "val": [1, 2, 3, 4, 5],
            }
        )
        types = infer_column_types(df)
        assert types["date"] == "datetime"
        assert types["val"] == "numeric"


class TestInferIdLikeColumns:
    def test_infer_id_column(self, df_with_id_column):
        id_cols = infer_id_like_columns(df_with_id_column)
        assert "user_id" in id_cols

    def test_non_id_columns_not_flagged(self, sample_df):
        id_cols = infer_id_like_columns(sample_df)
        assert len(id_cols) == 0


class TestInferTargetCandidates:
    def test_infer_target_by_keyword(self, insurance_datakit):
        candidates = infer_target_candidates(insurance_datakit.df)
        assert "charges" in candidates

    def test_infer_target_by_position(self):
        df = pd.DataFrame(
            {
                "id": range(10),
                "feat1": range(10),
                "outcome_val": range(10),
            }
        )
        candidates = infer_target_candidates(df)
        assert "outcome_val" in candidates


class TestInferSuspiciousDtypes:
    def test_detect_numeric_as_string(self, df_mixed_types):
        suspicious = infer_suspicious_dtypes(df_mixed_types)
        cols = [col for col, _ in suspicious]
        assert "numeric_as_string" in cols
        assert "real_numeric" not in cols


class TestCompare:
    def test_compare_datasets(self, sample_datakit):
        df_mod = sample_datakit.df.copy()
        df_mod["new_col"] = 1.0
        df_mod = df_mod.drop(columns=["age"])

        res = sample_datakit.compare(df_mod)
        assert res.shape_a == sample_datakit.df.shape
        assert "new_col" in res.added_columns
        assert "age" in res.removed_columns
        assert "=== Dataset Comparison Report ===" in res.summary()
