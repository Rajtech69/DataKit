"""Tests for data.relationships() method and RelationshipResult."""
import pandas as pd
import pytest

from datakit import DataKit
from datakit.core.exceptions import InsufficientDataError
from datakit.core.results import RelationshipResult


class TestRelationships:
    def test_relationships_basic(self, sample_datakit):
        res = sample_datakit.relationships()
        assert isinstance(res, RelationshipResult)
        assert res.method == "pearson"
        assert isinstance(res.matrix, pd.DataFrame)
        assert isinstance(res.strong_pairs, list)

    def test_relationships_spearman(self, insurance_datakit):
        res = insurance_datakit.relationships(method="spearman", threshold=0.3)
        assert res.method == "spearman"
        assert len(res.strong_pairs) > 0
        for col1, col2, corr in res.strong_pairs:
            assert abs(corr) >= 0.3
            assert col1 != col2

    def test_relationships_insufficient_data(self):
        df_single_num = pd.DataFrame({"a": [1, 2, 3], "cat": ["x", "y", "z"]})
        dk = DataKit(df_single_num)
        with pytest.raises(InsufficientDataError):
            dk.relationships()

    def test_relationships_summary(self, sample_datakit):
        res = sample_datakit.relationships()
        summary = res.summary()
        assert "Relationship Summary" in summary or "matrix" in repr(res).lower() or len(summary) > 0
