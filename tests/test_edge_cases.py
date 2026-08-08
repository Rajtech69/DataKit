"""Tests for boundary conditions, edge cases, empty datasets, and unusual structures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from datakit import DataKit
from datakit.core.exceptions import EmptyDataError, InsufficientDataError


class TestEdgeCases:
    def test_single_row_dataframe(self, df_single_row):
        dk = DataKit(df_single_row)
        res_inspect = dk.inspect()
        assert res_inspect.shape == (1, 2)

        res_audit = dk.audit()
        assert isinstance(res_audit.summary, str)

    def test_single_column_dataframe(self):
        df = pd.DataFrame({"single_num": [10.0, 20.0, 30.0, 40.0, 50.0]})
        dk = DataKit(df)
        res_dist = dk.distributions()
        assert "single_num" in res_dist.stats.index

        # Relationships requires at least 2 numeric columns
        with pytest.raises(InsufficientDataError):
            dk.relationships()

    def test_all_null_column(self, df_all_null):
        dk = DataKit(df_all_null)
        res_audit = dk.audit()
        missing_issue = [i for i in res_audit.issues if i.column == "b"]
        assert len(missing_issue) > 0
        assert missing_issue[0].severity == "critical"

    def test_multi_index_dataframe(self):
        tuples = [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
        idx = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = pd.DataFrame({"val": [10, 20, 30, 40]}, index=idx)
        dk = DataKit(df)

        res_inspect = dk.inspect()
        assert res_inspect.index_info["type"] == "MultiIndex"
        res_audit = dk.audit()
        assert res_audit is not None
