"""Tests for data.memory() method and MemoryResult."""
import pandas as pd
import pytest

from datakit import DataKit
from datakit.core.results import MemoryResult


class TestMemory:
    def test_memory_profiling(self, sample_datakit):
        res = sample_datakit.memory()
        assert isinstance(res, MemoryResult)
        assert res.total_mb > 0
        assert "age" in res.by_column
        assert isinstance(res.suggestions, list)

    def test_memory_suggestions(self):
        df = pd.DataFrame({
            "big_int": pd.Series(range(30), dtype="int64"),
            "low_card_str": ["a", "b", "a"] * 10,
        })
        dk = DataKit(df)
        res = dk.memory()
        assert len(res.suggestions) > 0
        summary = res.summary()
        assert "Total Memory Usage" in summary
