"""Tests for data.inspect() method and InspectResult."""
import pandas as pd
import pytest

from datakit import DataKit, config
from datakit.core.exceptions import EmptyDataError
from datakit.core.results import InspectResult
from datakit.core.warnings import LargeDatasetWarning


class TestInspect:
    def test_inspect_returns_inspect_result(self, sample_datakit):
        res = sample_datakit.inspect()
        assert isinstance(res, InspectResult)
        assert res.shape == (5, 4)
        assert res.memory_mb > 0
        assert len(res.head) == 5
        assert len(res.tail) == 5
        assert res.index_info["type"] == "RangeIndex"

    def test_inspect_custom_n(self, insurance_datakit):
        res = insurance_datakit.inspect(n=3)
        assert len(res.head) == 3
        assert len(res.tail) == 3

    def test_inspect_memory_false(self, sample_datakit):
        res = sample_datakit.inspect(memory=False)
        assert res.memory_mb == 0.0

    def test_inspect_empty_dataframe_raises(self):
        empty_dk = DataKit(pd.DataFrame())
        with pytest.raises(EmptyDataError):
            empty_dk.inspect()

    def test_inspect_large_dataset_warning(self, sample_datakit):
        try:
            config.set(large_dataset_mb=0.00001)  # trigger warning on small df
            with pytest.warns(LargeDatasetWarning, match="Dataset memory size"):
                sample_datakit.inspect(memory=True)
        finally:
            config.reset()

    def test_inspect_result_to_dict(self, sample_datakit):
        res = sample_datakit.inspect()
        d = res.to_dict()
        assert "shape" in d
        assert d["shape"] == (5, 4)
        assert "head" in d
        assert "tail" in d

    def test_inspect_result_repr_html(self, sample_datakit):
        res = sample_datakit.inspect()
        html = res._repr_html_()
        assert "InspectResult" in html
        assert "Shape:" in html
