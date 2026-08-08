"""Tests for built-in dataset loader module."""
import pytest

import datakit as dk
from datakit.core.datakit import DataKit


class TestDatasets:
    def test_list_datasets(self):
        datasets = dk.list_datasets()
        assert isinstance(datasets, list)
        assert "insurance" in datasets
        assert "housing" in datasets
        assert "churn" in datasets

    def test_load_insurance_dataset(self):
        data = dk.load_dataset("insurance")
        assert isinstance(data, DataKit)
        assert data.df.shape[0] > 0
        assert "charges" in data.df.columns

    def test_load_housing_dataset(self):
        data = dk.load_dataset("housing")
        assert isinstance(data, DataKit)
        assert data.df.shape[0] > 0
        assert "price" in data.df.columns

    def test_load_churn_dataset(self):
        data = dk.load_dataset("churn")
        assert isinstance(data, DataKit)
        assert data.df.shape[0] > 0
        assert "churn" in data.df.columns

    def test_load_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Dataset 'insuranc' not found. Did you mean 'insurance'?"):
            dk.load_dataset("insuranc")
