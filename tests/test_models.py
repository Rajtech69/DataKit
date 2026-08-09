"""Tests for machine learning models module src/datakit/ml/models.py."""
import pandas as pd
import pytest

from datakit import DataKit
from datakit.core.results import ModelResult, PlotResult


class TestFitModel:
    def test_fit_random_forest_classification(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="smoker", model="rf", random_state=42)

        assert isinstance(res, ModelResult)
        assert res.task == "classification"
        assert "accuracy" in res.metrics
        assert "f1" in res.metrics
        assert res.feature_importances is not None
        assert "=== DataKit Model Training Report" in res.summary()

        plot_imp = res.plot_importance()
        assert isinstance(plot_imp, PlotResult)

        plot_eval = res.plot_evaluation()
        assert isinstance(plot_eval, PlotResult)

        preds = res.predict(res.X_test)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(res.y_test)

    def test_fit_random_forest_regression(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="charges", model="rf", task="regression", random_state=42)

        assert isinstance(res, ModelResult)
        assert res.task == "regression"
        assert "r2" in res.metrics
        assert "mae" in res.metrics
        assert "rmse" in res.metrics

        plot_eval = res.plot_evaluation()
        assert isinstance(plot_eval, PlotResult)

    def test_fit_logistic_regression(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="smoker", model="logistic", random_state=42)

        assert isinstance(res, ModelResult)
        assert "accuracy" in res.metrics

    def test_fit_linear_regression(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="charges", model="linear", task="regression")

        assert isinstance(res, ModelResult)
        assert "r2" in res.metrics

    def test_fit_knn(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        res = cleaned.fit(target="smoker", model="knn", n_neighbors=3)

        assert isinstance(res, ModelResult)
        assert "accuracy" in res.metrics

    def test_fit_unsupported_model_raises(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        with pytest.raises(ValueError, match="Unsupported classification model"):
            cleaned.fit(target="smoker", model="nonexistent_algorithm")

    def test_fit_small_dataset_auto_task(self):
        data = DataKit({
            "age": [19, 28, 33, 32, 31, 46, 54, 37],
            "sex": ["female", "male", "male", "male", "female", "female", "female", "male"],
            "bmi": [27.9, 33.77, 22.7, 28.88, 25.74, 33.44, 30.8, 27.74],
            "smoker": ["yes", "no", "no", "no", "no", "no", "no", "no"],
            "charges": [16884.92, 1725.55, 4449.46, 21984.47, 3866.86, 8240.58, 11299.34, 7281.50]
        })
        res = data.fit(target="charges", model="linear")
        assert isinstance(res, ModelResult)
        assert res.task == "regression"
        assert res.model_name == "LinearRegression"
        assert "r2" in res.metrics
