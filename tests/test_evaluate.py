"""Tests for model evaluation module src/datakit/ml/evaluate.py."""
import pandas as pd
import pytest

import datakit as dk
from datakit.core.results import EvaluationResult, PlotResult


class DummyClassifier:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return [0 if float(x) < 30 else 1 for x in X.iloc[:, 0]]


class DummyRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return [float(x) * 100.0 for x in X.iloc[:, 0]]


class TestEvaluate:
    def test_evaluate_classification(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        # Create binary target
        cleaned.df["is_smoker"] = (cleaned.df["smoker"] == "yes").astype(int)
        ml_data = cleaned.prepare(target="is_smoker", task="classification", random_state=42)

        clf = DummyClassifier().fit(ml_data.X_train, ml_data.y_train)
        res = cleaned.evaluate(clf, ml_data.X_test, ml_data.y_test, task="classification")

        assert isinstance(res, EvaluationResult)
        assert "accuracy" in res.metrics
        assert "f1" in res.metrics
        assert res.confusion_matrix is not None
        assert "=== Model Evaluation Report" in res.summary()

        plot_res = res.plot_confusion_matrix()
        assert isinstance(plot_res, PlotResult)

    def test_evaluate_regression(self, insurance_datakit):
        cleaned = insurance_datakit.clean(missing="impute_median", confirm=True)
        ml_data = cleaned.prepare(target="charges", task="regression", random_state=42)

        reg = DummyRegressor().fit(ml_data.X_train, ml_data.y_train)
        res = cleaned.evaluate(reg, ml_data.X_test, ml_data.y_test, task="regression")

        assert isinstance(res, EvaluationResult)
        assert "r2" in res.metrics
        assert "mae" in res.metrics
        assert "rmse" in res.metrics

        plot_pred = res.plot_predictions()
        assert isinstance(plot_pred, PlotResult)

        plot_res = res.plot_residuals()
        assert isinstance(plot_res, PlotResult)
