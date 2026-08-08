"""Model evaluation module for DataKit.

Evaluates Scikit-Learn classification and regression models safely, calculating key metrics
and generating evaluation plots.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from datakit.core.results import EvaluationResult


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    task: Literal["classification", "regression", "auto"] = "auto",
) -> EvaluationResult:
    """Evaluate a trained scikit-learn model on test data.

    Args:
        model: Trained scikit-learn model or pipeline object.
        X_test: Test features DataFrame or array.
        y_test: True test target Series or array.
        task: Task type ("classification", "regression", or "auto").

    Returns:
        EvaluationResult object.

    Raises:
        ImportError: If scikit-learn is missing.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
        )
    except ImportError as e:
        raise ImportError("scikit-learn is required for model evaluation. Install with 'pip install scikit-learn'.") from e

    y_true = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test
    y_pred = pd.Series(model.predict(X_test), index=y_true.index)

    model_name = type(model).__name__

    # Infer task if auto
    if task == "auto":
        unique_vals = y_true.nunique() if hasattr(y_true, "nunique") else len(np.unique(y_true))
        if pd.api.types.is_numeric_dtype(y_true) and unique_vals > 15:
            task_clean = "regression"
        else:
            task_clean = "classification"
    else:
        task_clean = task.lower()

    metrics: dict[str, float] = {}
    cm_df: pd.DataFrame | None = None

    if task_clean == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        labels = sorted(list(y_true.unique()))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    else:
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))

    return EvaluationResult(
        task=task_clean,
        metrics=metrics,
        confusion_matrix=cm_df,
        predictions=y_pred,
        actuals=y_true,
        model_name=model_name,
    )
