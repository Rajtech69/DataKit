"""Machine Learning model training and auto-fitting module for DataKit.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from datakit.core.exceptions import ColumnNotFoundError, DataKitError
from datakit.core.results import ModelResult
from datakit.ml.prepare import prepare_data


def fit_model(
    df: pd.DataFrame,
    target: str,
    model: str = "rf",
    task: Literal["classification", "regression", "auto"] = "auto",
    test_size: float = 0.2,
    scale: bool = True,
    encode: Literal["onehot", "ordinal", "none"] = "onehot",
    random_state: int | None = 42,
    **model_kwargs: Any,
) -> ModelResult:
    """Train and evaluate any machine learning algorithm in one simple, safe step.

    Purpose:
        Provides an easy-to-use, unified interface to train scikit-learn models directly from DataKit,
        handling preprocessing, pipeline fitting, metric evaluation, and feature importance extraction.

    Params:
        df (pd.DataFrame): Source DataFrame.
        target (str): Target column name.
        model (str): Algorithm shortcut name ("rf", "linear", "logistic", "tree", "svm", "svc", "svr", "gb", "knn", "ridge", "lasso", "naive_bayes").
        task (str): Task type ("classification", "regression", or "auto").
        test_size (float): Proportion of dataset for testing (default: 0.2).
        scale (bool): Whether to scale numeric features with StandardScaler.
        encode (str): Categorical encoding ("onehot", "ordinal", "none").
        random_state (int | None): Random seed for reproducible training.
        **model_kwargs: Keyword arguments passed directly to the underlying scikit-learn estimator.

    Returns:
        ModelResult: Dataclass holding trained model, metrics, feature importances, and evaluation plots.

    Mutates: No (returns ModelResult wrapping new fitted model pipeline).
    Chainable: No.
    Version Added: v0.2.0

    Errors:
        ImportError: If scikit-learn is not installed.
        ColumnNotFoundError: If target column is missing.
        ValueError: If unsupported model shortcut requested.
    """
    try:
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC, SVR
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    except ImportError as e:
        raise ImportError(
            "fit_model() requires scikit-learn. Install it with: pip install 'datakit[ml]'"
        ) from e

    if target not in df.columns:
        raise ColumnNotFoundError(target, list(df.columns))

    model_key = model.lower().strip()

    # Explicit model type overrides for auto task resolution
    regression_models = {"linear", "linear_regression", "lr_reg", "ridge", "lasso", "svr"}
    classification_models = {"logistic", "log_reg", "svc", "naive_bayes", "nb"}

    if task == "auto":
        if model_key in regression_models:
            task = "regression"
        elif model_key in classification_models:
            task = "classification"
        else:
            target_series = df[target].dropna()
            if pd.api.types.is_float_dtype(target_series.dtype) or (
                pd.api.types.is_numeric_dtype(target_series.dtype) and len(target_series.unique()) > 10
            ):
                task = "regression"
            else:
                task = "classification"

    # Split and preprocess data using prepare_data
    prep_res = prepare_data(
        df,
        target=target,
        task=task,
        test_size=test_size,
        random_state=random_state,
        scale=scale,
        encode=encode,
    )

    estimator: Any

    if task == "classification":
        if model_key in ("rf", "random_forest", "randomforest"):
            estimator = RandomForestClassifier(random_state=random_state, **model_kwargs)
            name = "RandomForestClassifier"
        elif model_key in ("logistic", "log_reg"):
            estimator = LogisticRegression(random_state=random_state, max_iter=1000, **model_kwargs)
            name = "LogisticRegression"
        elif model_key in ("tree", "decision_tree"):
            estimator = DecisionTreeClassifier(random_state=random_state, **model_kwargs)
            name = "DecisionTreeClassifier"
        elif model_key in ("svm", "svc"):
            estimator = SVC(random_state=random_state, probability=True, **model_kwargs)
            name = "SVC"
        elif model_key in ("gb", "gradient_boosting"):
            estimator = GradientBoostingClassifier(random_state=random_state, **model_kwargs)
            name = "GradientBoostingClassifier"
        elif model_key in ("extra_trees", "et"):
            estimator = ExtraTreesClassifier(random_state=random_state, **model_kwargs)
            name = "ExtraTreesClassifier"
        elif model_key == "knn":
            estimator = KNeighborsClassifier(**model_kwargs)
            name = "KNeighborsClassifier"
        elif model_key in ("naive_bayes", "nb"):
            estimator = GaussianNB(**model_kwargs)
            name = "GaussianNB"
        else:
            raise ValueError(
                f"Unsupported classification model '{model}'. Valid choices: 'rf', 'logistic', 'tree', 'svc', 'gb', 'extra_trees', 'knn', 'naive_bayes'."
            )
    else:  # regression
        if model_key in ("rf", "random_forest", "randomforest"):
            estimator = RandomForestRegressor(random_state=random_state, **model_kwargs)
            name = "RandomForestRegressor"
        elif model_key in ("linear", "lr", "linear_regression"):
            estimator = LinearRegression(**model_kwargs)
            name = "LinearRegression"
        elif model_key in ("tree", "decision_tree"):
            estimator = DecisionTreeRegressor(random_state=random_state, **model_kwargs)
            name = "DecisionTreeRegressor"
        elif model_key in ("svm", "svr"):
            estimator = SVR(**model_kwargs)
            name = "SVR"
        elif model_key in ("gb", "gradient_boosting"):
            estimator = GradientBoostingRegressor(random_state=random_state, **model_kwargs)
            name = "GradientBoostingRegressor"
        elif model_key in ("extra_trees", "et"):
            estimator = ExtraTreesRegressor(random_state=random_state, **model_kwargs)
            name = "ExtraTreesRegressor"
        elif model_key == "knn":
            estimator = KNeighborsRegressor(**model_kwargs)
            name = "KNeighborsRegressor"
        elif model_key == "ridge":
            estimator = Ridge(random_state=random_state, **model_kwargs)
            name = "Ridge"
        elif model_key == "lasso":
            estimator = Lasso(random_state=random_state, **model_kwargs)
            name = "Lasso"
        else:
            raise ValueError(
                f"Unsupported regression model '{model}'. Valid choices: 'rf', 'linear', 'tree', 'svr', 'gb', 'extra_trees', 'knn', 'ridge', 'lasso'."
            )

    # Fit estimator directly on preprocessed X_train features
    estimator.fit(prep_res.X_train, prep_res.y_train)
    y_pred = estimator.predict(prep_res.X_test)
    y_pred_series = pd.Series(y_pred, index=prep_res.y_test.index)

    metrics: dict[str, float] = {}
    if task == "classification":
        metrics["accuracy"] = float(accuracy_score(prep_res.y_test, y_pred_series))
        metrics["f1"] = float(f1_score(prep_res.y_test, y_pred_series, average="weighted", zero_division=0))
        metrics["precision"] = float(precision_score(prep_res.y_test, y_pred_series, average="weighted", zero_division=0))
        metrics["recall"] = float(recall_score(prep_res.y_test, y_pred_series, average="weighted", zero_division=0))
    else:
        metrics["r2"] = float(r2_score(prep_res.y_test, y_pred_series))
        metrics["mae"] = float(mean_absolute_error(prep_res.y_test, y_pred_series))
        mse = float(mean_squared_error(prep_res.y_test, y_pred_series))
        metrics["mse"] = mse
        metrics["rmse"] = float(np.sqrt(mse))

    # Feature importances extraction if supported
    feat_importances: pd.Series | None = None
    if hasattr(estimator, "feature_importances_"):
        try:
            feature_names = prep_res.preprocessing_pipeline.get_feature_names_out()
        except Exception:
            feature_names = [f"feat_{i}" for i in range(len(estimator.feature_importances_))]
        feat_importances = pd.Series(estimator.feature_importances_, index=feature_names).sort_values(ascending=False)

    return ModelResult(
        model_name=name,
        task=task,
        model=estimator,
        metrics=metrics,
        feature_importances=feat_importances,
        X_train=prep_res.X_train,
        X_test=prep_res.X_test,
        y_train=prep_res.y_train,
        y_test=prep_res.y_test,
        y_pred=y_pred_series,
    )
