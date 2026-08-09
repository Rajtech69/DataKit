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
        from sklearn.linear_model import (
            Lasso,
            LinearRegression,
            LogisticRegression,
            Ridge,
            RidgeClassifier,
        )
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
    regression_models = {
        "linear", "linear_regression", "lr", "lr_reg", "ridge", "ridge_regression", "lasso", "lasso_regression", "svr", "support_vector_regressor"
    }
    classification_models = {
        "logistic", "log_reg", "logistic_regression", "svc", "support_vector_classifier", "naive_bayes", "nb", "gaussian_nb", "gnb", "ridge_classifier"
    }

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

    # Normalize kwargs for distance, kernel, and regularized models
    kwargs = model_kwargs.copy()
    if model_key in ("knn", "kneighbors", "k_neighbors", "k_nearest_neighbors"):
        if "k" in kwargs and "n_neighbors" not in kwargs:
            kwargs["n_neighbors"] = kwargs.pop("k")
        if "neighbors" in kwargs and "n_neighbors" not in kwargs:
            kwargs["n_neighbors"] = kwargs.pop("neighbors")

    if model_key in ("svm", "svc", "svr", "support_vector_classifier", "support_vector_regressor"):
        if "c" in kwargs and "C" not in kwargs:
            kwargs["C"] = kwargs.pop("c")

    if model_key in ("naive_bayes", "nb", "gaussian_nb", "gnb"):
        if "smoothing" in kwargs and "var_smoothing" not in kwargs:
            kwargs["var_smoothing"] = kwargs.pop("smoothing")

    params = kwargs.copy()
    if "random_state" in params:
        random_state = params.pop("random_state")

    # Uniform kwargs alias conversions between alpha and C for regularized models
    if "alpha" in params and "C" not in params:
        alpha_val = params.pop("alpha")
        if model_key in ("logistic", "log_reg", "logistic_regression", "logisticregression"):
            params["C"] = 1.0 / alpha_val if alpha_val != 0 else 1e10
        else:
            params["alpha"] = alpha_val
    elif "C" in params and "alpha" not in params:
        c_val = params.pop("C")
        if model_key in (
            "ridge",
            "ridge_regression",
            "ridgeregression",
            "lasso",
            "lasso_regression",
            "lassoregression",
            "ridge_classifier",
        ):
            params["alpha"] = 1.0 / c_val if c_val != 0 else 1e-10
        else:
            params["C"] = c_val

    # Decision tree estimators do not accept n_estimators
    tree_params = params.copy()
    tree_params.pop("n_estimators", None)

    estimator: Any

    if task == "classification":
        if model_key in ("rf", "random_forest", "randomforest"):
            estimator = RandomForestClassifier(random_state=random_state, **params)
            name = "RandomForestClassifier"
        elif model_key in ("logistic", "log_reg", "logistic_regression", "logisticregression"):
            log_params = {"max_iter": 1000, "random_state": random_state}
            log_params.update(params)
            estimator = LogisticRegression(**log_params)
            name = "LogisticRegression"
        elif model_key in ("ridge", "ridge_classifier", "ridge_classification"):
            ridge_params = {"random_state": random_state}
            ridge_params.update(params)
            estimator = RidgeClassifier(**ridge_params)
            name = "RidgeClassifier"
        elif model_key in ("tree", "decision_tree", "dt", "decisiontree"):
            estimator = DecisionTreeClassifier(random_state=random_state, **tree_params)
            name = "DecisionTreeClassifier"
        elif model_key in ("svm", "svc"):
            estimator = SVC(random_state=random_state, probability=True, **params)
            name = "SVC"
        elif model_key in ("gb", "gradient_boosting", "gradientboosting", "gbc"):
            estimator = GradientBoostingClassifier(random_state=random_state, **params)
            name = "GradientBoostingClassifier"
        elif model_key in ("extra_trees", "et", "extratrees"):
            estimator = ExtraTreesClassifier(random_state=random_state, **params)
            name = "ExtraTreesClassifier"
        elif model_key == "knn":
            estimator = KNeighborsClassifier(**params)
            name = "KNeighborsClassifier"
        elif model_key in ("naive_bayes", "nb"):
            estimator = GaussianNB(**params)
            name = "GaussianNB"
        else:
            raise ValueError(
                f"Unsupported classification model '{model}'. Valid choices: 'rf', 'random_forest', 'gb', 'gradient_boosting', 'extra_trees', 'et', 'tree', 'decision_tree', 'logistic', 'svc', 'knn', 'naive_bayes', 'ridge'."
            )
    else:  # regression
        if model_key in ("rf", "random_forest", "randomforest"):
            estimator = RandomForestRegressor(random_state=random_state, **params)
            name = "RandomForestRegressor"
        elif model_key in ("linear", "lr", "linear_regression", "linearregression"):
            lin_params = params.copy()
            lin_params.pop("random_state", None)
            estimator = LinearRegression(**lin_params)
            name = "LinearRegression"
        elif model_key in ("tree", "decision_tree", "dt", "decisiontree"):
            estimator = DecisionTreeRegressor(random_state=random_state, **tree_params)
            name = "DecisionTreeRegressor"
        elif model_key in ("svm", "svr"):
            estimator = SVR(**params)
            name = "SVR"
        elif model_key in ("gb", "gradient_boosting", "gradientboosting", "gbr"):
            estimator = GradientBoostingRegressor(random_state=random_state, **params)
            name = "GradientBoostingRegressor"
        elif model_key in ("extra_trees", "et", "extratrees"):
            estimator = ExtraTreesRegressor(random_state=random_state, **params)
            name = "ExtraTreesRegressor"
        elif model_key == "knn":
            estimator = KNeighborsRegressor(**params)
            name = "KNeighborsRegressor"
        elif model_key in ("ridge", "ridge_regression", "ridgeregression"):
            ridge_params = {"random_state": random_state}
            ridge_params.update(params)
            estimator = Ridge(**ridge_params)
            name = "Ridge"
        elif model_key in ("lasso", "lasso_regression", "lassoregression"):
            lasso_params = {"random_state": random_state, "max_iter": 1000}
            lasso_params.update(params)
            estimator = Lasso(**lasso_params)
            name = "Lasso"
        else:
            raise ValueError(
                f"Unsupported regression model '{model}'. Valid choices: 'rf', 'random_forest', 'gb', 'gradient_boosting', 'extra_trees', 'et', 'tree', 'decision_tree', 'linear', 'svr', 'knn', 'ridge', 'lasso'."
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
    if hasattr(estimator, "feature_importances_") and estimator.feature_importances_ is not None:
        try:
            feature_names = list(prep_res.X_train.columns)
        except Exception:
            try:
                feature_names = list(prep_res.preprocessing_pipeline.get_feature_names_out())
            except Exception:
                feature_names = [f"feat_{i}" for i in range(len(estimator.feature_importances_))]

        if len(feature_names) == len(estimator.feature_importances_):
            feat_importances = pd.Series(
                estimator.feature_importances_, index=feature_names, name="importance"
            ).sort_values(ascending=False)
        else:
            feat_importances = pd.Series(
                estimator.feature_importances_, name="importance"
            ).sort_values(ascending=False)
    elif hasattr(estimator, "coef_") and estimator.coef_ is not None:
        coef = np.asarray(estimator.coef_)
        if coef.ndim == 1:
            raw_imp = np.abs(coef)
        elif coef.ndim == 2:
            if coef.shape[0] == 1:
                raw_imp = np.abs(coef[0])
            else:
                raw_imp = np.abs(coef).mean(axis=0)
        else:
            raw_imp = np.abs(coef).ravel()

        try:
            feature_names = list(prep_res.X_train.columns)
        except Exception:
            try:
                feature_names = list(prep_res.preprocessing_pipeline.get_feature_names_out())
            except Exception:
                feature_names = [f"feat_{i}" for i in range(len(raw_imp))]

        if len(feature_names) == len(raw_imp):
            feat_importances = pd.Series(
                raw_imp, index=feature_names, name="importance"
            ).sort_values(ascending=False)
        else:
            feat_importances = pd.Series(
                raw_imp, name="importance"
            ).sort_values(ascending=False)

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
