"""ML Preparation module for DataKit.

Provides safety-first preprocessing, target-leakage checks, high-cardinality warnings,
and scikit-learn pipeline wrapping.
"""
from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from datakit.config import config
from datakit.core.exceptions import ColumnNotFoundError, DataKitError
from datakit.core.infer import infer_column_types, infer_id_like_columns
from datakit.core.results import PrepareResult
from datakit.core.warnings import HighCardinalityEncodingWarning, PotentialLeakageWarning
from datakit.safety.safe_ops import align_check


def prepare_data(
    df: pd.DataFrame,
    target: str,
    task: Literal["classification", "regression"] = "classification",
    test_size: float = 0.2,
    random_state: int | None = None,
    scale: bool = True,
    encode: Literal["onehot", "ordinal", "none"] = "onehot",
    strict_leakage: bool = False,
) -> PrepareResult:
    """Prepare a dataset for Machine Learning with safety checks and sklearn pipelines.

    Args:
        df: Input DataFrame.
        target: Target column name (required).
        task: ML task type ("classification" or "regression").
        test_size: Proportion of dataset for test split (default: 0.2).
        random_state: Random seed for train_test_split.
        scale: Whether to apply StandardScaler to numeric features.
        encode: Categorical encoding strategy ("onehot", "ordinal", "none").
        strict_leakage: If True, raises DataKitError on target leakage correlation; if False, warns.

    Returns:
        PrepareResult dataclass holding split datasets and trained sklearn Pipeline.

    Raises:
        ImportError: If scikit-learn is not installed.
        ColumnNotFoundError: If target column is missing.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    except ImportError as e:
        raise ImportError(
            "prepare() requires scikit-learn. Install it with: pip install 'datakit[ml]'"
        ) from e

    if target not in df.columns:
        raise ColumnNotFoundError(target, list(df.columns))

    if encode not in ("onehot", "ordinal", "none"):
        raise ValueError(f"Unsupported encoding strategy '{encode}'. Valid: 'onehot', 'ordinal', 'none'.")

    # Separate target and feature candidates
    id_cols = set(infer_id_like_columns(df))
    feature_cols = [col for col in df.columns if col != target and col not in id_cols]

    if not feature_cols:
        raise ValueError("No valid feature columns remaining after removing target and ID-like columns.")

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Target leakage check
    if pd.api.types.is_numeric_dtype(y.dtype):
        leakage_thresh = config.get("leakage_correlation_threshold")
        numeric_features = X.select_dtypes(include=[np.number]).columns
        for col in numeric_features:
            corr_val = float(X[col].corr(y))
            if not np.isnan(corr_val) and abs(corr_val) >= leakage_thresh:
                msg = (
                    f"Feature '{col}' has near-perfect correlation ({corr_val:.3f}) "
                    f"with target column '{target}'. This indicates potential target leakage."
                )
                if strict_leakage:
                    raise DataKitError(msg)
                warnings.warn(msg, PotentialLeakageWarning, stacklevel=2)

    # Column classification
    col_types = infer_column_types(X)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t in ("categorical", "boolean")]

    # High-cardinality encoding check
    if encode == "onehot":
        card_thresh = config.get("high_cardinality_threshold")
        for col in cat_cols:
            n_unique = X[col].nunique(dropna=True)
            if n_unique > card_thresh:
                warnings.warn(
                    f"One-hot encoding column '{col}' with high cardinality ({n_unique} categories) "
                    f"will significantly expand feature space. Consider encode='ordinal'.",
                    HighCardinalityEncodingWarning,
                    stacklevel=2,
                )

    # Build sklearn ColumnTransformer transformers
    transformers = []

    if num_cols:
        num_transformer = StandardScaler() if scale else "passthrough"
        transformers.append(("num", num_transformer, num_cols))

    if cat_cols and encode != "none":
        if encode == "onehot":
            cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:  # ordinal
            cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers.append(("cat", cat_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Pre-flight index alignment safety verification
    align_check(X_train, y_train)
    align_check(X_test, y_test)

    # Fit pipeline on training data, transform both splits
    X_train_trans_arr = pipeline.fit_transform(X_train)
    X_test_trans_arr = pipeline.transform(X_test)

    # Get feature names after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_trans_arr.shape[1])]

    X_train_df = pd.DataFrame(X_train_trans_arr, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_trans_arr, columns=feature_names, index=X_test.index)

    report_lines = [
        f"ML Preparation Report (Task: {task}, Target: '{target}')",
        f"Train shape: {X_train_df.shape}, Test shape: {X_test_df.shape}",
        f"Numeric features: {len(num_cols)}, Categorical features: {len(cat_cols)}",
        f"Scaling: {scale}, Encoding: {encode}",
    ]
    report_str = "\n".join(report_lines)

    return PrepareResult(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train,
        y_test=y_test,
        preprocessing_pipeline=pipeline,
        report=report_str,
    )
