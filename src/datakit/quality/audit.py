"""Data audit module for DataKit.

Produces structured, prioritized data-quality reports with recommendations.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd

from datakit.config import config
from datakit.core.infer import (
    infer_column_types,
    infer_id_like_columns,
    infer_suspicious_dtypes,
    infer_target_candidates,
)
from datakit.core.results import AuditResult, Issue
from datakit.core.warnings import ConstantColumnWarning, HighCardinalityWarning

ALL_CHECKS = {"missing", "duplicates", "constant", "cardinality", "suspicious_dtype", "id_like", "leakage"}
SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


def audit(
    df: pd.DataFrame,
    checks: list[str] | None = None,
    severity_threshold: Literal["info", "warning", "critical"] = "info",
) -> AuditResult:
    """Perform a structured data quality audit on a DataFrame.

    Args:
        df: Input DataFrame.
        checks: List of check names to run, or None for all checks.
            Supported: "missing", "duplicates", "constant", "cardinality",
            "suspicious_dtype", "id_like", "leakage".
        severity_threshold: Minimum severity to include in returned issues.

    Returns:
        AuditResult object.

    Raises:
        ValueError: If an unknown check name is specified.
    """
    if checks is None:
        run_checks = set(ALL_CHECKS)
    else:
        invalid = set(checks) - ALL_CHECKS
        if invalid:
            raise ValueError(
                f"Unknown check(s) specified: {sorted(invalid)}. "
                f"Available checks: {sorted(ALL_CHECKS)}"
            )
        run_checks = set(checks)

    issues: list[Issue] = []
    n_rows = len(df)

    if n_rows == 0:
        return AuditResult(
            summary="Dataset has 0 rows.",
            issues=[
                Issue(
                    column=None,
                    severity="critical",
                    message="DataFrame has 0 rows.",
                    recommendation="Ensure upstream pipeline loads non-empty data.",
                )
            ],
        )

    col_types = infer_column_types(df)

    # 1. Missing values check
    if "missing" in run_checks:
        crit_thresh = config.get("missing_critical_threshold")
        warn_thresh = config.get("missing_warning_threshold")

        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                pct = (null_count / n_rows) * 100
                ratio = null_count / n_rows

                if ratio >= crit_thresh:
                    severity = "critical"
                    rec = f"Consider dropping column '{col}' or investigating why >{crit_thresh*100:.0f}% values are missing."
                elif ratio >= warn_thresh:
                    severity = "warning"
                    rec = f"Impute missing values in '{col}' or flag missingness before modeling."
                else:
                    severity = "info"
                    rec = f"Review missing values in '{col}' and consider imputation strategy."

                issues.append(
                    Issue(
                        column=str(col),
                        severity=severity,
                        message=f"Column '{col}' has {null_count} missing values ({pct:.1f}%).",
                        recommendation=rec,
                    )
                )

    # 2. Duplicate rows check
    if "duplicates" in run_checks:
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            dup_pct = (dup_count / n_rows) * 100
            severity = "warning" if dup_pct < 10 else "critical"
            issues.append(
                Issue(
                    column=None,
                    severity=severity,
                    message=f"Dataset contains {dup_count} duplicate rows ({dup_pct:.1f}%).",
                    recommendation="Review and drop duplicate rows using data.clean(duplicates='drop', confirm=True).",
                )
            )

    # 3. Constant and near-constant columns check
    if "constant" in run_checks:
        near_constant_thresh = config.get("near_constant_threshold")

        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            n_unique = series.nunique()
            if n_unique == 1:
                val = series.iloc[0]
                warnings.warn(
                    f"Column '{col}' is constant (all non-null values = {val!r}).",
                    ConstantColumnWarning,
                )
                issues.append(
                    Issue(
                        column=str(col),
                        severity="warning",
                        message=f"Column '{col}' is constant (all values = {val!r}).",
                        recommendation=f"Drop constant column '{col}' as it contains zero variance.",
                    )
                )
            else:
                top_val_count = series.value_counts().iloc[0]
                top_ratio = top_val_count / len(series)
                if top_ratio >= near_constant_thresh:
                    top_val = series.value_counts().index[0]
                    issues.append(
                        Issue(
                            column=str(col),
                            severity="info",
                            message=f"Column '{col}' is near-constant ({top_ratio*100:.1f}% single value {top_val!r}).",
                            recommendation=f"Verify if column '{col}' provides useful signal.",
                        )
                    )

    # 4. High cardinality check
    if "cardinality" in run_checks:
        card_thresh = config.get("high_cardinality_threshold")

        for col in df.columns:
            if col_types.get(col) == "categorical":
                n_unique = df[col].nunique(dropna=True)
                if n_unique > card_thresh:
                    warnings.warn(
                        f"Column '{col}' has high cardinality ({n_unique} unique categories).",
                        HighCardinalityWarning,
                    )
                    issues.append(
                        Issue(
                            column=str(col),
                            severity="warning",
                            message=f"Column '{col}' has high cardinality ({n_unique} unique categories).",
                            recommendation=f"Group rare categories or use target/frequency encoding for '{col}'.",
                        )
                    )

    # 5. Suspicious dtype check
    if "suspicious_dtype" in run_checks:
        suspicious = infer_suspicious_dtypes(df)
        for col, suggested_type in suspicious:
            issues.append(
                Issue(
                    column=col,
                    severity="warning",
                    message=f"Column '{col}' has object/string dtype but appears to be {suggested_type}.",
                    recommendation=f"Convert '{col}' to {suggested_type} dtype using data.clean(dtypes='coerce', confirm=True).",
                )
            )

    # 6. ID-like column check
    if "id_like" in run_checks:
        id_cols = infer_id_like_columns(df)
        for col in id_cols:
            issues.append(
                Issue(
                    column=col,
                    severity="info",
                    message=f"Column '{col}' appears to be a row identifier.",
                    recommendation=f"Exclude ID column '{col}' from features, correlations, and scaling.",
                )
            )

    # 7. Leakage check
    if "leakage" in run_checks:
        targets = infer_target_candidates(df)
        leakage_thresh = config.get("leakage_correlation_threshold")

        if targets:
            target_col = targets[0]
            if pd.api.types.is_numeric_dtype(df[target_col].dtype):
                numeric_df = df.select_dtypes(include=[np.number])
                if target_col in numeric_df.columns and numeric_df.shape[1] > 1:
                    corrs = numeric_df.corr()[target_col].abs()
                    for col, corr_val in corrs.items():
                        col_str = str(col)
                        if col_str == target_col or col_str in infer_id_like_columns(df):
                            continue
                        if corr_val >= leakage_thresh:
                            issues.append(
                                Issue(
                                    column=col_str,
                                    severity="critical",
                                    message=f"Column '{col_str}' has near-perfect correlation ({corr_val:.3f}) with target candidate '{target_col}'.",
                                    recommendation=f"Inspect '{col_str}' for target leakage and exclude if computed post-outcome.",
                                )
                            )

    # Filter issues by severity threshold
    min_level = SEVERITY_ORDER.get(severity_threshold, 0)
    filtered_issues = [
        issue for issue in issues if SEVERITY_ORDER.get(issue.severity, 0) >= min_level
    ]

    # Generate summary
    n_crit = sum(1 for i in filtered_issues if i.severity == "critical")
    n_warn = sum(1 for i in filtered_issues if i.severity == "warning")
    n_info = sum(1 for i in filtered_issues if i.severity == "info")

    summary = (
        f"Audit completed: {len(filtered_issues)} issue(s) found "
        f"({n_crit} critical, {n_warn} warning, {n_info} info) across {len(df.columns)} columns."
    )

    return AuditResult(summary=summary, issues=filtered_issues)
