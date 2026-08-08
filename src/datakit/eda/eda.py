"""Automated EDA and visualization synthesis module for DataKit.

Executes a 7-step analysis pipeline combining inspect, audit, distributions,
outliers, relationships, and automated visualization into a single structured EDAResult.
"""
from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from datakit.config import config
from datakit.core.exceptions import ColumnNotFoundError
from datakit.core.infer import infer_column_types
from datakit.core.results import EDAResult, PlotResult

VALID_STEPS = {"inspect", "audit", "distributions", "outliers", "relationships", "plots"}


def run_eda(
    df: pd.DataFrame,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    plots: bool = True,
    target: str | None = None,
    sample: int | None = None,
    strict: bool = False,
) -> EDAResult:
    """Run an automated exploratory data analysis pipeline.

    Args:
        df: Input DataFrame.
        include: List of step names to include, or None for all steps.
        exclude: List of step names to exclude.
        plots: Whether to generate default plot visualizations.
        target: Optional target column name to bias relationship & plot analysis.
        sample: Random sample size (int) to use for large datasets.
        strict: If True, any step failure raises immediately; if False, fails soft.

    Returns:
        EDAResult composite object.
    """
    if target and target not in df.columns:
        raise ColumnNotFoundError(target, list(df.columns))

    # Apply sampling if specified
    if sample and sample < len(df):
        work_df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    else:
        work_df = df

    # Determine steps to run
    steps_to_run = set(VALID_STEPS)
    if include:
        invalid = set(include) - VALID_STEPS
        if invalid:
            raise ValueError(f"Unknown EDA step(s): {sorted(invalid)}. Valid: {sorted(VALID_STEPS)}")
        steps_to_run = set(include)

    if exclude:
        steps_to_run -= set(exclude)

    if not plots:
        steps_to_run.discard("plots")

    inspect_res = None
    audit_res = None
    dist_res = None
    outlier_res = None
    rel_res = None
    figures: list[PlotResult] = []
    warnings_collected: list[str] = []

    # Step 1: inspect
    if "inspect" in steps_to_run:
        try:
            from datakit.core.datakit import DataKit
            inspect_res = DataKit(work_df).inspect()
        except Exception as e:
            if strict:
                raise
            warnings_collected.append(f"inspect step failed: {e}")

    # Step 2: audit
    if "audit" in steps_to_run:
        try:
            from datakit.quality.audit import audit
            audit_res = audit(work_df)
        except Exception as e:
            if strict:
                raise
            warnings_collected.append(f"audit step failed: {e}")

    # Step 3: distributions
    if "distributions" in steps_to_run:
        try:
            from datakit.analysis.distributions import analyze_distributions
            dist_res = analyze_distributions(work_df)
        except Exception as e:
            if strict:
                raise
            warnings_collected.append(f"distributions step failed: {e}")

    # Step 4: outliers
    if "outliers" in steps_to_run:
        try:
            from datakit.analysis.outliers import detect_outliers
            outlier_res = detect_outliers(work_df)
        except Exception as e:
            if strict:
                raise
            warnings_collected.append(f"outliers step failed: {e}")

    # Step 5: relationships
    if "relationships" in steps_to_run:
        try:
            from datakit.analysis.relationships import analyze_relationships
            rel_res = analyze_relationships(work_df)
        except Exception as e:
            if strict:
                raise
            warnings_collected.append(f"relationships step failed: {e}")

    # Step 6: default plots
    if "plots" in steps_to_run:
        try:
            figures = generate_default_plots(work_df, target=target)
        except Exception as e:
            if strict:
                raise
            warnings_collected.append(f"plots step failed: {e}")

    return EDAResult(
        inspect=inspect_res,
        audit=audit_res,
        distributions=dist_res,
        outliers=outlier_res,
        relationships=rel_res,
        figures=figures,
        warnings_collected=warnings_collected,
    )


def generate_default_plots(df: pd.DataFrame, target: str | None = None) -> list[PlotResult]:
    """Generate default plots for EDA: histograms, count plots, correlation heatmap, and target plots."""
    from datakit.core.datakit import DataKit
    dk = DataKit(df)

    figures: list[PlotResult] = []
    col_types = infer_column_types(df)

    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in col_types.items() if t == "categorical"]

    # 1. Histograms for numeric columns (capped at 5)
    for col in numeric_cols[:5]:
        try:
            fig_res = dk.plot.hist(col)
            figures.append(fig_res)
        except Exception:
            pass

    # 2. Count plots for categorical columns (capped at 5)
    for col in categorical_cols[:5]:
        try:
            fig_res = dk.plot.count(col)
            figures.append(fig_res)
        except Exception:
            pass

    # 3. Correlation heatmap if >=2 numeric columns
    if len(numeric_cols) >= 2:
        try:
            fig_res = dk.plot.heatmap(columns=numeric_cols[:10])
            figures.append(fig_res)
        except Exception:
            pass

    # 4. Target plots if target specified
    if target and target in df.columns:
        target_type = col_types.get(target, "numeric")
        for col in df.columns:
            if col == target:
                continue
            col_t = col_types.get(col, "numeric")

            try:
                if target_type == "numeric" and col_t == "numeric":
                    figures.append(dk.plot.scatter(col, target, trend=True))
                elif target_type == "numeric" and col_t == "categorical":
                    figures.append(dk.plot.box(target, by=col))
                elif target_type == "categorical" and col_t == "numeric":
                    figures.append(dk.plot.box(col, by=target))
            except Exception:
                pass

    return figures


def visualize(df: pd.DataFrame) -> list[PlotResult]:
    """Auto-plot reasonable default visualizations for a dataset based on column types."""
    return generate_default_plots(df)
