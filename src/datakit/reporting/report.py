"""Reporting module for DataKit.

Generates standalone HTML, Markdown, and JSON reports from EDAResult / AuditResult
using minimal hand-rolled templates (no external template engine dependencies).
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from datakit.core.results import AuditResult, EDAResult


def generate_report(
    source: Any,
    result: EDAResult | AuditResult | None = None,
    format: Literal["html", "markdown", "json"] = "html",
    path: str | Path | None = None,
) -> str:
    """Generate a structured 8-section report in HTML, Markdown, or JSON.

    Args:
        source: DataKit instance or Pandas DataFrame.
        result: Pre-computed EDAResult or AuditResult, or None to run eda() fresh.
        format: Output format ("html", "markdown", "json").
        path: File path to save report. If None, uses default filename.

    Returns:
        String file path where report was written.

    Raises:
        ValueError: If format is 'pdf' or unsupported.
    """
    fmt_clean = format.lower()
    if fmt_clean == "pdf":
        raise ValueError(
            "PDF export is not supported per PRD §7/§32. "
            "Use format='html' or format='markdown' instead."
        )

    if fmt_clean not in ("html", "markdown", "md", "json"):
        raise ValueError(f"Unsupported report format '{format}'. Valid: 'html', 'markdown', 'json'.")

    # If result not provided, run eda() on source
    if result is None:
        if hasattr(source, "eda"):
            eda_result = source.eda(plots=False)
        else:
            from datakit.core.datakit import DataKit
            eda_result = DataKit(source).eda(plots=False)
    elif isinstance(result, EDAResult):
        eda_result = result
    elif isinstance(result, AuditResult):
        eda_result = EDAResult(audit=result)
    else:
        raise TypeError(f"result must be EDAResult, AuditResult, or None. Got {type(result).__name__}")

    # Set default file path if not provided
    if path is None:
        ext = "html" if fmt_clean == "html" else ("json" if fmt_clean == "json" else "md")
        save_path = Path(f"datakit_report.{ext}")
    else:
        save_path = Path(path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if fmt_clean == "json":
        content = _render_json(eda_result, timestamp)
    elif fmt_clean in ("markdown", "md"):
        content = _render_markdown(eda_result, timestamp)
    else:
        content = _render_html(eda_result, timestamp)

    save_path.write_text(content, encoding="utf-8")
    return str(save_path.resolve())


def _render_json(eda: EDAResult, timestamp: str) -> str:
    data = {
        "report_title": "DataKit Synthesis Report",
        "generated_at": timestamp,
        "summary": eda.summary(),
        "inspect": eda.inspect.to_dict() if eda.inspect else None,
        "audit": eda.audit.to_dict() if eda.audit else None,
        "distributions": eda.distributions.to_dict() if eda.distributions else None,
        "outliers": eda.outliers.to_dict() if eda.outliers else None,
        "relationships": eda.relationships.to_dict() if eda.relationships else None,
        "recommendations": eda.audit.recommendations if eda.audit else [],
    }
    return json.dumps(data, indent=2, default=str)


def _render_markdown(eda: EDAResult, timestamp: str) -> str:
    lines = [
        "# DataKit Synthesis Report",
        f"*Generated at: {timestamp}*",
        "",
        "## 1. Executive Summary",
        eda.summary(),
        "",
    ]

    if eda.inspect:
        lines.extend([
            "## 2. Dataset Overview",
            f"- **Rows:** {eda.inspect.shape[0]:,}",
            f"- **Columns:** {eda.inspect.shape[1]}",
            f"- **Memory:** {eda.inspect.memory_mb:.2f} MB",
            "",
        ])

    if eda.audit:
        lines.extend([
            "## 3. Data Quality Audit",
            f"Total Issues Found: {len(eda.audit.issues)}",
            "",
            "| Column | Severity | Issue Message | Recommendation |",
            "|---|---|---|---|",
        ])
        for issue in eda.audit.issues:
            col_str = issue.column or "row-level"
            lines.append(f"| {col_str} | **{issue.severity}** | {issue.message} | {issue.recommendation} |")
        lines.append("")

    if eda.distributions:
        lines.extend([
            "## 4. Distributions Analysis",
            "```",
            eda.distributions.summary(),
            "```",
            "",
        ])

    if eda.outliers:
        lines.extend([
            "## 5. Outliers Analysis",
            eda.outliers.summary(),
            "",
        ])

    if eda.relationships:
        lines.extend([
            "## 6. Relationships & Correlations",
            eda.relationships.summary(),
            "",
        ])

    if eda.audit and eda.audit.recommendations:
        lines.extend([
            "## 7. Actionable Recommendations",
        ])
        for rec in eda.audit.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)


def _render_html(eda: EDAResult, timestamp: str) -> str:
    issues_html = ""
    if eda.audit and eda.audit.issues:
        colors = {"critical": "#d9534f", "warning": "#f0ad4e", "info": "#5bc0de"}
        rows = []
        for issue in eda.audit.issues:
            color = colors.get(issue.severity, "#333")
            col_str = html.escape(str(issue.column or "row-level"))
            rows.append(
                f"<tr style='color:{color};'>"
                f"<td>{col_str}</td>"
                f"<td><strong>{html.escape(issue.severity)}</strong></td>"
                f"<td>{html.escape(issue.message)}</td>"
                f"<td>{html.escape(issue.recommendation)}</td>"
                f"</tr>"
            )
        issues_table = "".join(rows)
        issues_html = f"""
        <table>
            <thead>
                <tr><th>Column</th><th>Severity</th><th>Message</th><th>Recommendation</th></tr>
            </thead>
            <tbody>{issues_table}</tbody>
        </table>
        """
    else:
        issues_html = "<p>No quality issues detected.</p>"

    recs_html = ""
    if eda.audit and eda.audit.recommendations:
        items = "".join(f"<li>{html.escape(r)}</li>" for r in eda.audit.recommendations)
        recs_html = f"<ul>{items}</ul>"

    inspect_html = ""
    if eda.inspect:
        inspect_html = f"""
        <p><strong>Shape:</strong> {eda.inspect.shape[0]:,} rows &times; {eda.inspect.shape[1]} columns</p>
        <p><strong>Memory:</strong> {eda.inspect.memory_mb:.2f} MB</p>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DataKit Synthesis Report</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        h1 {{ border-bottom: 2px solid #0066cc; color: #0066cc; padding-bottom: 10px; }}
        h2 {{ margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; color: #444; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .card {{ background: #fafafa; border: 1px solid #e5e5e5; border-radius: 6px; padding: 15px; margin: 15px 0; }}
    </style>
</head>
<body>
    <h1>DataKit Synthesis Report</h1>
    <p><em>Generated at: {timestamp}</em></p>

    <h2>1. Executive Summary</h2>
    <div class="card"><pre>{html.escape(eda.summary())}</pre></div>

    <h2>2. Dataset Overview</h2>
    {inspect_html}

    <h2>3. Quality Audit</h2>
    {issues_html}

    <h2>4. Actionable Recommendations</h2>
    {recs_html}
</body>
</html>
"""
