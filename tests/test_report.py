"""Tests for data.report() method and report generation."""
import json
from pathlib import Path
import pytest

from datakit import DataKit


class TestReport:
    def test_report_html_default(self, sample_datakit, tmp_path):
        out_path = tmp_path / "report.html"
        res_path = sample_datakit.report(format="html", path=out_path)
        assert Path(res_path).exists()

        content = out_path.read_text(encoding="utf-8")
        assert "DataKit Synthesis Report" in content
        assert "Executive Summary" in content
        assert "Quality Audit" in content

    def test_report_markdown(self, sample_datakit, tmp_path):
        out_path = tmp_path / "report.md"
        res_path = sample_datakit.report(format="markdown", path=out_path)
        assert Path(res_path).exists()

        content = out_path.read_text(encoding="utf-8")
        assert "# DataKit Synthesis Report" in content
        assert "## 1. Executive Summary" in content

    def test_report_json(self, sample_datakit, tmp_path):
        out_path = tmp_path / "report.json"
        res_path = sample_datakit.report(format="json", path=out_path)
        assert Path(res_path).exists()

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "report_title" in data
        assert "summary" in data
        assert "audit" in data

    def test_report_pdf_raises_value_error(self, sample_datakit):
        with pytest.raises(ValueError, match="PDF export is not supported"):
            sample_datakit.report(format="pdf")

    def test_report_from_precomputed_audit(self, insurance_datakit, tmp_path):
        audit_res = insurance_datakit.audit()
        out_path = tmp_path / "audit_report.html"
        res_path = insurance_datakit.report(result=audit_res, format="html", path=out_path)
        assert Path(res_path).exists()
