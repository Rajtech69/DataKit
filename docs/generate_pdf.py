"""Generate detailed book-style PDF user manual for DataKit using reportlab."""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak, Table, TableStyle
from reportlab.lib import colors

def build_pdf():
    os.makedirs("docs", exist_ok=True)
    pdf_path = os.path.join("docs", "DataKit_User_Guide.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("DocTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#003366"), alignment=0)
    subtitle_style = ParagraphStyle("DocSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=15, textColor=colors.HexColor("#444444"))
    h1_style = ParagraphStyle("Heading1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#003366"), spaceBefore=14, spaceAfter=6)
    h2_style = ParagraphStyle("Heading2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#222222"), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=colors.HexColor("#222222"))
    code_style = ParagraphStyle("CodeCustom", fontName="Courier", fontSize=8, leading=10.5, textColor=colors.HexColor("#002b36"), backColor=colors.HexColor("#f5f5f5"), borderPadding=6)

    story = []

    # Title Page Header
    story.append(Paragraph("DataKit: A Safety-First Abstraction Layer for Python Data Science", title_style))
    story.append(Paragraph("Comprehensive Technical Guide & Architectural Reference Manual | Author: <b>Rajtech69</b> | v0.1.0 (August 2026)", subtitle_style))
    story.append(Spacer(1, 10))

    # Abstract Card
    abstract_text = "<b>Abstract:</b> DataKit is a human-oriented, safety-first Python library built directly on top of NumPy, Pandas, Matplotlib, and Seaborn. It provides explicit safety guards against silent broadcasting bugs, non-destructive data cleaning with mandatory confirmation gates, pure Object-Oriented visual rendering without global state-bleed, scikit-learn pipeline dataset preparation, and zero-dependency multi-format report generation."
    story.append(Paragraph(abstract_text, body_style))
    story.append(Spacer(1, 10))

    # Table of Contents Summary
    story.append(Paragraph("Table of Contents", h1_style))
    toc_items = [
        "1. Architectural Philosophy & Safety-First Paradigm",
        "2. Installation & Package Configuration",
        "3. Core DataKit Engine & File Loading Matrix",
        "4. Broadcasting & Safety Module (dk.safe)",
        "5. Quality Audit & Non-Destructive Guided Cleaning",
        "6. Statistical Analysis & Exploratory Profiling",
        "7. Object-Oriented Visualization Engine (data.plot)",
        "8. Machine Learning Dataset Preparation (data.prepare)",
        "9. Zero-Dependency Synthesis Reporting & Memory Profiler",
        "10. Built-In Synthetic Datasets (dk.load_dataset)",
        "11. End-to-End Practical Case Study"
    ]
    for item in toc_items:
        story.append(Paragraph(f"&bull; {item}", body_style))
        story.append(Spacer(1, 2))
    story.append(Spacer(1, 10))

    # Chapter 1
    story.append(Paragraph("Chapter 1: Architectural Philosophy & Safety-First Paradigm", h1_style))
    story.append(Paragraph("DataKit is designed around five core pillars:", body_style))
    principles = [
        "<b>1. Explicit Safety over Silent Convenience:</b> Destructive cleaning requires confirm=True.",
        "<b>2. Immutable Pipeline Ergonomics:</b> Methods that transform data return a new DataKit instance.",
        "<b>3. Zero Matplotlib State-Bleed:</b> All plots operate on pure OO Figure and Axes objects.",
        "<b>4. 3-Part Explained Errors:</b> Error messages state what happened, why it matters, and how to fix it.",
        "<b>5. Thin Abstraction Layer:</b> Orchestrates Pandas and NumPy directly without redundant wrappers."
    ]
    for p in principles:
        story.append(Paragraph(f"&bull; {p}", body_style))
        story.append(Spacer(1, 2))

    # Chapter 2 & 3
    story.append(Paragraph("Chapter 2 & 3: Installation & Core DataKit Engine", h1_style))
    code_core = "pip install git+https://github.com/Rajtech69/DataKit.git\n\nimport datakit as dk\n\n# Load CSV, Excel, Parquet, JSON, or DataFrame\ndata = dk.DataKit(\"insurance.csv\")\nprint(data.inspect())\n\n# Live DataFrame escape hatch\ndf_live = data.df"
    story.append(Preformatted(code_core, code_style))

    # Chapter 4 & 5
    story.append(Paragraph("Chapter 4 & 5: Safety Module & Non-Destructive Cleaning", h1_style))
    code_safe_clean = "import numpy as np\n\na = np.arange(5)          # (5,)\nb = np.arange(5)[:, None]  # (5, 1)\n\n# Intercepts rank mismatch (5,) - (5, 1) -> (5, 5)\nres = dk.safe.subtract(a, b)\n\n# Audit quality & clean with confirmation gate\naudit = data.audit()\ncleaned = data.clean(missing=\"impute_median\", duplicates=\"drop\", confirm=True)"
    story.append(Preformatted(code_safe_clean, code_style))

    # Chapter 6 & 7
    story.append(Paragraph("Chapter 6 & 7: Statistical Analysis & OO Visualization", h1_style))
    code_viz = "import matplotlib.pyplot as plt\n\n# Pure OO Plotting into custom subplots\nfig, axes = plt.subplots(1, 2, figsize=(10, 4))\ncleaned.plot.hist(\"charges\", ax=axes[0])\ncleaned.plot.scatter(\"age\", \"charges\", hue=\"smoker\", trend=True, ax=axes[1])"
    story.append(Preformatted(code_viz, code_style))

    # Chapter 8, 9 & 10
    story.append(Paragraph("Chapter 8, 9 & 10: ML Prep, Reporting & Built-In Datasets", h1_style))
    code_ml_rep = "# ML Preparation with Scikit-Learn Pipelines\nml_data = cleaned.prepare(target=\"charges\", task=\"regression\", scale=True, encode=\"onehot\")\n\n# Multi-Format Report Export\ncleaned.report(format=\"html\", path=\"synthesis_report.html\")\n\n# Built-In Synthetic Datasets\ninsurance = dk.load_dataset(\"insurance\")\nhousing = dk.load_dataset(\"housing\")\nchurn = dk.load_dataset(\"churn\")"
    story.append(Preformatted(code_ml_rep, code_style))

    doc.build(story)
    print("Book-style PDF user manual built: docs/DataKit_User_Guide.pdf")

if __name__ == "__main__":
    build_pdf()
