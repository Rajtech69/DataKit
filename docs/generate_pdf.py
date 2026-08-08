"""Generate PDF documentation for DataKit using reportlab."""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak
from reportlab.lib import colors

def build_pdf():
    os.makedirs("docs", exist_ok=True)
    pdf_path = os.path.join("docs", "DataKit_User_Guide.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("DocTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#003366"), alignment=0)
    subtitle_style = ParagraphStyle("DocSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=15, textColor=colors.HexColor("#444444"))
    h1_style = ParagraphStyle("Heading1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#003366"), spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#222222"))
    code_style = ParagraphStyle("CodeCustom", fontName="Courier", fontSize=8.5, leading=11, textColor=colors.HexColor("#002b36"), backColor=colors.HexColor("#f5f5f5"), borderPadding=6)

    story = []

    # Title & Metadata
    story.append(Paragraph("DataKit: Safety-First Python Data Science Layer", title_style))
    story.append(Paragraph("Official Technical User Guide & Reference Manual | Author: <b>Rajtech69</b> | v0.1.0 (August 2026)", subtitle_style))
    story.append(Spacer(1, 12))

    # Abstract
    abstract_text = "<b>Abstract:</b> DataKit is a human-oriented, safety-first Python library built directly on top of NumPy, Pandas, Matplotlib, and Seaborn. It provides explicit safety guards against silent broadcasting bugs, non-destructive data cleaning with mandatory confirmation gates, pure Object-Oriented visual rendering without global state-bleed, scikit-learn pipeline dataset preparation, and zero-dependency multi-format report generation."
    story.append(Paragraph(abstract_text, body_style))
    story.append(Spacer(1, 10))

    # Section 1
    story.append(Paragraph("1. Architectural Foundations", h1_style))
    story.append(Paragraph("DataKit is built around five explicit design principles:", body_style))
    principles = [
        "<b>1. Explicit Safety over Silent Convenience:</b> Destructive cleaning operations require confirm=True.",
        "<b>2. Immutable Pipeline Ergonomics:</b> Methods that clean or transform data return a new DataKit instance.",
        "<b>3. Zero Matplotlib State-Bleed:</b> All plots operate on pure OO Figure and Axes objects.",
        "<b>4. 3-Part Explained Errors:</b> Error messages state what happened, why it matters, and how to fix it.",
        "<b>5. Thin Abstraction Layer:</b> Orchestrates Pandas and NumPy directly without redundant wrappers."
    ]
    for p in principles:
        story.append(Paragraph(f"&bull; {p}", body_style))
        story.append(Spacer(1, 3))

    # Section 2
    story.append(Paragraph("2. Quick Installation", h1_style))
    code_install = "pip install git+https://github.com/Rajtech69/DataKit.git\n\n# With optional extras (scikit-learn, openpyxl, pyarrow)\npip install \"git+https://github.com/Rajtech69/DataKit.git#egg=datakit[all]\""
    story.append(Preformatted(code_install, code_style))

    # Section 3
    story.append(Paragraph("3. Core API (dk.DataKit)", h1_style))
    story.append(Paragraph("The <b>DataKit</b> class automatically loads CSV, Excel, Parquet, and JSON files, keeping an immutable copy internally while exposing .df for raw Pandas access.", body_style))
    code_core = "import datakit as dk\n\ndata = dk.DataKit(\"insurance.csv\")\nprint(data.inspect())\ndf_raw = data.df  # Live DataFrame escape hatch"
    story.append(Preformatted(code_core, code_style))

    # Section 4
    story.append(Paragraph("4. Safety Module (dk.safe.*)", h1_style))
    story.append(Paragraph("Prevents silent rank mismatch broadcasting bugs (e.g. (n,) - (n, 1) producing (n, n)).", body_style))
    code_safe = "import datakit as dk\nimport numpy as np\n\na = np.arange(5)          # (5,)\nb = np.arange(5)[:, None]  # (5, 1)\n\n# Issues ImplicitBroadcastWarning naming (5,) - (5, 1) -> (5, 5)\nres = dk.safe.subtract(a, b)"
    story.append(Preformatted(code_safe, code_style))

    # Section 5
    story.append(Paragraph("5. Quality Audit & Non-Destructive Cleaning", h1_style))
    code_clean = "# 1. Audit dataset quality\naudit = data.audit()\nprint(audit.summary)\n\n# 2. Non-destructive cleaning (confirm=True gate required)\ncleaned = data.clean(missing=\"impute_median\", duplicates=\"drop\", confirm=True)\nprint(cleaned.last_clean_report.diff_summary())"
    story.append(Preformatted(code_clean, code_style))

    # Section 6
    story.append(Paragraph("6. Object-Oriented Visualizations & ML Prep", h1_style))
    code_plot = "import matplotlib.pyplot as plt\n\n# OO Plotting into custom subplot grid\nfig, axes = plt.subplots(1, 2, figsize=(10, 4))\ncleaned.plot.hist(\"charges\", ax=axes[0])\ncleaned.plot.scatter(\"age\", \"charges\", hue=\"smoker\", trend=True, ax=axes[1])\n\n# Scikit-Learn Machine Learning Preparation\nml_data = cleaned.prepare(target=\"charges\", task=\"regression\", scale=True, encode=\"onehot\")\nprint(\"Pipeline:\", ml_data.preprocessing_pipeline)"
    story.append(Preformatted(code_plot, code_style))

    # Section 7
    story.append(Paragraph("7. Zero-Dependency Synthesis Reports", h1_style))
    code_report = "# Generates standalone 8-section reports\ncleaned.report(format=\"html\", path=\"report.html\")\ncleaned.report(format=\"markdown\", path=\"report.md\")\ncleaned.report(format=\"json\", path=\"report.json\")"
    story.append(Preformatted(code_report, code_style))

    doc.build(story)
    print("PDF build complete: docs/DataKit_User_Guide.pdf")

if __name__ == "__main__":
    build_pdf()
