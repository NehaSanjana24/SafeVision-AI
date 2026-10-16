"""CSV and PDF report generation for filtered violation data."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def dataframe_to_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def build_pdf_report(frame: pd.DataFrame, scans: pd.DataFrame | None = None) -> bytes:
    """Build a compact PDF report with summary, type breakdown, and evidence thumbnails."""
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=38, leftMargin=38, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph("AI Safety Monitoring | Violation Report", styles["Title"]),
        Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
        Spacer(1, 12),
    ]
    scan_count = 0 if scans is None else len(scans)
    compliance = 100.0 if scans is None or scans.empty else float(scans["compliance_pct"].mean())
    story.append(Paragraph(f"Scans: {scan_count} &nbsp;&nbsp; Violations: {len(frame)} &nbsp;&nbsp; Mean compliance: {compliance:.1f}%", styles["Heading2"]))
    story.append(Spacer(1, 8))
    if frame.empty:
        story.append(Paragraph("No violations were recorded in the selected period.", styles["Normal"]))
    else:
        breakdown = frame.groupby("violation_type").size().sort_values(ascending=False).reset_index(name="count")
        table_data = [["Violation type", "Count"]] + breakdown.values.tolist()
        table = Table(table_data, repeatRows=1, colWidths=[350, 100])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18352d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d2ce")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f3")]),
            ("PADDING", (0, 0), (-1, -1), 7),
        ]))
        chart = Drawing(450, 190)
        chart.add(String(12, 170, "Violation breakdown", fontName="Helvetica-Bold", fontSize=11))
        bar = VerticalBarChart()
        bar.x, bar.y, bar.width, bar.height = 40, 20, 380, 130
        bar.data = [tuple(int(value) for value in breakdown["count"].tolist())]
        bar.categoryAxis.categoryNames = [str(value)[:20] for value in breakdown["violation_type"].tolist()]
        bar.categoryAxis.labels.angle = 25
        bar.categoryAxis.labels.boxAnchor = "ne"
        bar.valueAxis.valueMin = 0
        bar.valueAxis.valueMax = max(1, int(breakdown["count"].max()))
        bar.valueAxis.valueStep = max(1, int(breakdown["count"].max()) // 5)
        bar.bars[0].fillColor = colors.HexColor("#f4a340")
        chart.add(bar)
        story.extend([table, Spacer(1, 10), chart, Spacer(1, 10), Paragraph("Annotated evidence", styles["Heading2"])])
        evidence_count = 0
        if "evidence_path" in frame.columns:
            for image_path in frame["evidence_path"].dropna().drop_duplicates().head(8):
                try:
                    story.append(Image(str(image_path), width=250, height=150, kind="proportional"))
                    story.append(Spacer(1, 8))
                    evidence_count += 1
                except (OSError, ValueError):
                    continue
        if evidence_count == 0:
            story.append(Paragraph("No evidence images are available for this period.", styles["Normal"]))
    document.build(story)
    return buffer.getvalue()
