from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph,
    Spacer, Image, Table,
    TableStyle, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.pdfgen import canvas


def create_pdf(filename="example_report.pdf"):
    # --- Setup document ---
    doc = SimpleDocTemplate(
        filename, pagesize=LETTER,
        rightMargin=36, leftMargin=36,
        topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()

    # Add custom style
    styles.add(ParagraphStyle(name="TitleStyle",
                              fontSize=20,
                              leading=24,
                              textColor=colors.darkblue,
                              alignment=1))  # Centered

    flowables = []  # story content

    # --- Title Page ---
    flowables.append(Paragraph("ReportLab Demo Report", styles["TitleStyle"]))
    flowables.append(Spacer(1, 24))
    flowables.append(Paragraph(
        "This document demonstrates some of the capabilities of the ReportLab library in Python.",
        styles["Normal"]
    ))
    flowables.append(PageBreak())

    # --- Paragraph Styling ---
    flowables.append(Paragraph("Formatted Text Examples", styles["Heading2"]))
    flowables.append(Spacer(1, 12))

    for i in range(1, 4):
        text = f"This is example paragraph number <b>{i}</b> with some <i>italic</i>, <u>underline</u>, and <font color='red'>colored</font> text."
        flowables.append(Paragraph(text, styles["BodyText"]))
        flowables.append(Spacer(1, 6))

    flowables.append(PageBreak())

    # --- Table Example ---
    flowables.append(Paragraph("Table Example", styles["Heading2"]))
    data = [
        ["Item", "Quantity", "Price"],
        ["Apples", 10, "$2.50"],
        ["Bananas", 5, "$1.20"],
        ["Cherries", 20, "$7.00"],
        ["Dates", 12, "$3.40"],
    ]
    table = Table(data, colWidths=[100, 100, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
    ]))
    flowables.append(table)
    flowables.append(PageBreak())

    # --- Charts ---
    flowables.append(Paragraph("Charts Example", styles["Heading2"]))

    # Bar Chart
    drawing = Drawing(400, 200)
    bc = VerticalBarChart()
    bc.x = 50
    bc.y = 50
    bc.height = 125
    bc.width = 300
    bc.data = [
        (13, 20, 15, 22),
        (14, 18, 19, 21),
    ]
    bc.categoryAxis.categoryNames = ["Q1", "Q2", "Q3", "Q4"]
    bc.barLabels.nudge = 7
    bc.barLabelFormat = "%d"
    bc.groupSpacing = 10
    bc.bars[0].fillColor = colors.lightblue
    bc.bars[1].fillColor = colors.lightgreen
    drawing.add(bc)
    flowables.append(drawing)

    flowables.append(Spacer(1, 24))

    # Pie Chart
    drawing2 = Drawing(200, 150)
    pc = Pie()
    pc.x = 65
    pc.y = 15
    pc.data = [10, 20, 30, 40]
    pc.labels = ["North", "South", "East", "West"]
    pc.slices[0].fillColor = colors.pink
    pc.slices[1].fillColor = colors.lightblue
    pc.slices[2].fillColor = colors.lightgreen
    pc.slices[3].fillColor = colors.orange
    drawing2.add(pc)
    flowables.append(drawing2)

    flowables.append(PageBreak())

    # --- Custom Graphics ---
    flowables.append(Paragraph("Custom Graphics with Shapes", styles["Heading2"]))
    drawing3 = Drawing(400, 200)
    rect = Rect(50, 50, 300, 100, strokeColor=colors.black, fillColor=colors.lightgrey)
    text = String(120, 90, "Hello ReportLab!", fontSize=16, fillColor=colors.darkred)
    drawing3.add(rect)
    drawing3.add(text)
    flowables.append(drawing3)

    # --- Build Document ---
    doc.build(flowables)


# Extra: Draw directly with Canvas (optional)
def create_canvas_demo(filename="canvas_demo.pdf"):
    c = canvas.Canvas(filename, pagesize=landscape(LETTER))
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(colors.darkblue)
    c.drawCentredString(400, 500, "Canvas Drawing Example")

    # Draw lines and shapes
    c.setStrokeColor(colors.red)
    c.line(100, 450, 700, 450)
    c.rect(100, 300, 200, 100, stroke=1, fill=1)

    c.setFillColor(colors.green)
    c.circle(500, 350, 50, stroke=1, fill=1)

    c.showPage()
    c.save()


if __name__ == "__main__":
    create_pdf()
    create_canvas_demo()
    print("PDFs created: example_report.pdf and canvas_demo.pdf")
