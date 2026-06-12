from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(
    age,
    bmi,
    premium,
    risk
):

    pdf = SimpleDocTemplate(
        "customer_report.pdf"
    )

    styles = getSampleStyleSheet()

    content = [

        Paragraph(
            "Insurance Risk Report",
            styles["Title"]
        ),

        Paragraph(
            f"Age: {age}",
            styles["BodyText"]
        ),

        Paragraph(
            f"BMI: {bmi}",
            styles["BodyText"]
        ),

        Paragraph(
            f"Premium: ₹{premium:,.0f}",
            styles["BodyText"]
        ),

        Paragraph(
            f"Risk: {risk}",
            styles["BodyText"]
        )
    ]

    pdf.build(content)