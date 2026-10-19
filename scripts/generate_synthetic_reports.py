"""
MediNova Synthetic Clinical Laboratory Report Generator
=======================================================
Generates authenticated, realistic medical laboratory report PDF files
in static/samples/ for all 5 tabular diagnostic modules.
Explicitly labeled:
"SYNTHETIC DEMO — NOT A REAL PATIENT — FOR SOFTWARE TESTING ONLY"
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

REPORTS_DATA = {
    # 1. Diabetes
    "diabetes_sample_positive.pdf": {
        "title": "Comprehensive Glycemic & Metabolic Biomarker Panel",
        "patient": {"Age": "62 yrs", "Gender": "Male", "Sample ID": "SYN-GLYC-8821", "Status": "Hyperglycemic"},
        "analytes": [
            ("Glycated Hemoglobin (HbA1c)", "8.8", "%", "4.0 - 5.6"),
            ("Fasting Blood Glucose", "215", "mg/dL", "70 - 99"),
            ("Body Mass Index (BMI)", "33.5", "kg/m²", "18.5 - 24.9"),
            ("Patient Age", "62", "years", "18 - 85"),
            ("Biological Gender", "Male", "", "Male / Female"),
            ("Hypertension History", "Positive", "", "Negative"),
            ("Coronary Heart Disease", "Positive", "", "Negative"),
            ("Smoking Status", "Current", "", "Never / Non-Smoker")
        ]
    },
    "diabetes_sample_negative.pdf": {
        "title": "Comprehensive Glycemic & Metabolic Biomarker Panel",
        "patient": {"Age": "29 yrs", "Gender": "Female", "Sample ID": "SYN-GLYC-1092", "Status": "Euglycemic"},
        "analytes": [
            ("Glycated Hemoglobin (HbA1c)", "5.1", "%", "4.0 - 5.6"),
            ("Fasting Blood Glucose", "86", "mg/dL", "70 - 99"),
            ("Body Mass Index (BMI)", "22.4", "kg/m²", "18.5 - 24.9"),
            ("Patient Age", "29", "years", "18 - 85"),
            ("Biological Gender", "Female", "", "Male / Female"),
            ("Hypertension History", "Negative", "", "Negative"),
            ("Coronary Heart Disease", "Negative", "", "Negative"),
            ("Smoking Status", "Never", "", "Never / Non-Smoker")
        ]
    },

    # 2. Heart Disease
    "heart_sample_positive.pdf": {
        "title": "Cardiovascular Stress & Coronary Biomarker Evaluation",
        "patient": {"Age": "63 yrs", "Gender": "Male", "Sample ID": "SYN-CARD-9912", "Status": "Elevated Risk"},
        "analytes": [
            ("Resting Blood Pressure (BP)", "155", "mm Hg", "90 - 120"),
            ("Serum Cholesterol", "285", "mg/dL", "125 - 200"),
            ("Fasting Blood Sugar", "140", "mg/dL", "70 - 110"),
            ("Maximum Heart Rate (Max HR)", "115", "bpm", "60 - 100"),
            ("ST Depression (Oldpeak)", "2.8", "", "0.0 - 1.0"),
            ("Resting ECG Results", "ST-T Wave Abnormality", "", "Normal"),
            ("Exercise Induced Angina", "Positive", "", "Negative"),
            ("Chest Pain Type", "Typical Angina", "", "Asymptomatic"),
            ("Slope of Peak ST", "Downsloping", "", "Upsloping"),
            ("Major Vessels by Fluoroscopy", "2", "vessels", "0 vessels"),
            ("Thallium Stress Test", "Reversible Defect", "", "Normal (3)")
        ]
    },
    "heart_sample_negative.pdf": {
        "title": "Cardiovascular Stress & Coronary Biomarker Evaluation",
        "patient": {"Age": "42 yrs", "Gender": "Female", "Sample ID": "SYN-CARD-4401", "Status": "Optimal Profile"},
        "analytes": [
            ("Resting Blood Pressure (BP)", "118", "mm Hg", "90 - 120"),
            ("Serum Cholesterol", "180", "mg/dL", "125 - 200"),
            ("Fasting Blood Sugar", "88", "mg/dL", "70 - 110"),
            ("Maximum Heart Rate (Max HR)", "168", "bpm", "60 - 100"),
            ("ST Depression (Oldpeak)", "0.2", "", "0.0 - 1.0"),
            ("Resting ECG Results", "Normal", "", "Normal"),
            ("Exercise Induced Angina", "Negative", "", "Negative"),
            ("Chest Pain Type", "Non-Anginal", "", "Asymptomatic"),
            ("Slope of Peak ST", "Upsloping", "", "Upsloping"),
            ("Major Vessels by Fluoroscopy", "0", "vessels", "0 vessels"),
            ("Thallium Stress Test", "Normal", "", "Normal (3)")
        ]
    },

    # 3. Kidney Disease
    "kidney_sample_positive.pdf": {
        "title": "Renal Function Panel & Urinalysis Chemistry",
        "patient": {"Age": "66 yrs", "Gender": "Male", "Sample ID": "SYN-RENL-7740", "Status": "Impaired Renal Function"},
        "analytes": [
            ("Serum Creatinine", "3.8", "mg/dL", "0.6 - 1.2"),
            ("Blood Urea", "68", "mg/dL", "15 - 45"),
            ("Blood Glucose Random", "185", "mg/dL", "70 - 140"),
            ("Serum Sodium", "130", "mEq/L", "135 - 145"),
            ("Serum Potassium", "5.6", "mEq/L", "3.5 - 5.0"),
            ("Hemoglobin", "9.2", "g/dL", "12.0 - 16.5"),
            ("Packed Cell Volume (PCV)", "28", "%", "36 - 50"),
            ("White Blood Cell Count", "11500", "cells/cumm", "4000 - 11000"),
            ("Red Blood Cell Count", "3.4", "millions/cmm", "4.2 - 5.8"),
            ("Resting Blood Pressure (BP)", "90", "mm Hg", "80 - 120"),
            ("Specific Gravity (sg)", "1.010", "", "1.015 - 1.025"),
            ("Urine Albumin", "3", "grade", "0 (Negative)"),
            ("Urine Sugar", "2", "grade", "0 (Negative)"),
            ("Hypertension", "Yes", "", "No"),
            ("Diabetes Mellitus", "Yes", "", "No"),
            ("Appetite", "Poor", "", "Good"),
            ("Pedal Edema", "Yes", "", "No")
        ]
    },
    "kidney_sample_negative.pdf": {
        "title": "Renal Function Panel & Urinalysis Chemistry",
        "patient": {"Age": "38 yrs", "Gender": "Female", "Sample ID": "SYN-RENL-1120", "Status": "Optimal Renal Function"},
        "analytes": [
            ("Serum Creatinine", "0.9", "mg/dL", "0.6 - 1.2"),
            ("Blood Urea", "26", "mg/dL", "15 - 45"),
            ("Blood Glucose Random", "95", "mg/dL", "70 - 140"),
            ("Serum Sodium", "141", "mEq/L", "135 - 145"),
            ("Serum Potassium", "4.3", "mEq/L", "3.5 - 5.0"),
            ("Hemoglobin", "14.8", "g/dL", "12.0 - 16.5"),
            ("Packed Cell Volume (PCV)", "44", "%", "36 - 50"),
            ("White Blood Cell Count", "6800", "cells/cumm", "4000 - 11000"),
            ("Red Blood Cell Count", "4.9", "millions/cmm", "4.2 - 5.8"),
            ("Resting Blood Pressure (BP)", "75", "mm Hg", "80 - 120"),
            ("Specific Gravity (sg)", "1.020", "", "1.015 - 1.025"),
            ("Urine Albumin", "0", "grade", "0 (Negative)"),
            ("Urine Sugar", "0", "grade", "0 (Negative)"),
            ("Hypertension", "No", "", "No"),
            ("Diabetes Mellitus", "No", "", "No"),
            ("Appetite", "Good", "", "Good"),
            ("Pedal Edema", "No", "", "No")
        ]
    },

    # 4. Liver Disease
    "liver_sample_positive.pdf": {
        "title": "Hepatic Function Profile (Liver Function Test - LFT)",
        "patient": {"Age": "56 yrs", "Gender": "Male", "Sample ID": "SYN-HEPA-6611", "Status": "Hepatic Injury Marker Elevation"},
        "analytes": [
            ("Total Bilirubin", "4.6", "mg/dL", "0.2 - 1.2"),
            ("Direct Bilirubin", "2.4", "mg/dL", "0.0 - 0.3"),
            ("Alkaline Phosphatase (ALP)", "380", "U/L", "30 - 120"),
            ("Alanine Aminotransferase (ALT / SGPT)", "95", "U/L", "7 - 56"),
            ("Aspartate Aminotransferase (AST / SGOT)", "110", "U/L", "10 - 40"),
            ("Total Proteins", "5.2", "g/dL", "6.0 - 8.3"),
            ("Serum Albumin", "2.3", "g/dL", "3.4 - 5.4"),
            ("Albumin and Globulin Ratio", "0.75", "", "1.0 - 2.2"),
            ("Patient Age", "56", "years", "18 - 85"),
            ("Biological Gender", "Male", "", "Male / Female")
        ]
    },
    "liver_sample_negative.pdf": {
        "title": "Hepatic Function Profile (Liver Function Test - LFT)",
        "patient": {"Age": "35 yrs", "Gender": "Female", "Sample ID": "SYN-HEPA-2015", "Status": "Physiological Reference Limits"},
        "analytes": [
            ("Total Bilirubin", "0.7", "mg/dL", "0.2 - 1.2"),
            ("Direct Bilirubin", "0.2", "mg/dL", "0.0 - 0.3"),
            ("Alkaline Phosphatase (ALP)", "85", "U/L", "30 - 120"),
            ("Alanine Aminotransferase (ALT / SGPT)", "22", "U/L", "7 - 56"),
            ("Aspartate Aminotransferase (AST / SGOT)", "24", "U/L", "10 - 40"),
            ("Total Proteins", "7.2", "g/dL", "6.0 - 8.3"),
            ("Serum Albumin", "4.4", "g/dL", "3.4 - 5.4"),
            ("Albumin and Globulin Ratio", "1.55", "", "1.0 - 2.2"),
            ("Patient Age", "35", "years", "18 - 85"),
            ("Biological Gender", "Female", "", "Male / Female")
        ]
    },

    # 5. Breast Cancer
    "breast_cancer_sample_positive.pdf": {
        "title": "Fine-Needle Aspiration (FNA) Nuclear Morphometry Cytology",
        "patient": {"Age": "54 yrs", "Gender": "Female", "Sample ID": "SYN-ONCO-5582", "Status": "Atypical Neoplastic Criteria"},
        "analytes": [
            ("Mean Nuclear Radius", "17.99", "µm", "6.0 - 13.9"),
            ("Mean Texture Variation", "20.38", "", "9.0 - 20.0"),
            ("Mean Nuclear Perimeter", "122.80", "µm", "40.0 - 89.9"),
            ("Mean Nuclear Area", "1001.0", "µm²", "140.0 - 499.0"),
            ("Mean Smoothness Index", "0.1184", "", "0.06 - 0.099")
        ]
    },
    "breast_cancer_sample_negative.pdf": {
        "title": "Fine-Needle Aspiration (FNA) Nuclear Morphometry Cytology",
        "patient": {"Age": "41 yrs", "Gender": "Female", "Sample ID": "SYN-ONCO-1294", "Status": "Benign Cytological Architecture"},
        "analytes": [
            ("Mean Nuclear Radius", "11.42", "µm", "6.0 - 13.9"),
            ("Mean Texture Variation", "14.02", "", "9.0 - 20.0"),
            ("Mean Nuclear Perimeter", "73.18", "µm", "40.0 - 89.9"),
            ("Mean Nuclear Area", "398.0", "µm²", "140.0 - 499.0"),
            ("Mean Smoothness Index", "0.0825", "", "0.06 - 0.099")
        ]
    }
}


def build_pdf_report(filename: str, report_info: dict):
    out_path = os.path.join(SAMPLES_DIR, filename)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Synthetic Demo Watermark Banner
    banner_style = ParagraphStyle(
        'BannerStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.HexColor('#991b1b'),
        alignment=1
    )

    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        textColor=colors.HexColor('#0891b2'),
        spaceAfter=4
    )

    sub_style = ParagraphStyle(
        'ReportSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.HexColor('#475569'),
        spaceAfter=10
    )

    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        textColor=colors.HexColor('#0f172a')
    )

    cell_normal = ParagraphStyle(
        'CellNorm',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        textColor=colors.HexColor('#334155')
    )

    story = []

    # Mandatory Notice Banner
    banner_p = Paragraph(
        "<b>SYNTHETIC DEMO — NOT A REAL PATIENT — FOR SOFTWARE TESTING ONLY</b>",
        banner_style
    )
    banner_table = Table([[banner_p]], colWidths=[540])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fee2e2')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#ef4444')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 12))

    # Header
    story.append(Paragraph("MediNova Clinical Intelligence Diagnostic Core", title_style))
    story.append(Paragraph(f"Laboratory Document: {report_info['title']}", sub_style))

    # Patient Meta Card
    p_meta = report_info["patient"]
    meta_data = [
        [
            Paragraph(f"<b>Specimen ID:</b> {p_meta['Sample ID']}", cell_normal),
            Paragraph(f"<b>Demographics:</b> {p_meta['Age']}, {p_meta['Gender']}", cell_normal),
            Paragraph(f"<b>Date:</b> 2026-10-19", cell_normal)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[180, 180, 180])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Analytes Table
    table_data = [
        [
            Paragraph("<b>Diagnostic Analyte / Biomarker Tested</b>", cell_bold),
            Paragraph("<b>Patient Result</b>", cell_bold),
            Paragraph("<b>Units</b>", cell_bold),
            Paragraph("<b>Laboratory Reference Interval</b>", cell_bold)
        ]
    ]

    for item in report_info["analytes"]:
        table_data.append([
            Paragraph(item[0], cell_bold),
            Paragraph(f"<b>{item[1]}</b>", cell_bold),
            Paragraph(item[2], cell_normal),
            Paragraph(f"Ref: {item[3]}", cell_normal)
        ])

    table = Table(table_data, colWidths=[220, 80, 70, 170])
    t_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    for row_i in range(1, len(table_data)):
        if row_i % 2 == 0:
            t_style.append(('BACKGROUND', (0, row_i), (-1, row_i), colors.HexColor('#f8fafc')))
    table.setStyle(TableStyle(t_style))
    story.append(table)
    story.append(Spacer(1, 16))

    # Pathologist Signoff
    footer_text = Paragraph(
        "<b>CLINICAL VALIDATION:</b> Specimen analyzed in accordance with automated clinical chemistry protocols. "
        "Verified for automated AI report scanner integration testing. "
        "Certified Synthetic Test Specimen • MediNova Quality Assurance Core",
        ParagraphStyle('Foot', parent=styles['Normal'], fontSize=7.5, textColor=colors.HexColor('#64748b'))
    )
    story.append(footer_text)

    doc.build(story)
    print(f"  [OK] Generated {out_path} ({os.path.getsize(out_path):,} bytes)")


def main():
    print("Generating synthetic demo lab reports for all 5 modules...")
    for filename, data in REPORTS_DATA.items():
        build_pdf_report(filename, data)
    print("All synthetic reports successfully generated in static/samples/.")


if __name__ == "__main__":
    main()
