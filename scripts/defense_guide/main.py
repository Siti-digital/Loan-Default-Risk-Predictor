import os
import sys

# Ensure local scripts/ package is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors

from styles import (
    styles, p, bullet, num_bullet, callout, divider, NumberedCanvas,
    NAVY, BLUE, LIGHT_BLUE, BORDER_BLUE, GRAY_BG, GRAY_BORDER, TEXT_DARK, TEXT_MUTED, WHITE
)
from ch1_intuition import get_chapter1_story
from ch2_financial_foundations import get_chapter2_story
from ch3_variables import get_chapter3_story
from ch4_maturity_audit import get_chapter4_story
from ch5_code_walkthrough_part1 import get_chapter5_part1_story
from ch5_code_walkthrough_part2 import get_chapter5_part2_story
from ch6_metrics_and_evaluation import get_chapter6_story
from ch7_deployment_and_web_export import get_chapter7_story
from ch8_readme_walkthrough import get_chapter8_story
from ch9_interview_qa import get_chapter9_story
from ch10_glossary_formulas import get_chapter10_story

def build_cover_page():
    story = []
    story.append(Spacer(1, 40))
    story.append(Paragraph("ES 335: Machine Learning &bull; Master Defense Guide", styles['CoverSubtitle']))
    story.append(Paragraph("Macro-Aware Credit Risk & Systematic Stress Testing on Retail Loan Defaults", styles['CoverTitle']))
    story.append(divider())
    story.append(Spacer(1, 10))

    meta_text = """
    <b>Author & Model Developer:</b> Kunal Kashyap<br/>
    <b>Email:</b> kunalkashyap2603@gmail.com &bull; <b>Contact:</b> +91 7577815607<br/>
    <b>Academic Affiliation:</b> Indian Institute of Technology Gandhinagar (Palaj, Gujarat, PIN 382355)<br/>
    <b>Target Domain:</b> Quantitative Risk & Credit Portfolio Analytics (Barclays, J.P. Morgan, Goldman Sachs)<br/>
    <b>Date of Defense:</b> November 2026 &bull; Final Course Submission
    """
    story.append(Paragraph(meta_text, styles['CoverMeta']))
    story.append(Spacer(1, 15))

    overview_box = [
        "<b>DOCUMENT PURPOSE & ARCHIVAL SCOPE:</b><br/>"
        "This master reference document provides an exhaustive, institutional-grade mathematical and econometric record of the entire loan default stress-testing project. "
        "It is engineered specifically for oral thesis defense, rigorous professor viva grilling, and institutional quantitative credit risk interviews.<br/><br/>"
        "&bull; <b>Part I:</b> Intuitive foundations designed for clear conceptual understanding (explaining credit risk to a 12-year-old).<br/>"
        "&bull; <b>Part II:</b> Deep theoretical finance: Merton's Structural Model, Basel II/III IRB capital adequacy, CCAR/DFAST stress testing, Marcos López de Prado's <i>Advances in Financial Machine Learning</i> (purging & embargoing), and Bart Baesens' <i>Credit Risk Analytics</i> (calibration vs discrimination).<br/>"
        "&bull; <b>Part III:</b> Complete line-by-line syntax and econometric breakdown of every single code block in <code>macro_stress_testing.ipynb</code> (Phases 1 to 7).<br/>"
        "&bull; <b>Part IV:</b> Deep metric analysis (PR-AUC vs ROC-AUC, Brier score, block bootstrapping) and 15 tough viva questions with model answers."
    ]
    story.append(callout("Executive Defense Summary", overview_box, kind='note'))
    story.append(Spacer(1, 15))

    # Table of Contents
    story.append(Paragraph("<b>Master Table of Contents:</b>", styles['SectionTitle']))
    toc_data = [
        [Paragraph("<b>Chapter</b>", styles['TableHeader']), Paragraph("<b>Title & Core Subject Matter</b>", styles['TableHeader'])],
        [Paragraph("Chapter 1", styles['TableCellBold']), Paragraph("The Intuitive Foundation: Credit Risk & Systematic Shocks for a 12-Year-Old", styles['TableCell'])],
        [Paragraph("Chapter 2", styles['TableCellBold']), Paragraph("Financial & Econometric Foundations: Merton, López de Prado, & Baesens", styles['TableCell'])],
        [Paragraph("Chapter 3", styles['TableCellBold']), Paragraph("Exhaustive Variable Dictionary: Every Macro & Micro Feature Demystified", styles['TableCell'])],
        [Paragraph("Chapter 4", styles['TableCellBold']), Paragraph("The 2.26 Million Loan Maturity & Right-Censoring Audit (Why 2016–2018 were rejected)", styles['TableCell'])],
        [Paragraph("Chapter 5 (Part 1)", styles['TableCellBold']), Paragraph("Line-by-Line Code Breakdown: Phases 1 to 4 (Setup, Data, Cleaning, Pipelines)", styles['TableCell'])],
        [Paragraph("Chapter 5 (Part 2)", styles['TableCellBold']), Paragraph("Line-by-Line Code Breakdown: Phases 5 to 7 (Training, Logit SEs, Holdout Test, Export)", styles['TableCell'])],
        [Paragraph("Chapter 6", styles['TableCellBold']), Paragraph("Deep Dive into Model Evaluation Outputs, Calibration Curves, and Bootstrap CIs", styles['TableCell'])],
        [Paragraph("Chapter 7", styles['TableCellBold']), Paragraph("Phase 7 Model Export & Static Zero-Server GitHub Pages Deployment Architecture", styles['TableCell'])],
        [Paragraph("Chapter 8", styles['TableCellBold']), Paragraph("Detailed Walkthrough of the README.md Repository Blueprint & Rubric Mapping", styles['TableCell'])],
        [Paragraph("Chapter 9", styles['TableCellBold']), Paragraph("The Master Interview & Viva Defense Toolkit (Top 15 Questions with Model Answers)", styles['TableCell'])],
        [Paragraph("Chapter 10", styles['TableCellBold']), Paragraph("Technical Glossary & Formula Cheat-Sheet (Full LaTeX Formulas & Acronyms)", styles['TableCell'])],
    ]
    toc_table = Table(toc_data, colWidths=[100, 432])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('GRID', (0,0), (-1,-1), 0.5, GRAY_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(toc_table)

    story.append(PageBreak())
    return story

def main():
    project_root = os.path.dirname(parent_dir)
    output_pdf = os.path.join(project_root, "reports", "Macro_Stress_Testing_Master_Defense_Guide.pdf")
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    print(f"Generating Master Defense Guide PDF at: {output_pdf}...")

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=45
    )

    story = []
    print("Building Cover Page...")
    story.extend(build_cover_page())

    print("Building Chapter 1 (Intuition for 12-Year-Old)...")
    story.extend(get_chapter1_story())

    print("Building Chapter 2 (Financial & Econometric Foundations)...")
    story.extend(get_chapter2_story())

    print("Building Chapter 3 (Exhaustive Variable Dictionary)...")
    story.extend(get_chapter3_story())

    print("Building Chapter 4 (2.26M Loan Maturity Audit)...")
    story.extend(get_chapter4_story())

    print("Building Chapter 5 Part 1 (Code Walkthrough Phases 1-4)...")
    story.extend(get_chapter5_part1_story())

    print("Building Chapter 5 Part 2 (Code Walkthrough Phases 5-7)...")
    story.extend(get_chapter5_part2_story())

    print("Building Chapter 6 (Evaluation Outputs & Bootstrap CIs)...")
    story.extend(get_chapter6_story())

    print("Building Chapter 7 (Web Deployment & Parameter Export)...")
    story.extend(get_chapter7_story())

    print("Building Chapter 8 (README.md Walkthrough)...")
    story.extend(get_chapter8_story())

    print("Building Chapter 9 (Interview & Viva Toolkit)...")
    story.extend(get_chapter9_story())

    print("Building Chapter 10 (Technical Glossary & Formulas)...")
    story.extend(get_chapter10_story())

    print("Compiling document layout with NumberedCanvas...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Master Defense Guide PDF generated successfully! ({os.path.getsize(output_pdf):,} bytes)")

if __name__ == "__main__":
    main()
