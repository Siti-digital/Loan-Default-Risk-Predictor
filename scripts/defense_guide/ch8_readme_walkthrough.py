from reportlab.platypus import Paragraph, Spacer, PageBreak, Table, TableStyle
from styles import styles, p, bullet, num_bullet, callout, divider, GRAY_BG, GRAY_BORDER, BLUE, NAVY, WHITE

def get_chapter8_story():
    story = []
    story.append(Paragraph("Chapter 8: Detailed Walkthrough of the README.md File", styles['ChapterTitle']))
    story.append(Paragraph("Deconstructing the Repository Blueprint, Rubric Mapping, and Documentation Standards", styles['CoverSubtitle']))
    story.append(divider())

    story.append(p(
        "A project's `README.md` is the primary entry point for instructors, teaching assistants, peer reviewers, and technical interviewers. "
        "A mediocre README merely lists install commands. An institutional-grade README provides the theoretical blueprint, data provenance, "
        "reproducibility instructions, and an explicit mapping against evaluation criteria. "
        "This chapter walks through each section of our project's `README.md` and explains why it was designed that way."
    ))

    story.append(Paragraph("8.1 Section-by-Section Anatomy of the README", styles['SectionTitle']))

    story.append(Paragraph("1. Header & Submission Quick Access Links", styles['SubsectionTitle']))
    story.append(p(
        "The very top of the README contains a prominent quick-access block providing the four key deliverables requested by the ES 335 prompt: "
        "the live GitHub Pages URL (`https://siti-digital.github.io/Loan-Default-Risk-Predictor/`), the source repository URL, "
        "the core research notebook (`macro_stress_testing.ipynb`), the interactive CLI tool (`predict.py`), and the pre-trained binary model artifact (`loan_default_predictor.bin`). "
        "This ensures graders can instantly locate every deliverable without scrolling."
    ))

    story.append(Paragraph("2. Executive Summary & Problem Formulation (Section 1)", styles['SubsectionTitle']))
    story.append(p(
        "Establishes the quantitative research question and explicitly defines the <b>unit of analysis</b> (a single originated retail loan), "
        "the <b>target outcome</b> ($Y \\in \\{0, 1\\}$), and the <b>decision horizon</b> (origination time $t_{\\text{issue}}$). "
        "It emphasizes that the project is a <i>predictive comparison</i>, not an econometric causal estimation, avoiding common academic over-claims."
    ))

    story.append(Paragraph("3. Domain Concepts & Econometric Rigor (Section 2)", styles['SubsectionTitle']))
    story.append(p(
        "Documents the five core principles that elevate this project above toy ML assignments: "
        "unweighted base-rate preservation, Point-in-Time 2-month publication lag, maturity auditing, lender pricing endogeneity separation, and time-block inference."
    ))

    story.append(Paragraph("4. Architecture Tree & Pipeline Flowchart (Sections 3 & 4)", styles['SubsectionTitle']))
    story.append(p(
        "Provides an ASCII file tree of all modular scripts (`src/`), configuration files (`config/experiment.yaml`), reports, and static web files, "
        "paired with a Mermaid flowchart illustrating the exact data flow from raw LendingClub CSVs and FRED indicators to the out-of-time evaluation."
    ))

    story.append(Paragraph("5. Setup, Rebuild Steps & Code Location Map (Section 5)", styles['SubsectionTitle']))
    story.append(p(
        "Provides crystal-clear, foolproof reproduction commands for: "
        "1) Activating the Python 3.12 virtual environment (`ai-env`), "
        "2) Acquiring the 2.26M loan dataset from Kaggle, "
        "3) Executing the interactive menu-driven terminal predictor (`python predict.py`), "
        "4) Re-running the 7-phase research notebook in Jupyter Lab, and "
        "5) Opening the static web app locally (`docs/index.html`)."
    ))

    story.append(Paragraph("6. Course Rubric Alignment (Section 6 - 100 Points)", styles['SubsectionTitle']))
    story.append(p(
        "Features a detailed 10-row matrix mapping every rubric dimension from the official ES 335 guidelines directly to specific code implementations and report files. "
        "This makes it effortless for graders to assign maximum marks across all criteria."
    ))

    story.append(Paragraph("7. AI-Use Disclosure & Academic Integrity (Section 7)", styles['SubsectionTitle']))
    story.append(p(
        "Provides full transparency in compliance with IIT Gandhinagar's academic policies: "
        "explicitly disclosing generative AI assistance for boilerplate scaffolding, CSS styling, and markdown formatting, "
        "while clearly identifying the independent human contributions (problem formulation, maturity audit, econometric modeling, and conclusions)."
    ))
    story.append(PageBreak())
    return story
