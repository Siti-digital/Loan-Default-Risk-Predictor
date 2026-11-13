from reportlab.platypus import Paragraph, Spacer, PageBreak, Table, TableStyle
from styles import styles, p, bullet, num_bullet, callout, divider, GRAY_BG, GRAY_BORDER, BLUE, NAVY, WHITE

def get_chapter10_story():
    story = []
    story.append(Paragraph("Chapter 10: Technical Glossary & Formula Cheat-Sheet", styles['ChapterTitle']))
    story.append(Paragraph("Comprehensive Formula Compendium, Econometric Derivations, and Acronym Dictionary", styles['CoverSubtitle']))
    story.append(divider())

    story.append(p(
        "This final chapter serves as an instant-lookup technical cheat-sheet for your viva defense. "
        "It compiles every key mathematical formula, statistical estimator, and regulatory acronym into a unified reference guide."
    ))

    story.append(Paragraph("10.1 Key Mathematical & Econometric Formulas", styles['SectionTitle']))

    formulas = [
        (
            "1. Standardized Logistic Logit Formulation",
            "$$z = \\beta_0 + \\sum_{i=1}^k \\beta_i \\left(\\frac{X_i - \\mu_i}{\\sigma_i}\\right)$$",
            "Calculates the linear log-odds index. Each feature X_i is centered by training mean μ_i and scaled by training standard deviation σ_i."
        ),
        (
            "2. Logistic Sigmoid Function (Probability of Default)",
            "$$P(\\text{Default} \\mid \\vec{X}) = \\sigma(z) = \\frac{1}{1 + e^{-z}} = \\frac{e^z}{1 + e^z}$$",
            "Maps the unbounded linear score z ∈ (-∞, +∞) onto a valid continuous probability interval [0, 1]."
        ),
        (
            "3. Odds Ratio (OR) Interpretation",
            "$$\\text{Odds} = \\frac{p}{1 - p} = e^z, \\quad \\text{Odds Ratio}_i = e^{\\beta_i}$$",
            "A 1-standard-deviation increase in feature i multiplies the baseline default odds by e^(β_i). If β_i = +0.146 (DTI), default odds increase by +15.7% (OR = 1.157). If β_i = -0.401 (FICO), default odds decline by 33.1% (OR = 0.669)."
        ),
        (
            "4. Expected Loss (EL) Formula",
            "$$\\text{EL} = \\text{PD} \\times \\text{LGD} \\times \\text{EAD}$$",
            "Determines required portfolio-level accounting loan loss reserves (IFRS 9 / CECL / Basel III)."
        ),
        (
            "5. Brier Score Loss (Calibration Assessment)",
            "$$\\text{BS} = \\frac{1}{N} \\sum_{i=1}^N (\\hat{p}_i - y_i)^2$$",
            "Mean squared error of probabilistic forecasts against binary realized outcomes y_i ∈ {0, 1}. Ranges from 0 (perfect) to 1. Murphy's decomposition splits BS into Reliability - Resolution + Uncertainty."
        ),
        (
            "6. Cluster-Robust Sandwich Covariance Estimator",
            "$$\\text{Var}_{\\text{cluster}}(\\hat{\\beta}) = (X^T W X)^{-1} \\left(\\sum_{g=1}^G X_g^T e_g e_g^T X_g\\right) (X^T W X)^{-1}$$",
            "Computes valid asymptotic standard errors in the presence of intra-cluster correlation across G calendar months, eliminating false significance from grouped macro observations."
        ),
        (
            "7. Vasicek Single-Risk-Factor Portfolio Loss Model",
            "$$P(Y=1 \\mid Z) = \\Phi\\left(\\frac{\\Phi^{-1}(\\text{PD}) - \\sqrt{\\rho} Z}{\\sqrt{1 - \\rho}}\\right)$$",
            "The regulatory engine of Basel II/III IRB capital requirements. Connects idiosyncratic borrower risk (PD) to the systematic macroeconomic shock Z with asset correlation ρ."
        ),
        (
            "8. Merton Distance-to-Default (1974)",
            "$$DD = \\frac{\\ln(V_0 / D) + (\\mu - \\frac{1}{2}\\sigma^2)T}{\\sigma \\sqrt{T}}, \\quad \\text{PD} = \\Phi(-DD)$$",
            "Structural default probability where borrower assets V_t cross liability boundary D."
        )
    ]

    for title, eq, desc in formulas:
        story.append(callout(title, [f"<b>Formula:</b> {eq}<br/><br/><b>Explanation:</b> {desc}"], kind='note'))
        story.append(Spacer(1, 4))

    story.append(Paragraph("10.2 Comprehensive Acronym Dictionary", styles['SectionTitle']))

    acronyms_data = [
        [Paragraph("<b>Acronym</b>", styles['TableHeader']), Paragraph("<b>Full Title</b>", styles['TableHeader']), Paragraph("<b>Domain Definition & Role</b>", styles['TableHeader'])],
        [Paragraph("<b>PD</b>", styles['TableCellBold']), Paragraph("Probability of Default", styles['TableCell']), Paragraph("Continuous estimated probability that an obligor fails to service debt within the time horizon.", styles['TableCell'])],
        [Paragraph("<b>LGD</b>", styles['TableCellBold']), Paragraph("Loss Given Default", styles['TableCell']), Paragraph("Percentage of exposure that is permanently lost if default occurs (1 - Recovery Rate).", styles['TableCell'])],
        [Paragraph("<b>EAD</b>", styles['TableCellBold']), Paragraph("Exposure at Default", styles['TableCell']), Paragraph("Gross dollar amount at risk at the moment default occurs.", styles['TableCell'])],
        [Paragraph("<b>EL / UL</b>", styles['TableCellBold']), Paragraph("Expected / Unexpected Loss", styles['TableCell']), Paragraph("EL is covered by provisions; UL is volatile tail loss covered by regulatory equity capital.", styles['TableCell'])],
        [Paragraph("<b>OOT</b>", styles['TableCellBold']), Paragraph("Out-of-Time", styles['TableCell']), Paragraph("Chronological temporal split where the test partition strictly succeeds the training partition in time.", styles['TableCell'])],
        [Paragraph("<b>PIT</b>", styles['TableCellBold']), Paragraph("Point-in-Time", styles['TableCell']), Paragraph("Evaluation conditioning exclusively on information published and available at origination month t.", styles['TableCell'])],
        [Paragraph("<b>TTC</b>", styles['TableCellBold']), Paragraph("Through-the-Cycle", styles['TableCell']), Paragraph("Long-run average credit rating that strips out macroeconomic cyclical fluctuations.", styles['TableCell'])],
        [Paragraph("<b>CCAR</b>", styles['TableCellBold']), Paragraph("Comprehensive Capital Analysis and Review", styles['TableCell']), Paragraph("US Federal Reserve mandatory forward-looking stress-testing regime for major bank holding companies.", styles['TableCell'])],
        [Paragraph("<b>DFAST</b>", styles['TableCellBold']), Paragraph("Dodd-Frank Act Stress Testing", styles['TableCell']), Paragraph("Statutory capital stress-testing rules established following the 2008 Financial Crisis.", styles['TableCell'])],
        [Paragraph("<b>IRB</b>", styles['TableCellBold']), Paragraph("Internal Ratings-Based Approach", styles['TableCell']), Paragraph("Basel Committee framework allowing banks to use internal mathematical models for regulatory capital.", styles['TableCell'])],
        [Paragraph("<b>DTI</b>", styles['TableCellBold']), Paragraph("Debt-to-Income Ratio", styles['TableCell']), Paragraph("Total monthly non-housing debt obligations divided by gross monthly income.", styles['TableCell'])],
        [Paragraph("<b>FICO</b>", styles['TableCellBold']), Paragraph("Fair Isaac Corporation Score", styles['TableCell']), Paragraph("Standardized consumer credit rating scoring creditworthiness between 300 and 850.", styles['TableCell'])],
        [Paragraph("<b>FRED</b>", styles['TableCellBold']), Paragraph("Federal Reserve Economic Data", styles['TableCell']), Paragraph("Official economic research database maintained by the Federal Reserve Bank of St. Louis.", styles['TableCell'])],
        [Paragraph("<b>PR-AUC</b>", styles['TableCellBold']), Paragraph("Precision-Recall Area Under Curve", styles['TableCell']), Paragraph("Primary discrimination metric for imbalanced classes; random baseline equals empirical base rate.", styles['TableCell'])],
        [Paragraph("<b>ROC-AUC</b>", styles['TableCellBold']), Paragraph("Receiver Operating Characteristic AUC", styles['TableCell']), Paragraph("Ranks defaulters vs non-defaulters; random baseline is 0.50.", styles['TableCell'])],
        [Paragraph("<b>BS</b>", styles['TableCellBold']), Paragraph("Brier Score Loss", styles['TableCell']), Paragraph("Mean squared error assessing absolute probability calibration.", styles['TableCell'])],
    ]

    acr_table = Table(acronyms_data, colWidths=[65, 145, 322])
    acr_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('GRID', (0,0), (-1,-1), 0.5, GRAY_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(acr_table)
    story.append(Spacer(1, 10))

    story.append(p("<b>Defense Readiness Declaration:</b> With mastery of these 10 chapters, the student is fully prepared to defend every econometric, mathematical, and software engineering choice made in this project before any academic or institutional quantitative review board."))
    return story
