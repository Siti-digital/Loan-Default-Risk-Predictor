from reportlab.platypus import Paragraph, Spacer, PageBreak, Table, TableStyle
from styles import styles, p, bullet, num_bullet, callout, divider, GRAY_BG, GRAY_BORDER, BLUE, NAVY, WHITE, LIGHT_GREEN, LIGHT_RED

def get_chapter6_story():
    story = []
    story.append(Paragraph("Chapter 6: Deep Dive into Model Evaluation Outputs & Scorecards", styles['ChapterTitle']))
    story.append(Paragraph("Understanding PR-AUC, ROC-AUC, Brier Calibration, Confusion Matrices, and Block Bootstrap CIs", styles['CoverSubtitle']))
    story.append(divider())

    story.append(p(
        "A central requirement of your oral defense is being able to explain the exact output table generated in Phase 6. "
        "Professors and interviewers will scrutinize the column headers (PR-AUC, ROC-AUC, Brier Score, Mean Pred PD vs Actual DR), "
        "the differences between models, and what the final numbers reveal about credit risk."
    ))

    story.append(Paragraph("6.1 The 2015 Out-of-Time Holdout Scorecard", styles['SectionTitle']))
    story.append(p(
        "Table 6.1 displays the official evaluation results across all primary models and benchmarks on the <b>283,026 completed loans</b> "
        "originated during calendar year 2015 (Empirical Default Rate = 14.8863%):"
    ))

    # Benchmark Table
    headers = [
        Paragraph("<b>Model Name</b>", styles['TableHeader']),
        Paragraph("<b>Feature Set Description</b>", styles['TableHeader']),
        Paragraph("<b>PR-AUC</b>", styles['TableHeader']),
        Paragraph("<b>ROC-AUC</b>", styles['TableHeader']),
        Paragraph("<b>Brier Score</b>", styles['TableHeader']),
        Paragraph("<b>Mean Pred PD</b>", styles['TableHeader']),
        Paragraph("<b>Actual DR</b>", styles['TableHeader']),
    ]

    raw_data = [
        ["MODEL_A", "Historical Baseline (Uninformed)", "0.1489", "0.5000", "0.1270", "13.15%", "14.89%"],
        ["MODEL_B", "Borrower-Only Logit (FICO, DTI, Inc)", "0.2224", "0.6380", "0.1231", "13.86%", "14.89%"],
        ["MODEL_C (Chosen)", "Borrower + Macro Logit (UNRATE, CPI)", "0.2218", "0.6375", "0.1234", "13.11%", "14.89%"],
        ["MODEL_D_BASE", "LightGBM on Borrower Features", "0.2262", "0.6457", "0.1227", "13.76%", "14.89%"],
        ["MODEL_D", "LightGBM on Borrower + Macro", "0.2240", "0.6421", "0.1228", "15.69%", "14.89%"],
        ["LENDER_BENCHMARK", "Lender Pricing (int_rate, grade)", "0.2663", "0.6863", "0.1209", "12.08%", "14.89%"],
        ["MODEL_C_ABLATION", "Model C + Interaction Terms", "0.2214", "0.6369", "0.1233", "13.37%", "14.89%"],
    ]

    table_rows = [headers]
    for row in raw_data:
        is_model_c = "MODEL_C (Chosen)" in row[0]
        row_cells = []
        for c in row:
            st = 'TableCellBold' if is_model_c else 'TableCell'
            row_cells.append(Paragraph(c, styles[st]))
        table_rows.append(row_cells)

    scorecard_table = Table(table_rows, colWidths=[95, 140, 55, 55, 58, 65, 64])
    scorecard_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('GRID', (0,0), (-1,-1), 0.5, GRAY_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0, 3), (-1, 3), LIGHT_GREEN), # Model C row
    ]))

    story.append(scorecard_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("6.2 Detailed Metric-by-Metric Deconstruction", styles['SectionTitle']))

    story.append(Paragraph("1. PR-AUC (Precision-Recall Area Under Curve)", styles['SubsectionTitle']))
    story.append(p(
        "<b>What it measures:</b> The area under the curve plotting Precision (y-axis) against Recall (x-axis) across all classification thresholds.<br/>"
        "<b>Baseline Reference:</b> For a completely random, uninformed classifier, PR-AUC equals the <b>empirical base rate of the minority class</b>. "
        "Notice Model A achieves a PR-AUC of exactly <b>0.1489</b>, which matches the 2015 default rate of 14.89%!<br/>"
        "<b>Model B & C Performance:</b> Model B achieves <b>0.2224</b> and Model C achieves <b>0.2218</b>. "
        "This represents a massive <b>+49.0% relative improvement</b> over random baseline guessing!"
    ))

    story.append(Paragraph("2. ROC-AUC (Receiver Operating Characteristic AUC)", styles['SubsectionTitle']))
    story.append(p(
        "<b>What it measures:</b> The probability that the model will assign a higher predicted default probability to a randomly selected defaulter "
        "than to a randomly selected non-defaulter ($P(\\hat{p}_{\\text{def}} > \\hat{p}_{\\text{paid}})$).<br/>"
        "<b>Interpretation:</b> Model A scores <b>0.5000</b> (a coin flip). "
        "Model B scores <b>0.6380</b> and Model C scores <b>0.6375</b>. "
        "In unsecured retail credit, ROC-AUC values between 0.63 and 0.66 are standard institutional performance benchmarks when lender pricing features are excluded."
    ))

    story.append(Paragraph("3. Brier Score Loss (Probabilistic Calibration Error)", styles['SubsectionTitle']))
    story.append(p(
        "<b>Formula:</b> $BS = \\frac{1}{N}\\sum_{i=1}^N (\\hat{p}_i - y_i)^2$<br/>"
        "<b>What it measures:</b> Mean squared error between predicted continuous probabilities and binary outcomes ($y_i \\in \\{0, 1\\}$). "
        "A lower Brier score represents superior calibration. Model B scores <b>0.1231</b> and Model C scores <b>0.1234</b>, "
        "improving significantly upon Model A's 0.1270."
    ))

    story.append(Paragraph("4. Mean Predicted PD vs. Actual Default Rate", styles['SubsectionTitle']))
    story.append(p(
        "<b>What it measures:</b> Portfolio-level calibration. Model B predicts an average default probability of <b>13.86%</b> "
        "and Model C predicts <b>13.11%</b> against the actual 2015 default rate of <b>14.89%</b>. "
        "Both models are well-calibrated without wild under- or over-estimation."
    ))

    story.append(Paragraph("6.3 Classification Metrics at the Default-Prevalence Threshold (τ = 14.89%)", styles['SectionTitle']))
    story.append(p(
        "Rather than using an arbitrary $\\tau = 0.50$ threshold (which would classify almost no one as defaulting because defaults are only 15% of loans), "
        "we evaluate binary decision metrics at the <b>empirical default prevalence cutoff</b> $\\tau = 14.89\\%$. "
        "At this cutoff, any borrower with predicted $PD \\ge 14.89\\%$ is flagged as high-risk."
    ))
    story.append(callout(
        "Model B Confusion Matrix on 283,026 Loans",
        [
            "&bull; <b>True Negatives (TN):</b> 146,069 loans (correctly identified as safe)<br/>"
            "&bull; <b>False Positives (FP):</b> 94,825 loans (false alarms: flagged as high-risk but repaid)<br/>"
            "&bull; <b>False Negatives (FN):</b> 17,179 loans (missed defaults: expected to repay but defaulted)<br/>"
            "&bull; <b>True Positives (TP):</b> 24,953 loans (successfully caught defaults)<br/><br/>"
            "&bull; <b>Recall (Sensitivity):</b> <code>TP / (TP + FN) = 24,953 / 42,132 = 59.23%</code>. The model catches nearly 60% of all eventual defaults!<br/>"
            "&bull; <b>Precision:</b> <code>TP / (TP + FP) = 24,953 / 119,778 = 20.83%</code>. When the model flags a loan, 1 in 5 actually defaults (a 40% lift over the 14.9% base rate)."
        ],
        kind='note'
    ))

    story.append(Paragraph("6.4 Month-Level Block Bootstrap Confidence Intervals (B = 500 Repetitions)", styles['SectionTitle']))
    story.append(p(
        "To rigorously test whether Model C (Borrower + Macro) is statistically superior to Model B (Borrower-Only), "
        "we ran 500 Monte Carlo block bootstrap resamples across calendar month clusters. The results were:"
    ))
    story.append(bullet("<b>PR-AUC Lift (Model C - Model B):</b> Point estimate = <code>-0.00060</code>; 95% Confidence Interval = <code>[-0.00125, +0.00006]</code>."))
    story.append(bullet("<b>ROC-AUC Lift (Model C - Model B):</b> Point estimate = <code>-0.00052</code>; 95% Confidence Interval = <code>[-0.00171, +0.00068]</code>."))

    story.append(callout(
        "The Core Scientific Conclusion of the Entire Research Project",
        [
            "Notice that the 95% Confidence Interval for discrimination lift spans zero (crosses from negative to positive). "
            "<b>Conclusion 1 (Cross-Sectional Discrimination):</b> During a calm, steady macroeconomic expansion year (such as 2015, where national unemployment hovered stably between 5.0% and 5.7%), "
            "incorporating national macroeconomic indicators at loan origination does NOT provide statistically significant ranking lift over individual borrower credit fundamentals (FICO score and DTI).<br/><br/>"
            "<b>Conclusion 2 (Systematic Baseline Drift Correction):</b> However, macro indicators are essential for <i>baseline drift correction</i> across multi-year credit cycles (e.g., distinguishing between 2008 and 2015) and for regulatory stress-testing simulations."
        ],
        kind='amber'
    ))
    story.append(PageBreak())
    return story
