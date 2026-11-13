from reportlab.platypus import Paragraph, Spacer, PageBreak, Table, TableStyle
from styles import styles, p, bullet, num_bullet, callout, divider, GRAY_BG, GRAY_BORDER, BLUE, NAVY, WHITE

def get_chapter2_story():
    story = []
    story.append(Paragraph("Chapter 2: Financial & Econometric Foundations — Institutional Credit Analytics", styles['ChapterTitle']))
    story.append(Paragraph("From Merton's Structural Theory to López de Prado's Financial Machine Learning and Baesens' Validation Framework", styles['CoverSubtitle']))
    story.append(divider())

    story.append(p(
        "To defend this project before institutional risk managers (such as quantitative teams at Barclays, J.P. Morgan, or Goldman Sachs) "
        "or academic professors, one must speak the language of modern mathematical finance. This chapter establishes the rigorous theoretical "
        "architecture underpinning credit risk modeling, regulatory stress testing, and financial machine learning."
    ))

    story.append(Paragraph("2.1 The Pillar Equations of Credit Risk: EL, LGD, EAD, and PD", styles['SectionTitle']))
    story.append(p(
        "When an institutional bank holds a portfolio of $N$ loans, the fundamental equation that determines its required regulatory capital is "
        "the <b>Expected Loss (EL)</b> formula:"
    ))
    story.append(callout(
        "The Fundamental Expected Loss Formula",
        [
            "<b>EL = PD × LGD × EAD</b><br/><br/>"
            "&bull; <b>PD (Probability of Default):</b> The statistical probability that the obligor will fail to service debt obligations within a specified time window ($P(Y=1 | X) \in [0, 1]$). <i>This project focuses 100% on modeling PD.</i><br/>"
            "&bull; <b>LGD (Loss Given Default):</b> The percentage of the loan balance that is permanently unrecoverable after collateral liquidation and recovery legal costs (typically 40% to 70% in unsecured consumer credit; Recovery Rate = 1 - LGD).<br/>"
            "&bull; <b>EAD (Exposure at Default):</b> The gross dollar amount outstanding at the moment of default, including remaining principal and accrued unpaid interest."
        ],
        kind='note'
    ))
    story.append(p(
        "While Expected Loss determines the standard loan loss provisions that a bank deducts from earnings, it is the <b>Unexpected Loss (UL)</b> "
        "—the volatile tail distribution of defaults during recessions—that threatens a bank's solvency. To protect against Unexpected Loss, "
        "regulators require banks to maintain <b>Economic Capital</b> calculated using Value-at-Risk (VaR) under the Vasicek asymptotic single-risk factor model."
    ))

    story.append(Paragraph("2.2 Probability Calibration vs. Arbitrary Classification Cutoffs", styles['SectionTitle']))
    story.append(p(
        "In undergraduate computer science classes, machine learning is often taught as binary classification where models predict 0 or 1 "
        "using an arbitrary threshold ($\tau = 0.50$). Techniques like SMOTE (Synthetic Minority Over-sampling Technique) or heavy class re-weighting "
        "are often recommended to 'balance' the dataset. <b>In quantitative credit risk, this is strictly forbidden.</b>"
    ))
    story.append(callout(
        "Why We Reject SMOTE and Class-Weighting (The Calibration Imperative)",
        [
            "Banks do not need a binary label; banks price loans using continuous probabilities. If a borrower has a true 4.0% PD, "
            "the bank charges an interest rate to cover that 4.0% expected loss. If an engineer applies SMOTE or `class_weight='balanced'`, "
            "the model artificially inflates predicted default probabilities from 13% to 50%! "
            "The bank would misprice loans, reject creditworthy prime borrowers, and collapse in competitive markets. "
            "Therefore, throughout this project, we enforce <b>unweighted modeling (`class_weight=None`)</b> to preserve the true empirical base-rate calibration."
        ],
        kind='warning'
    ))

    story.append(Paragraph("2.3 Regulatory Stress Testing: Basel II/III, CCAR, and DFAST", styles['SectionTitle']))
    story.append(p(
        "Following the 2007–2008 Global Financial Crisis, global banking authorities recognized that looking only at borrower credit snapshots during economic booms "
        "leads to severe under-capitalization. When the economy is growing, everyone's credit score looks acceptable. But when systematic shocks hit, defaults cluster simultaneously."
    ))
    story.append(p(
        "This led to two major regulatory frameworks:"
    ))
    story.append(bullet("<b>Basel II / III Internal Ratings-Based (IRB) Approach:</b> Mandates that financial institutions estimate Through-the-Cycle (TTC) and Point-in-Time (PIT) probability of default parameters. Capital requirements are scaled dynamically against portfolio risk-weighted assets."))
    story.append(bullet("<b>Federal Reserve CCAR & DFAST Stress Testing:</b> The Federal Reserve annually forces major bank holding companies to submit their loan portfolios to severe forward-looking macroeconomic stress scenarios. These scenarios include sharp unemployment spikes (e.g., UNRATE jumping to 10%), GDP contraction, and housing market deflation to ensure banks have sufficient capital cushions to survive deep recessions."))

    story.append(Paragraph("2.4 Merton's Structural Model of Default (1974)", styles['SectionTitle']))
    story.append(p(
        "The theoretical foundation connecting macroeconomic variables to default probabilities was formulated by Nobel laureate Robert C. Merton in 1974. "
        "In Merton's structural model, a borrower's equity is treated as a European call option on their total economic assets $V_t$."
    ))
    story.append(p(
        "Let the borrower's total asset value follow a geometric Brownian motion: "
        "$$dV_t = \mu V_t dt + \sigma V_t dW_t$$ "
        "where $V_t$ represents the borrower's wealth, income stream, and home value, $\mu$ is asset drift, and $\sigma$ is asset volatility. "
        "The borrower owes a total contractual debt barrier $D$ due at time $T$. Default occurs if and only if:"
    ))
    story.append(p("<b>Default Event:</b> $V_T < D$"))
    story.append(p(
        "Using the Black-Scholes-Merton framework, the theoretical Distance-to-Default ($DD$) is:"
        "$$DD = \frac{\ln(V_0 / D) + (\mu - \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}$$ "
        "And the structural Probability of Default is $\Phi(-DD)$."
    ))
    story.append(callout(
        "Connecting Merton's Math to Our Macro Project Levers",
        [
            "&bull; <b>Borrower Micro-Data (FICO, Income, DTI):</b> Determines the initial asset cushion $V_0 / D$. A high income and low DTI creates a massive buffer above the default barrier $D$.<br/>"
            "&bull; <b>Macroeconomic Shocks (UNRATE, CPI):</b> Drastically reduces asset drift $\mu$ (wages freeze or jobs are lost) and spikes volatility $\sigma$ (uncertainty in living expenses). When $\mu$ drops and $\sigma$ rises, Distance-to-Default ($DD$) collapses, pulling the borrower into the default zone."
        ],
        kind='green'
    ))

    story.append(Paragraph("2.5 Marcos López de Prado: 'Advances in Financial Machine Learning'", styles['SectionTitle']))
    story.append(p(
        "In his seminal 2018 treatise <i>Advances in Financial Machine Learning</i>, Marcos López de Prado established that applying standard computer science machine learning "
        "techniques to financial datasets leads to catastrophic false backtest discoveries due to two fatal violations:"
    ))
    story.append(num_bullet("1", "<b>The Overlapping Label Problem:</b> In standard ML (e.g., MNIST digit recognition), each image is completely independent. In financial lending, a loan originated in June 2012 has an outcome evaluated over a 36-month horizon ending in June 2015! If a researcher performs random K-Fold cross-validation, observations in the training fold will overlap in calendar time with observations in the test fold. The model implicitly 'learns the future' because the economic conditions surrounding the test loan's performance are already baked into the training set!"))
    story.append(num_bullet("2", "<b>Serial Correlation & Information Leakage:</b> Macroeconomic time series have long memory and high autocorrelation. If a test set evaluates loans from 2013, including 2014 loans in the training set violates causality. López de Prado mandates <b>Purging</b> (removing overlapping outcomes) and <b>Embargoing</b> (enforcing time gaps between partitions)."))

    story.append(callout(
        "How Our Pipeline Implements López de Prado's Principles",
        [
            "1. <b>Strict Chronological Out-of-Time (OOT) Split:</b> We completely reject random K-fold CV. We train strictly on loans from <b>2007 through 2014</b> and evaluate on the untouched <b>2015 cohort</b>.<br/>"
            "2. <b>Maturity Purging (Right-Censoring Elimination):</b> Loans originated in 2016–2018 have not finished their 36-month lifespan. Including them would introduce massive survival bias. We purge all uncompleted vintages from our evaluation set.<br/>"
            "3. <b>2-Month Publication Lag Embargo:</b> Macro data is shifted back 2 months to reflect realistic publication delays, preventing look-ahead leakage."
        ],
        kind='tip'
    ))

    story.append(Paragraph("2.6 Bart Baesens: 'Credit Risk Analytics' — Discrimination vs. Calibration", styles['SectionTitle']))
    story.append(p(
        "In <i>Credit Risk Analytics: The R and Python Guide</i> (Baesens, Rösch, & Scheule), model evaluation is strictly partitioned into two complementary dimensions:"
    ))
    story.append(bullet("<b>Discrimination (Ranking Power):</b> Does the model rank higher-risk borrowers above lower-risk borrowers? Measured by <b>ROC-AUC</b> and <b>PR-AUC (Precision-Recall AUC)</b>. Because credit defaults are an imbalanced minority class (~13–15%), ROC-AUC can be overly optimistic due to the huge volume of true negatives. PR-AUC is far more sensitive to false alarms."))
    story.append(bullet("<b>Calibration (Probabilistic Accuracy):</b> Do the estimated probabilities match observed default frequencies? Measured by the <b>Brier Score</b> ($BS = \frac{1}{N}\sum (\hat{p}_i - y_i)^2$) and <b>Reliability Diagrams</b>. An uncalibrated model with high ROC-AUC is useless for bank capital provisioning because its dollar loss estimates will be completely wrong."))

    story.append(Paragraph("2.7 Lender Pricing Endogeneity: The Structural Separation of Models B & C", styles['SectionTitle']))
    story.append(p(
        "A critical econometric nuance in this project is the treatment of <b>interest rate (`int_rate`)</b>, <b>monthly installment (`installment`)</b>, and <b>loan grade (`grade`)</b>."
    ))
    story.append(p(
        "Why did we exclude `int_rate` and `grade` from Model B (Borrower-Only) and Model C (Borrower + Macro)? Because of <b>Lender Pricing Endogeneity</b>! "
        "LendingClub assigns interest rates and grades using its own internal credit scoring model. "
        "If an external researcher includes `grade` and `int_rate` as input features to predict default, they are not predicting the borrower's fundamental credit risk; "
        "they are merely predicting LendingClub's pricing formula! This circularity masks whether macroeconomic factors provide true marginal information. "
        "To preserve scientific purity, we isolate `int_rate` and `grade` into a separate <b>Lender Benchmark</b> model."
    ))
    story.append(PageBreak())
    return story
