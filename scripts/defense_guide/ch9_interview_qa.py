from reportlab.platypus import Paragraph, Spacer, PageBreak
from styles import styles, p, bullet, num_bullet, callout, divider

def get_chapter9_story():
    story = []
    story.append(Paragraph("Chapter 9: The Master Interview & Viva Defense Toolkit", styles['ChapterTitle']))
    story.append(Paragraph("Anticipating Tough Technical, Econometric, and Financial Interview Questions with Model Answers", styles['CoverSubtitle']))
    story.append(divider())

    story.append(p(
        "This chapter is your secret weapon for viva examinations, project presentations, and quantitative risk interviews. "
        "It compiles the 15 most challenging, probing questions that professors, risk directors, and machine learning interviewers "
        "are likely to ask about this project, paired with concise, mathematically bulletproof model answers."
    ))

    qa_list = [
        (
            "Q1: Why did you choose Logistic Regression (Model C) as your primary model instead of a deep neural network or XGBoost?",
            "Model interpretability and regulatory compliance (Basel III / SR 11-7 model risk management guidance). "
            "In institutional credit risk, models must provide transparent, additive log-odds that can be converted into adverse action notice codes "
            "under the Equal Credit Opportunity Act (ECOA). Black-box models cannot be deployed in core retail credit without extreme explainability overhead. "
            "Furthermore, in our Phase 6 holdout evaluation, LightGBM (Model D) only improved PR-AUC by +0.002 (0.224 vs 0.222)—a negligible lift that did not justify "
            "the loss of econometric interpretability and closed-form parameter export."
        ),
        (
            "Q2: Why does national unemployment rate (UNRATE) have a negative coefficient (β = -0.035) in Model C? Doesn't high unemployment cause more defaults?",
            "This reflects the 'Flight-to-Quality' underwriting effect during severe recessions. "
            "When national unemployment peaked at 9–10% between 2009 and 2011, bank underwriters drastically tightened credit standards, approving only the most pristine borrowers. "
            "Consequently, loans originated during peak unemployment had higher unobserved cash buffers. "
            "Meanwhile, defaults are truly triggered by acute *economic acceleration*, which is captured by our velocity feature `UNRATE_change_6m` (β = +0.076, p < 0.01). "
            "Static unemployment captures underwriting regime, while velocity captures the actual economic shock!"
        ),
        (
            "Q3: Why did you exclude interest rate, installment, and loan grade from Models B and C?",
            "Lender Pricing Endogeneity! LendingClub assigns interest rates and grades using its own internal credit scoring model. "
            "If an external researcher includes `grade` and `int_rate` as inputs to predict default, they are not predicting the borrower's fundamental credit risk; "
            "they are merely predicting LendingClub's pricing model! This creates circularity and masks whether macroeconomic variables provide true marginal value. "
            "To preserve scientific purity, we isolated `int_rate` and `grade` into a separate Lender Benchmark."
        ),
        (
            "Q4: Why did you reject SMOTE and class re-weighting when defaults are an imbalanced minority class (~15%)?",
            "Because banks price credit using continuous, calibrated probabilities of default (PD), not binary classifications. "
            "Under Basel III and IFRS 9, Expected Loss is calculated as EL = PD × LGD × EAD. "
            "If you apply SMOTE or `class_weight='balanced'`, you artificially inflate predicted default probabilities from 14% to 50%! "
            "The bank would misprice loans, reject safe borrowers, and destroy portfolio profitability. Unweighted modeling (`class_weight=None`) preserves true base-rate calibration."
        ),
        (
            "Q5: What is Right-Censoring, and why did you reject the 2016–2018 cohorts?",
            "Right-censoring occurs when an observation has not completed its full lifespan before the study ends. "
            "A 36-month loan originated in 2017 requires 3 full years (until 2020) to observe its true outcome. "
            "Our empirical audit of all 2.26 million loans revealed that 28.17% of 2016 loans, 59.88% of 2017 loans, and 88.03% of 2018 loans were still active (`Current`). "
            "Dropping `Current` loans creates massive survivor bias (keeping only early catastrophic defaults). "
            "The 2015 cohort was 99.95% completed (only 147 unresolved loans out of 283,173!), making it the single latest pristine out-of-time test cohort."
        ),
        (
            "Q6: Why did you apply a 2-month publication lag to macroeconomic indicators?",
            "To eliminate Point-in-Time (PIT) look-ahead leakage. "
            "Macroeconomic data is released on a reporting delay (June unemployment is published in mid-July; preliminary CPI is revised months later). "
            "A borrower signing a loan on June 2nd cannot be evaluated using June economic data that does not yet exist! "
            "Applying `shift(2)` strictly guarantees that every loan is merged only with economic information published and finalized *prior* to origination."
        ),
        (
            "Q7: Why did you use month-clustered standard errors instead of standard maximum likelihood standard errors?",
            "Loans originated in the same calendar month share the identical national unemployment rate and CPI inflation. "
            "Their residuals are correlated, violating the Independent and Identically Distributed (I.I.D.) assumption. "
            "Standard maximum likelihood assumes independence, dividing variance by N (337,000) and producing wildly deflated standard errors and fake z-scores of 25. "
            "Clustering across the 88 monthly cohorts computes cluster-robust sandwich covariance estimators, yielding honest asymptotic p-values."
        ),
        (
            "Q8: What is the difference between PR-AUC and ROC-AUC, and why is PR-AUC more informative in default prediction?",
            "ROC-AUC plots True Positive Rate vs False Positive Rate. Because non-defaulters outnumber defaulters 6-to-1, "
            "the denominator of False Positive Rate (True Negatives) is massive, which artificially suppresses the false positive rate and makes ROC-AUC look overly optimistic. "
            "PR-AUC plots Precision vs Recall; it focuses strictly on the minority default class and directly penalizes false alarms. "
            "Furthermore, an uninformed random baseline scores PR-AUC equal to the base rate (0.1489 in 2015), providing an absolute reference benchmark."
        ),
        (
            "Q9: What was your final scientific conclusion about whether macro at origination improves default prediction?",
            "During a calm, steady economic expansion (such as 2015, where national unemployment remained stable at 5.0–5.7%), "
            "national macroeconomic conditions at origination provide virtually zero cross-sectional discrimination lift (PR-AUC delta of -0.00060, 95% CI [-0.00125, +0.00006] spanning zero). "
            "Borrower fundamentals (FICO, DTI) overwhelmingly drive who defaults within a given cohort. "
            "However, macro indicators are essential for *baseline drift correction* (raising or lowering capital reserves across multi-year cycles) and for regulatory stress-testing simulations."
        ),
        (
            "Q10: How does your pipeline implement Marcos López de Prado's concepts from 'Advances in Financial Machine Learning'?",
            "Through Purging and Embargoing! "
            "López de Prado warns that standard cross-validation causes severe backtest overfitting in finance due to overlapping outcome windows and serial correlation. "
            "We implemented Purging by auditing loan maturity and discarding uncompleted 2016–2018 cohorts so test labels never overlap with training observation windows. "
            "We implemented Embargoing through our 2-month publication lag shift and chronological expanding-window splits."
        ),
        (
            "Q11: How did you ensure zero data leakage in your preprocessing and scaling pipelines?",
            "By keeping Scikit-Learn transformers (`StandardScaler`, `SimpleImputer`) unfitted during feature definition! "
            "Transformers were fit strictly inside training folds (`fit_transform` on train data only). "
            "Test data was strictly passed through `transform()`. This guaranteed that the test set's mean, standard deviation, and median never leaked into model parameters."
        ),
        (
            "Q12: What is Brier Score, and why does calibration matter more than raw accuracy in banking?",
            "Brier Score is the mean squared difference between predicted continuous probabilities and binary outcomes: BS = (1/N) ∑ (p̂_i - y_i)². "
            "Accuracy measures thresholded 0/1 correctness, which is useless in lending. "
            "Banks use default probabilities to calculate capital reserves, risk-based pricing, and Expected Loss. "
            "If a model has high accuracy but predicts 40% default for borrowers who actually default at 5%, the bank will collapse from mispriced capital."
        ),
        (
            "Q13: Why did you perform month-level block bootstrapping instead of row-level bootstrapping?",
            "Row-level bootstrap draws individual loan records with replacement. "
            "However, if loans in the same calendar month experience a shared macroeconomic shock, drawing individual rows breaks the temporal correlation structure! "
            "Month-level block bootstrapping resamples entire calendar months (all 20,000+ loans from a month together), "
            "preserving the real-world clustering of macroeconomic shocks across the 500 Monte Carlo repetitions."
        ),
        (
            "Q14: How does your web deployment work without a Python backend on GitHub Pages?",
            "By extracting the closed-form logistic regression parameters (intercept β_0, weights β_i, and training means μ_i and standard deviations σ_i) "
            "into `model_coefficients.json`. "
            "The client's web browser downloads the JSON file and evaluates the logistic sigmoid formula p = 1 / (1 + exp(-z)) directly in JavaScript in 0.001 milliseconds, "
            "achieving exact numerical parity within 10^-5 of Python."
        ),
        (
            "Q15: What is the primary limitation of your study?",
            "Two main limitations: "
            "1) Observational credit data is subject to unobserved underwriter discretion (concessionary loan adjustments during stress). "
            "2) Number of macroeconomic test clusters (G = 12): Because the untouched 2015 holdout contains exactly 12 calendar months, "
            "statistical power to detect macro effects in the holdout is constrained by the small number of macroeconomic time steps, "
            "even though the borrower sample size (N = 283,026) is enormous."
        )
    ]

    for question, answer in qa_list:
        story.append(callout(
            question,
            [f"<b>Model Viva / Interview Defense:</b><br/>{answer}"],
            kind='tip' if "Q2" in question or "Q9" in question else 'note'
        ))
        story.append(Spacer(1, 4))

    story.append(PageBreak())
    return story
