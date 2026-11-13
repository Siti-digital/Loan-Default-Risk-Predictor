from reportlab.platypus import Paragraph, Spacer, PageBreak
from styles import styles, p, bullet, num_bullet, callout, code_block, divider

def get_chapter7_story():
    story = []
    story.append(Paragraph("Chapter 7: Model Export & Static Deployment Architecture", styles['ChapterTitle']))
    story.append(Paragraph("Converting Model Parameters to JSON for Pure Client-Side GitHub Pages Deployment", styles['CoverSubtitle']))
    story.append(divider())

    story.append(p(
        "A hallmark requirement of the ES 335 course deliverable is providing an interactive web application hosted live on GitHub Pages. "
        "Standard machine learning web applications typically rely on server runtimes (such as Flask, FastAPI, or Docker containers running Python in the cloud). "
        "However, GitHub Pages only serves <b>static files (HTML, CSS, JavaScript, and JSON)</b>. "
        "This chapter explains how we extracted Model C's mathematical parameters to make the entire model run natively inside the user's web browser."
    ))

    story.append(Paragraph("7.1 The Mathematical Bridge: From Scikit-Learn to Browser JS", styles['SectionTitle']))
    story.append(p(
        "A Logistic Regression model is not a mysterious black box; it is an exact, closed-form algebraic formula called the <b>logistic sigmoid function</b>. "
        "For any set of borrower attributes and macroeconomic levers $\\vec{X} = [X_1, X_2, \\dots, X_k]$, the log-odds (logit $z$) is calculated as:"
    ))
    story.append(p(
        "$$z = \\beta_0 + \\sum_{i=1}^k \\beta_i \\left(\\frac{X_i - \\mu_i}{\\sigma_i}\\right)$$"
    ))
    story.append(p(
        "where $\\beta_0$ is the intercept, $\\beta_i$ is the learned regression coefficient for feature $i$, "
        "and $\\mu_i$ and $\\sigma_i$ are the empirical mean and standard deviation learned strictly on the 2007–2014 training partition. "
        "Once the logit $z$ is computed, the continuous Probability of Default is evaluated via:"
    ))
    story.append(p(
        "$$P(\\text{Default}) = \\sigma(z) = \\frac{1}{1 + e^{-z}}$$"
    ))

    story.append(Paragraph("7.2 Parameter Serialization into `model_coefficients.json`", styles['SectionTitle']))
    story.append(p(
        "In Phase 7 (Cell 16 of `macro_stress_testing.ipynb`), Python extracts these exact quantities directly from the fitted `ColumnTransformer` and `LogisticRegression` objects "
        "and saves them into `web/model_coefficients.json`:"
    ))

    json_snippet = """{
  "model_id": "Model_C_Logit",
  "effective_intercept": -1.89598,
  "weights": {
    "fico_score": -0.401139,
    "dti": 0.145846,
    "annual_income": -0.245381,
    "unemployment_rate": -0.034996,
    "cpi_inflation_rate": 0.057986,
    "unemp_change_6m": 0.075759
  },
  "variable_means": {
    "fico_score": 693.385,
    "dti": 17.525,
    "annual_income": 72836.4,
    "unemployment_rate": 7.738,
    "cpi_inflation_rate": 1.865,
    "unemp_change_6m": -0.198
  },
  "variable_stds": {
    "fico_score": 30.134,
    "dti": 7.892,
    "annual_income": 55595.2,
    "unemployment_rate": 1.637,
    "cpi_inflation_rate": 1.346,
    "unemp_change_6m": 0.584
  }
}"""
    story.append(code_block(json_snippet))

    story.append(Paragraph("7.3 Zero-Server Client-Side Inference", styles['SectionTitle']))
    story.append(p(
        "When a user opens the GitHub Pages link, the client's web browser downloads `model_coefficients.json`. "
        "When the user adjusts the UNRATE or FICO slider and clicks <b>Predict Default Risk</b>, JavaScript executes the mathematical formula instantly in memory. "
        "This offers three enormous engineering benefits:"
    ))
    story.append(num_bullet("1", "<b>Zero Hosting Costs:</b> The website requires no expensive GPU/CPU servers, no AWS/GCP bills, and can be hosted permanently for free on GitHub Pages."))
    story.append(num_bullet("2", "<b>Sub-Millisecond Latency:</b> Predictions calculate in 0.001 milliseconds with zero network round-trip delay."))
    story.append(num_bullet("3", "<b>Total Auditability:</b> Every formula weight and scaling parameter is completely transparent and inspectable in JSON."))

    story.append(callout(
        "Numerical Parity Verification",
        [
            "In Phase 7, we built an automated parity test inside the notebook comparing Python's `clf.predict_proba()` against a simulated JavaScript formula across 5 distinct economic scenarios (Calm, Mild Stress, Severe 2008 Recession, Low FICO, High DTI). "
            "The numerical difference between Python and JavaScript was <b>less than 0.000001 (10^-6)</b> across all cases, proving perfect mathematical fidelity."
        ],
        kind='tip'
    ))
    story.append(PageBreak())
    return story
