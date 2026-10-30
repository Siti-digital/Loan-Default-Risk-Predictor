# Project Methodology & Statistical Rigor Roadmap

**Course:** ES 335 Machine Learning | IIT Gandhinagar  
**Project:** Macroeconomic Stress Testing on Retail Loan Defaults  
**Dataset Coverage:** 2007 – 2018 (LendingClub Micro-data & FRED Macroeconomic Series)

---

## 1. Executive Summary: Core Methodological Upgrades

Following an econometric and machine learning integrity audit, this project enforces three foundational design principles aligned with banking stress-testing standards (Basel III / CCAR) and the ES 335 course rubric:

1. **Elimination of Right-Censoring (Phase 4):** Restrict analysis strictly to matured **36-month installment loans** with completed outcomes (`Fully Paid` vs `Charged Off`). Ongoing `Current` loans are dropped only after maturity cutoff verification.
2. **Optimal Chronological Out-of-Time (OOT) Split (Phase 5):** Replaces arbitrary random or post-2016 splits. Trains on **2007–2014** (capturing the 2008–2010 Great Recession macroeconomic shock) and tests on **2015** (fully matured by the 2018 dataset cutoff with 0% censoring).
3. **Probability Calibration Preservation (Phase 5):** Discards `class_weight='balanced'` in favor of **unweighted modeling (`class_weight=None`)** to protect continuous Probability of Default (PD) calibration, reliability curves, and Brier score loss.

---

## 2. Phase-by-Phase Actionable Changes

### Phase 4: Cohort Selection & Pre-Origination Feature Engineering

* **Change 1 — 36-Month Matured Cohort Isolation:**
  * **Problem:** LendingClub loans are 36 or 60 months long. The dataset ends in late 2018. If 60-month loans or post-2015 loans are tested after dropping `Current` loans, the remaining records suffer severe right-censoring bias (only early defaults/prepayments remain).
  * **Implementation:**
    * Filter loans strictly to: `term.str.contains('36')`.
    * Retain only terminal loan outcomes: `loan_status.isin(['Fully Paid', 'Charged Off'])`.
    * Map binary target: $Y = 1$ for `Charged Off` (Default) and $Y = 0$ for `Fully Paid` (Non-Default).
* **Pre-Origination Feature Construction:**
  * Compute average FICO score: $\text{FICO} = \frac{\text{fico\_range\_low} + \text{fico\_range\_high}}{2}$.
  * Macro-Borrower stress interaction features:
    * Debt pressure under inflation: $\text{dti\_x\_inflation} = \text{dti} \times \text{CPI\_YoY\_Inflation}$
    * Income cushion under unemployment: $\text{income\_to\_unemp} = \frac{\text{annual\_inc}}{\text{UNRATE}}$
    * Debt service vs labor market: $\text{installment\_x\_unemp} = \text{installment} \times \text{UNRATE}$
  * Natural log transforms on skewed monetary variables: $\ln(1 + \text{annual\_inc})$, $\ln(1 + \text{loan\_amnt})$.

---

### Phase 5: Chronological Out-of-Time (OOT) Split & Model Training

* **Change 2 — Frozen Chronological Partitioning (2007–2014 Train / 2015 Test):**
  * **Problem:** Splitting randomly leaks future economic conditions into the past. Testing on 2017 loans causes massive right-censoring because 2017 loans had not reached 36-month maturity by late 2018.
  * **Implementation:**
    * **Training Set ($t_{\text{issue}} \le \text{2014-12-31}$):** Spans 2007 through 2014. Exposes the model to diverse macroeconomic regimes, specifically the 2008–2010 Great Recession (when national unemployment peaked at 10.0%) and the subsequent recovery.
    * **Test Set ($\text{2015-01-01} \le t_{\text{issue}} \le \text{2015-12-31}$):** All 36-month loans issued in 2015 reached full maturity before the dataset terminated in December 2018, guaranteeing **0% censoring bias and 0% look-ahead leakage**.
    * Scalers (`StandardScaler`) and imputers (`SimpleImputer(strategy='median')`) are fitted **strictly on the 2007–2014 training partition** and applied out-of-sample to the 2015 test set.

* **Change 3 — Unweighted Calibration Preservation:**
  * **Problem:** Setting `class_weight='balanced'` artificially shifts the logistic intercept to predict an ungrounded ~50% default probability. In credit risk, models must output the true, calibrated base-rate Probability of Default (PD ~15–20%) required for risk-weighted assets and Brier score evaluation.
  * **Implementation:**
    * Train `LogisticRegression(class_weight=None)` and econometric `statsmodels.api.Logit`.
    * Address class imbalance methodologically using **PR-AUC (Precision-Recall Area Under Curve)** and **Brier Score ($BS$)**:
      $$BS = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
    * Evaluate probability calibration using reliability diagrams (`sklearn.calibration.calibration_curve`).

---

## 3. Validation Matrix Summary

| Rubric Dimension | Methodological Flaw Prevented | Verified Mechanism |
| :--- | :--- | :--- |
| **Cohort Integrity** | Right-censoring of incomplete loans | Filter to 36-month tenure; maturity cutoff fixed at end of 2015. |
| **Validation Design** | Look-ahead temporal data leakage | Chronological OOT split (2007–2014 Train vs 2015 Test). |
| **Economic Variation** | Training during zero-variance macro conditions | Training window incorporates the 2008–2009 recession shock. |
| **Probability Calibration** | Distorted PDs from artificial class balancing | `class_weight=None`; calibrated base-rate probability outputs. |
