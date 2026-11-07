# Project Methodology & Statistical Rigor Roadmap

**Course:** ES 335 Machine Learning | IIT Gandhinagar  
**Project:** Macro-Aware Credit Risk: Does the Local / Systematic Economy at Origination Improve Default Prediction?  
**Dataset Coverage:** 2007 – 2018 (LendingClub Micro-data & FRED Macroeconomic Series)

---

## 1. Executive Summary: Core Methodological Upgrades

Following an econometric and machine learning integrity audit, this project enforces foundational design principles aligned with banking stress-testing standards (Basel III / CCAR) and the ES 335 course rubric:

1. **Elimination of Right-Censoring (Phase 4):** Restrict analysis strictly to matured **36-month installment loans** with completed outcomes (`Fully Paid` vs `Charged Off`). Ongoing `Current` loans and unresolved late loans are dropped after empirical maturity cutoff verification.
2. **Optimal Chronological Out-of-Time (OOT) Split (Phase 5):** Replaces arbitrary random or post-2016 splits. Trains on **2007–2014** (capturing the 2008–2010 Great Recession macroeconomic shock) and tests on **2015** (fully matured by the 2018 dataset cutoff with 0% censoring). Expanding-window CV validates on 2013 and 2014.
3. **Probability Calibration Preservation (Phase 5):** Discards `class_weight='balanced'` in favor of **unweighted modeling (`class_weight=None`)** to protect continuous Probability of Default (PD) calibration, reliability curves, and Brier score loss.
4. **Separation of Risk Tiers & Clean Feature Groups:** Excludes lender-priced variables (`int_rate`, `installment`, `grade`, `sub_grade`) from borrower-only baselines (Model B) and macro-augmented models (Model C & D) to evaluate pure origination prediction without endogenous lender pricing.

---

## 2. Phase-by-Phase Actionable Changes

### Phase 4: Cohort Selection & Pre-Origination Feature Engineering

* **Change 1 — 36-Month Matured Cohort Isolation & Target Resolution:**
  * **Problem:** LendingClub loans are 36 or 60 months long. The dataset ends in late 2018. If 60-month loans or post-2015 loans are tested after dropping `Current` loans, the remaining records suffer severe right-censoring bias (only early defaults/prepayments remain; 2017 is 59.88% unresolved).
  * **Implementation:**
    * Filter loans strictly to: `term.str.contains('36')`.
    * Include genuine completed outcomes:
      - Default ($Y=1$): `Charged Off`, `Default` (40 genuine pre-charge-off default loans), `Does not meet the credit policy. Status:Charged Off` (761 legacy loans).
      - Fully Paid ($Y=0$): `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid` (1,988 legacy loans).
    * Explicitly drop right-censored unresolved loans: `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)`.
* **Pre-Origination Feature Construction:**
  * Compute average FICO score: $\text{FICO} = \frac{\text{fico\_range\_low} + \text{fico\_range\_high}}{2}$.
  * Macroeconomic predictors (stationary, 2-month publication lag): `UNRATE`, `UNRATE_change_6m`, `CPI_YoY_Inflation`.
  * **Interaction Policy:** Ad-hoc nonlinear interaction terms (`dti_x_inflation`, `income_to_unemp`, `installment_x_unemp`) are **rejected from the primary baseline** to avoid multicollinearity and arbitrary feature inflation. Primary models use clean linear main effects; economically motivated interactions (`DTI × UNRATE`, `FICO × UNRATE`) are preserved strictly as optional ablation experiments.
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
| **Cohort Integrity** | Right-censoring of incomplete loans | Filter to 36-month tenure; maturity cutoff fixed at end of 2015 (2017 rejected due to 60% censoring). |
| **Validation Design** | Look-ahead temporal data leakage | Chronological OOT split (2007–2014 Train vs 2015 Holdout Test). |
| **Economic Variation** | Training during zero-variance macro conditions | Training window incorporates the 2008–2009 recession shock. |
| **Probability Calibration** | Distorted PDs from artificial class balancing | `class_weight=None`; calibrated base-rate probability outputs. |
| **Feature Endogeneity** | Confounding borrower risk with lender pricing | `int_rate`, `installment`, `grade`, `sub_grade` excluded from Models B, C, D; isolated in Lender Benchmark. |
| **Interaction Policy** | Arbitrary product inflation & collinearity | Primary models use clean main effects; `DTI × UNRATE` and `FICO × UNRATE` evaluated strictly as optional ablations. |
