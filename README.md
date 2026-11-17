# Macro Stress Testing on Loan Defaults

**Do macroeconomic conditions known at origination improve retail loan default prediction beyond the borrower's application data?**

ES 335 Machine Learning, IIT Gandhinagar

| Author | Roll no. | Contribution |
|---|---|---|
| Kunal Kashyap | 25110177 | Data acquisition (notebook Phases 1–2); final-model evaluation on the 2015 holdout, bootstrap and error analysis (Phase 6); parameter and result export, browser-parity check (Phase 7) |
| Anish Hongekar | 25110031 | Target definition and censoring audit (Phase 3); filtering, point-in-time merge, preprocessing, feature-vs-label analysis (Phase 4); cross-validated model selection (Phase 5); website design, content and implementation |

| | |
|---|---|
| Project website | **https://siti-digital.github.io/Loan-Default-Risk-Predictor/** |
| Full pipeline with outputs | [`macro_stress_testing.ipynb`](macro_stress_testing.ipynb) |
| Every result table | [`reports/results/`](reports/results/) |
| Figures | [`reports/figures/`](reports/figures/) |
| Command-line scoring tool | [`predict.py`](predict.py) + [`loan_default_predictor.bin`](loan_default_predictor.bin) |

---

## 1. Problem

| Item | Definition |
|---|---|
| Question | Does adding national macro indicators known at origination (unemployment rate, its 6-month change, CPI inflation) improve default prediction over a model that uses only borrower application data? |
| One row | One 36-month LendingClub personal loan, as it looked in its issue month. |
| Target | `default = 1` for `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off`; `0` for `Fully Paid` and its credit-policy twin. Loans without a final status are excluded. |
| Prediction time | The issue month. Inputs: application fields plus macro data published before that month (2-month lag). |
| Output | Probability of default over the loan's life. |
| Scope of the claim | Predictive comparison on one US lender, 2007–2015, approved loans only. Coefficients are associations, not causal effects. |

## 2. Headline results

Evaluated once on the untouched 2015 vintage (283,026 loans, default rate 14.89%). CV = mean of the 2013 and 2014 validation folds.

| Model | Inputs | PR-AUC | ROC-AUC | Brier | Mean PD | CV PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| A · baseline | training default rate | 0.1489 | 0.5000 | 0.1270 | 13.16% | 0.1303 |
| B · logistic regression | borrower | 0.2211 | 0.6387 | 0.1231 | 13.78% | 0.1882 |
| **C · logistic regression** | **borrower + macro** | **0.2207** | **0.6382** | **0.1233** | **13.09%** | **0.1888** |
| D-base · LightGBM | borrower | 0.2262 | 0.6458 | 0.1227 | 13.75% | 0.1923 |
| D · LightGBM | borrower + macro | 0.2258 | 0.6441 | 0.1226 | 15.48% | 0.1892 |
| Lender benchmark | borrower + `int_rate`, `installment` | 0.2621 | 0.6848 | 0.1210 | 12.12% | 0.2130 |

Model C − Model B, 95% month-block bootstrap (500 draws, 12 monthly clusters):

| Difference | Estimate | 95% interval | Reading |
|---|---:|---|---|
| PR-AUC | −0.00038 | [−0.00086, +0.00011] | no detectable difference |
| ROC-AUC | −0.00049 | [−0.00096, −0.00002] | B slightly better |
| Brier | +0.00025 | [+0.00014, +0.00040] | B slightly better |

**Conclusions**

1. Borrower data carry the signal (PR-AUC 0.221 vs 0.149 no-skill). Adding the macro block does not improve ranking or calibration on 2015, on the CV years, or with non-linear trees (D vs D-base).
2. Out of time, the macro terms push the predicted default rate the **wrong way** in all three years (2013, 2014, 2015). The monthly macro shift correlates −0.73 with the borrower model's miss. Falling unemployment told the model risk was falling, while LendingClub's later vintages defaulted more.
3. The lender benchmark shows the missing information is borrower-level: missed defaults look like good borrowers on our fields but were charged ~2 points more interest.
4. With realistic label timing (training only on loans whose outcome was known by Jan 2015), PR-AUC drops to 0.198 for both B and C, and C's calibration gets much worse (mean PD 22.1% vs 14.9% actual).

## 3. Data and access

| Source | What | Access |
|---|---|---|
| LendingClub accepted loans 2007–2018Q4 | 2,260,701 loans × 151 columns, ~1.6 GB | Kaggle dataset [`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club), free account. Not redistributed here. |
| FRED (Federal Reserve Bank of St. Louis) | Monthly `UNRATE`, `CPIAUCSL`, `FEDFUNDS`, `UMCSENT`, `DGS10`, 2007–2018 | Free API key from https://fred.stlouisfed.org/docs/api/api_key.html, stored in a git-ignored `.env` as `FRED_API_KEY=...` |

Only `UNRATE`, `UNRATE_change_6m` and `CPI_YoY_Inflation` enter the models. Credentials and raw data are excluded by `.gitignore`; no keys are in the history.

**Data-quality issues found and how they are handled** (details in the notebook, Phase 4):

- **Right-censoring:** 28% / 60% / 88% of 2016 / 2017 / 2018 loans are unresolved, so only vintages up to Dec 2015 are used (0.05% unresolved).
- **`mort_acc`** is not recorded for 2007–2011 and part of 2012 (6% of the cohort, 11% of training rows). It is median-imputed on training rows.
- **Income:** heavy right tail (113 loans above $1M, skew 45), so `log1p` is applied. **DTI:** 5 loans at 999 (placeholder), all in 2015, kept.
- 161 Jun–Aug 2007 loans are dropped because the 6-month unemployment change plus the 2-month lag needs 8 months of macro history.

## 4. Setup and reproduction

```bash
git clone https://github.com/Siti-digital/Loan-Default-Risk-Predictor.git
cd Loan-Default-Risk-Predictor
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. **Loans:** download `accepted_2007_to_2018Q4.csv.gz` from Kaggle, decompress it and save it as `data/raw/microdata.csv`. Then run
   `python src/microdatafilterscript.py`, which writes `data/processed/microdataprocessed.csv` (15 application-time columns plus outcome).
2. **Macro:** put `FRED_API_KEY=...` in `.env`, then run `python src/build_macrodata.py`, which writes `data/processed/macrodata.csv`.
3. **Optional censoring audit:** `python src/maturity_audit.py` regenerates `reports/maturity_audit_table.csv` and `reports/figures/loan_maturity_censoring.png`.
4. **Pipeline:** run `macro_stress_testing.ipynb` top to bottom (about 10 minutes on a laptop; the 500-draw bootstrap takes about 4). Or run it headless:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace macro_stress_testing.ipynb
   ```
   This rewrites every table in `reports/results/`, every figure in `reports/figures/`, `loan_default_predictor.bin`, and the website data file `docs/assets/data/site-data.js`.
5. **Website locally:** `python -m http.server --directory docs 8000`, then open http://localhost:8000.
6. **CLI scoring:** `python predict.py`

Randomness: every model and the bootstrap use seed 42.

## 5. Method: exact splits and preprocessing

- **Cohort:** 36-month loans, issued Sep 2007 – Dec 2015, resolved outcomes only, one exact duplicate removed → **620,860 loans** (13.94% default).
- **Point-in-time macro merge** ([`src/macro_processing.py`](src/macro_processing.py)): CPI YoY and the 6-month unemployment change are computed on raw monthly data, then all macro columns are shifted by **2 months** and joined on issue month. The code asserts `issue_date > economic_reference_date` for every row.
- **Features:** borrower = `annual_inc` (log1p), `dti`, `fico_avg`, `open_acc`, `pub_rec`, `revol_util`, `mort_acc`, `loan_amnt`, `home_ownership` (one-hot); macro = `UNRATE`, `UNRATE_change_6m`, `CPI_YoY_Inflation`. Lender benchmark adds `int_rate`, `installment` (`grade`/`sub_grade` are not in the processed extract).
- **Leakage control:** median imputation, scaling and encoding are inside a scikit-learn `ColumnTransformer`, fitted on the training window of each fold and on 2007–2014 for the final models. [`src/validation_checks.py`](src/validation_checks.py) runs six integrity checks.
- **Split (fixed before modelling, [`src/config.py`](src/config.py)):** train 2007–2014 (337,834 loans); expanding-window CV inside it: fold 1 = train ≤ 2012 → validate 2013 (100,422), fold 2 = train ≤ 2013 → validate 2014 (162,570); test = 2015 (283,026), used once.
- **Model selection on CV only:** raw vs log income; logistic `C` ∈ {0.01, 0.1, 1}; four LightGBM settings. Each is selected by mean CV PR-AUC → `C = 0.01` for B, C and the lender benchmark; LightGBM D-base 15 leaves / depth 5 / 200 trees, D 31 leaves / depth 6 / 200 trees (lr 0.03).
- **No class re-weighting** (keeps PDs calibrated). The `balanced` variant is reported as a diagnostic: mean PD 47.7%, Brier 0.235.
- **Metrics:** PR-AUC (primary), ROC-AUC, Brier, calibration-in-the-large and slope. Confusion matrices use the **training** base rate (13.16%) as the threshold.
- **Inference:** statsmodels Logit for Model C with standard errors clustered on 88 issue months. Model C − B differences use a month-block bootstrap on 2015.
- **Extra experiments:** label-availability embargo (train on loans issued ≤ 2011), macro-only model, `dti × UNRATE` / `fico × UNRATE` interactions, balanced weights.

## 6. Limitations

- Only 88 training months and 12 test months of macro variation; the bootstrap with 12 clusters is rough.
- 2015 was a calm year, so the test cannot show whether macro data help in a downturn.
- Approved loans only. LendingClub's underwriting changed over the cycle, so macro coefficients mix the economy with who was approved.
- Macro at origination only. Conditions during repayment would need a hazard model with time-varying covariates.
- National series; state-level unemployment (`addr_state`) is the natural next step.
- The main split trains on outcomes not yet observable at the start of 2015 (see the embargo experiment). FRED values are today's vintage, not first releases.

## 7. Repository map

```text
macro_stress_testing.ipynb      Full pipeline, Phases 1-7, executed with outputs
predict.py                      Interactive CLI scoring with the saved models
loan_default_predictor.bin      joblib bundle of fitted pipelines (written by Phase 7)
requirements.txt                Pinned package versions
src/
  config.py                     Feature groups, split dates, CV folds, grids, seeds (config/__init__.py mirrors it)
  microdatafilterscript.py      Raw LendingClub CSV -> 15-column processed extract
  build_macrodata.py            FRED API -> data/processed/macrodata.csv
  macro_processing.py           Stationary transforms, 2-month lag, point-in-time merge
  validation_checks.py          Six leakage / integrity checks
  maturity_audit.py             Right-censoring audit table and figure
config/experiment.yaml          Human-readable record of the frozen design
reports/
  results/                      All result tables (CSV) + site_data.json
  figures/                      All figures used on the website
  maturity_audit_table.csv      Censoring audit by vintage
  pre_phase4_final_audit.md     Design audit written before modelling
  Macro_Stress_Testing_Master_Defense_Guide.pdf   Personal viva notes from an earlier run (numbers superseded by the notebook)
docs/                           GitHub Pages site (index.html, assets/css, assets/js, assets/data, assets/figures)
.github/workflows/deploy-pages.yml   Publishes docs/ to GitHub Pages
```

## 8. AI-use disclosure

- **Google Antigravity (Gemini-based agent):** scaffolding of `src/` modules, the first website version, documentation formatting.
- **Anthropic Claude (Claude Code), November 2026:** audit of the repository against the rubric. It found and fixed:
  - inconsistent 1- vs 2-month lag code;
  - a decision threshold set from the test-set default rate;
  - a LightGBM "tuning" claim with no tuning code;
  - README entries for files that did not exist.

  It also added the CV-based model selection, data-quality checks, error analysis, the embargo experiment, the result exports, and rebuilt the website. Visual design tokens came from the DesignMD catalogue (a fintech design system); the Higgsfield connector was tried but has no generation capability, so no AI-generated imagery is used.
- **Author's own work:** problem formulation and scope, choice of data and the maturity cut-off, the point-in-time and calibration principles, final modelling decisions, reading and checking all outputs, and the interpretation and conclusions.

## 9. Citations

- LendingClub Corporation, accepted loan data 2007–2018Q4, via Kaggle (N. George, `wordsforthewise/lending-club`).
- U.S. Bureau of Labor Statistics, *Unemployment Rate* [UNRATE] and *CPI for All Urban Consumers* [CPIAUCSL], retrieved from FRED, Federal Reserve Bank of St. Louis.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3).
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414–427.
- Merton, R. C. (1974). On the pricing of corporate debt. *Journal of Finance*, 29(2), 449–470.
- Basel Committee on Banking Supervision, *International Convergence of Capital Measurement and Capital Standards* (IRB approach).
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python. *Proc. 9th Python in Science Conference*.
- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 30*.
- Chart.js (MIT licence) for the website charts.

Contact: kunalkashyap2603@gmail.com
