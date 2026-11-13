from reportlab.platypus import Paragraph, Spacer, PageBreak, Table, TableStyle
from styles import styles, p, bullet, num_bullet, callout, code_block, divider, GRAY_BG, GRAY_BORDER, BLUE, NAVY, WHITE, LIGHT_BLUE, LIGHT_RED

def get_chapter5_part2_story():
    story = []
    story.append(Paragraph("Chapter 5 (Continued): Line-by-Line Code Breakdown — Phases 5 to 7", styles['ChapterTitle']))
    story.append(Paragraph("Model Training, Econometric Inference, Holdout Testing, and Static Deployment Export", styles['CoverSubtitle']))
    story.append(divider())

    # =========================================================================
    # PHASE 5
    # =========================================================================
    story.append(Paragraph("5.5 Phase 5: Expanding-Window Validation & Econometric Inference (Cell 12)", styles['SectionTitle']))
    story.append(p("<b>Notebook Cell Index:</b> Cell 12 | <b>Type:</b> Code | <b>Execution Count:</b> 17 (218 lines)"))

    story.append(p(
        "Phase 5 implements the chronological expanding-window cross-validation strategy, "
        "fits all six model architectures on the frozen 2007–2014 training partition (337,834 loans), "
        "and estimates the econometric Model C using `statsmodels.Logit` with <b>month-clustered standard errors</b>."
    ))

    cell12_snippet1 = """# Chronological Expanding-Window Partitions
train_mask = (df['issue_date'] >= '2007-01-01') & (df['issue_date'] <= '2014-12-31')
test_mask  = (df['issue_date'] >= '2015-01-01') & (df['issue_date'] <= '2015-12-31')

train_df = df[train_mask].copy().reset_index(drop=True)
test_df  = df[test_mask].copy().reset_index(drop=True)

# Expanding-Window Folds for Validation:
# Fold 1: Train <= 2012-12-31 (74,842 loans) -> Val 2013-01-01 to 2013-12-31 (100,422 loans)
# Fold 2: Train <= 2013-12-31 (175,264 loans) -> Val 2014-01-01 to 2014-12-31 (162,570 loans)"""
    story.append(code_block(cell12_snippet1))
    story.append(p(
        "<b>Line-by-Line Rationale:</b> "
        "1) `train_df` is strictly frozen to originations from 2007 through 2014. "
        "2) `test_df` (the 2015 cohort) is isolated and locked away—it is NEVER touched during model training, tuning, or feature selection. "
        "3) Inside `train_df`, we establish two chronological validation folds. In Fold 1, the model is trained on 2007–2012 and validated on 2013. "
        "In Fold 2, the window expands to train on 2007–2013 and validate on 2014. This strictly respects temporal causality."
    ))

    cell12_snippet2 = """# Econometric Model C: statsmodels.Logit with Month-Clustered Covariance
X_train_c_prep = prep_c.fit_transform(train_df)
X_train_c_prep = sm.add_constant(X_train_c_prep)

# Identify 88 calendar month clusters (e.g., 2007-06, 2007-07, ..., 2014-12)
month_clusters = train_df['issue_date'].dt.to_period('M').astype(str)

# Fit Logit with robust cluster covariance
logit_c = sm.Logit(train_df['default'], X_train_c_prep)
res_c = logit_c.fit(
    cov_type='cluster',
    cov_kwds={'groups': month_clusters},
    disp=False
)"""
    story.append(code_block(cell12_snippet2))
    story.append(callout(
        "The Econometric Defense: Why Month-Clustered Standard Errors?",
        [
            "In your oral viva, a quantitative professor will ask: <i>'Why did you use `cov_type=\"cluster\"` instead of standard OLS/maximum likelihood standard errors?'</i><br/><br/>"
            "<b>The Winning Econometric Defense:</b> Loans originated in the same month share the identical national unemployment rate and CPI inflation. "
            "Therefore, the residuals within each calendar month are correlated (non-I.I.D. error structure). "
            "Traditional standard errors assume independent observations; they artificially deflate standard errors by $\\sqrt{N}$ and produce wildly inflated z-scores (e.g., z = 25!), "
            "creating a false illusion of statistical significance. "
            "Using <b>month-clustered standard errors</b> across the 88 monthly clusters clusters variance at the group level, "
            "yielding true asymptotic standard errors and unbiased hypothesis tests."
        ],
        kind='tip'
    ))

    story.append(Paragraph("<b>Table 5.1: Econometric Model C Regression Summary (statsmodels.Logit, 88 Monthly Clusters)</b>", styles['SubsectionTitle']))
    logit_headers = [
        Paragraph("<b>Standardized Feature</b>", styles['TableHeader']),
        Paragraph("<b>Coefficient (β)</b>", styles['TableHeader']),
        Paragraph("<b>Clustered SE</b>", styles['TableHeader']),
        Paragraph("<b>z-stat</b>", styles['TableHeader']),
        Paragraph("<b>p-value</b>", styles['TableHeader']),
        Paragraph("<b>Odds Ratio (e^β)</b>", styles['TableHeader']),
    ]
    logit_rows = [
        ["const (Intercept)", "-1.8954", "0.0127", "-149.70", "0.0000", "0.1503"],
        ["num__annual_inc", "-0.2426", "0.0248", "-9.77", "0.0000", "0.7846"],
        ["num__dti", "+0.1478", "0.0107", "+13.81", "0.0000", "1.1593"],
        ["num__fico_range_low", "-0.4027", "0.0096", "-42.11", "0.0000", "0.6685"],
        ["num__open_acc", "+0.0121", "0.0072", "+1.68", "0.0923", "1.0122"],
        ["num__pub_rec", "-0.0029", "0.0078", "-0.37", "0.7107", "0.9971"],
        ["num__revol_util", "-0.0164", "0.0068", "-2.41", "0.0159", "0.9838"],
        ["num__mort_acc", "-0.0841", "0.0066", "-12.73", "0.0000", "0.9194"],
        ["num__loan_amnt", "+0.0998", "0.0082", "+12.12", "0.0000", "1.1049"],
        ["num__UNRATE (Static)", "-0.0349", "0.0120", "-2.92", "0.0035", "0.9657"],
        ["num__UNRATE_change_6m (Shock)", "+0.0769", "0.0101", "+7.61", "0.0000", "1.0799"],
        ["num__CPI_YoY_Inflation", "+0.0585", "0.0113", "+5.17", "0.0000", "1.0603"],
        ["cat__home_MORTGAGE", "-0.1751", "0.0154", "-11.38", "0.0000", "0.8394"],
        ["cat__home_OTHER", "+0.2592", "0.2164", "+1.20", "0.2312", "1.2958"],
        ["cat__home_OWN", "-0.0798", "0.0170", "-4.70", "0.0000", "0.9233"],
    ]
    t_logit = [logit_headers]
    for r in logit_rows:
        is_macro = "UNRATE" in r[0] or "CPI" in r[0]
        st = 'TableCellBold' if is_macro else 'TableCell'
        t_logit.append([Paragraph(c, styles[st]) for c in r])
    table_logit = Table(t_logit, colWidths=[162, 75, 75, 70, 70, 80])
    table_logit.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('GRID', (0,0), (-1,-1), 0.5, GRAY_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('BACKGROUND', (0, 10), (-1, 12), LIGHT_BLUE), # Macro rows highlighted
    ]))
    story.append(table_logit)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Table 5.2: Mean Cross-Validation Performance Across Expanding Folds 1 & 2</b>", styles['SubsectionTitle']))
    cv_headers = [
        Paragraph("<b>Model Architecture</b>", styles['TableHeader']),
        Paragraph("<b>Calibration Setting</b>", styles['TableHeader']),
        Paragraph("<b>PR-AUC</b>", styles['TableHeader']),
        Paragraph("<b>ROC-AUC</b>", styles['TableHeader']),
        Paragraph("<b>Brier Score</b>", styles['TableHeader']),
        Paragraph("<b>Mean Pred PD</b>", styles['TableHeader']),
        Paragraph("<b>Calibration Error</b>", styles['TableHeader']),
    ]
    cv_rows = [
        ["MODEL_A", "baseline base-rate", "0.13026", "0.50000", "0.11333", "12.83%", "-0.00201"],
        ["MODEL_B", "unweighted (clean)", "0.18634", "0.62542", "0.11125", "14.21%", "+0.01182"],
        ["MODEL_C (Chosen)", "unweighted (clean)", "0.18685", "0.62598", "0.11138", "14.13%", "+0.01106"],
        ["LENDER_BENCHMARK", "unweighted (pricing)", "0.21283", "0.66335", "0.10975", "13.94%", "+0.00913"],
        ["MODEL_C_ABLATION", "unweighted (interact)", "0.18754", "0.62683", "0.11131", "14.31%", "+0.01286"],
        ["MODEL_C_BALANCED_CHECK", "balanced_diagnostic", "0.18601", "0.62527", "0.25885", "50.79%", "+0.37767"],
        ["MODEL_D_BASE", "unweighted (LightGBM)", "0.19198", "0.63411", "0.11072", "13.90%", "+0.00875"],
        ["MODEL_D", "unweighted (LightGBM)", "0.19047", "0.63155", "0.11091", "14.80%", "+0.01771"],
    ]
    t_cv = [cv_headers]
    for r in cv_rows:
        is_bad = "BALANCED" in r[0]
        st = 'TableCellBold' if is_bad else 'TableCell'
        t_cv.append([Paragraph(c, styles[st]) for c in r])
    table_cv = Table(t_cv, colWidths=[130, 95, 60, 60, 62, 65, 60])
    table_cv.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('GRID', (0,0), (-1,-1), 0.5, GRAY_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('BACKGROUND', (0, 6), (-1, 6), LIGHT_RED), # Balanced check highlighted in red
    ]))
    story.append(table_cv)
    story.append(Spacer(1, 8))

    story.append(callout(
        "Empirical Proof: The Destruction of Calibration under `class_weight='balanced'`",
        [
            "Look at the red row in Table 5.2 above (`MODEL_C_BALANCED_CHECK`). "
            "When `class_weight='balanced'` was applied, the model predicted an average default rate of <b>50.79%</b> on a portfolio where the actual default rate was only <b>13.03%</b>! "
            "This resulted in a massive calibration error of <b>+37.77%</b> and blew the Brier Score up from <b>0.111 to 0.259</b> (more than double!). "
            "This empirical test provides undeniable mathematical proof of why we enforced `class_weight=None`."
        ],
        kind='warning'
    ))

    # =========================================================================
    # PHASE 6
    # =========================================================================
    story.append(Paragraph("5.6 Phase 6: Holdout Evaluation & Statistical Uncertainty (Cell 14)", styles['SectionTitle']))
    story.append(p("<b>Notebook Cell Index:</b> Cell 14 | <b>Type:</b> Code | <b>Execution Count:</b> 19 (188 lines)"))

    story.append(p(
        "Phase 6 loads the pristine, untouched 2015 holdout dataset (283,026 completed loans), "
        "generates continuous default probabilities from all fitted models, and computes the complete battery of evaluation metrics."
    ))

    cell14_snippet1 = """# Phase 6: Out-of-Time Test Scoring on 2015 Holdout
y_test = test_df['default'].values
results_holdout = []

for model_name, clf in fitted_models.items():
    prep = fitted_preprocessors[model_name]
    X_test_prep = prep.transform(test_df)
    
    # Predict continuous default probabilities
    if hasattr(clf, 'predict_proba'):
        y_pred_proba = clf.predict_proba(X_test_prep)[:, 1]
    else: # Model A baseline
        y_pred_proba = np.full(len(y_test), baseline_train_pd)
        
    roc = roc_auc_score(y_test, y_pred_proba)
    prauc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    mean_pd = np.mean(y_pred_proba)
    
    results_holdout.append({
        'Model': model_name,
        'ROC-AUC': roc,
        'PR-AUC': prauc,
        'Brier_Score': brier,
        'Mean_Pred_PD': mean_pd,
        'Actual_DR': np.mean(y_test)
    })"""
    story.append(code_block(cell14_snippet1))
    story.append(p(
        "<b>Line-by-Line Rationale:</b> "
        "1) `prep.transform(test_df)` applies the training-fitted imputation medians and scaling parameters out-of-sample (zero test leakage). "
        "2) `clf.predict_proba(...)[:, 1]` extracts continuous Probability of Default ($p_i \in [0, 1]$). "
        "3) Computes PR-AUC (`average_precision_score`), ROC-AUC (`roc_auc_score`), and probability calibration via Brier score loss (`brier_score_loss`)."
    ))

    cell14_snippet2 = """# Month-Level Block Bootstrapping (B = 500 Repetitions)
# Preserves monthly macroeconomic cluster structure during uncertainty estimation
test_months = test_df['issue_date'].dt.to_period('M').unique()
G = len(test_months) # G = 12 calendar months in 2015

boot_deltas_prauc = []
for b in range(500):
    # Resample calendar months with replacement
    sampled_months = np.random.choice(test_months, size=G, replace=True)
    boot_indices = np.concatenate([month_to_idx[m] for m in sampled_months])
    
    pr_b = average_precision_score(y_test[boot_indices], preds_b[boot_indices])
    pr_c = average_precision_score(y_test[boot_indices], preds_c[boot_indices])
    boot_deltas_prauc.append(pr_c - pr_b)

ci_lower = np.percentile(boot_deltas_prauc, 2.5)
ci_upper = np.percentile(boot_deltas_prauc, 97.5)
print(f'95% CI for PR-AUC Lift (Model C - Model B): [{ci_lower:+.5f}, {ci_upper:+.5f}]')"""
    story.append(code_block(cell14_snippet2))
    story.append(callout(
        "Why Block Bootstrapping by Month?",
        [
            "Standard bootstrap draws individual loan rows with replacement. "
            "However, if loans in the same month have correlated default shocks, drawing individual rows breaks the macro correlation structure! "
            "<b>Month-Level Block Bootstrapping</b> resamples entire calendar months (all 20,000+ loans originated in January 2015 together). "
            "This preserves the real-world clustering of macroeconomic shocks across the 500 Monte Carlo repetitions."
        ],
        kind='note'
    ))

    # =========================================================================
    # PHASE 7
    # =========================================================================
    story.append(Paragraph("5.7 Phase 7: Model Export & Parity Verification (Cell 16)", styles['SectionTitle']))
    story.append(p("<b>Notebook Cell Index:</b> Cell 16 | <b>Type:</b> Code | <b>Execution Count:</b> 21 (155 lines)"))

    story.append(p(
        "Phase 7 bridges the Python machine learning research pipeline to the production client-side web application. "
        "It exports the fitted parameters of Model C into JSON format and tests exact mathematical parity against the JavaScript sigmoid formula."
    ))

    cell16_snippet = """# Export Model C Scalers and Coefficients to JSON
prep_c = fitted_preprocessors['MODEL_C']
clf_c  = fitted_models['MODEL_C']

scaler_num = prep_c.named_transformers_['num'].named_steps['scaler']
num_cols   = prep_c.transformers_[0][2]
means_dict = dict(zip(num_cols, scaler_num.mean_))
stds_dict  = dict(zip(num_cols, scaler_num.scale_))

# Map learned Logistic Regression weights
feat_names_c = list(prep_c.get_feature_names_out())
weights_dict = dict(zip(feat_names_c, clf_c.coef_[0]))
intercept    = float(clf_c.intercept_[0])

export_bundle = {
    'model_id': 'Model_C_Logit',
    'intercept': intercept,
    'weights': weights_dict,
    'variable_means': means_dict,
    'variable_stds': stds_dict
}

with open('web/model_coefficients.json', 'w') as f:
    json.dump(export_bundle, f, indent=2)
print('Successfully exported Model C parameters to web/model_coefficients.json.')"""
    story.append(code_block(cell16_snippet))
    story.append(p(
        "<b>Line-by-Line Rationale:</b> "
        "1) Extracts the exact empirical means (`scaler_num.mean_`) and standard deviations (`scaler_num.scale_`) learned on the 2007–2014 training partition. "
        "2) Extracts the fitted coefficient vector $\\beta$ and intercept $\\beta_0$. "
        "3) Serializes the bundle to `web/model_coefficients.json` so that client-side JavaScript can evaluate the formula without any backend server runtime."
    ))

    story.append(callout(
        "Phase 7 Parity Verification Output",
        [
            "<b>Output:</b> <code>[SUCCESS] Exact numerical parity verified within 10^-5 tolerance across all scenarios!</code><br/>"
            "<b>Significance:</b> Proves that a user adjusting sliders on GitHub Pages receives the exact identical default probability score "
            "that Python's Scikit-Learn / statsmodels would produce on the server."
        ],
        kind='tip'
    ))
    story.append(PageBreak())
    return story
