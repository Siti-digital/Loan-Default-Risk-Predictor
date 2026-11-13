from reportlab.platypus import Paragraph, Spacer, PageBreak, Table, TableStyle
from styles import styles, p, bullet, num_bullet, callout, code_block, divider, GRAY_BG, GRAY_BORDER, BLUE, NAVY, WHITE

def get_chapter5_part1_story():
    story = []
    story.append(Paragraph("Chapter 5: Line-by-Line Code Breakdown — Phases 1 to 4", styles['ChapterTitle']))
    story.append(Paragraph("Deconstructing the Jupyter Notebook: Syntax, Software Engineering, and Econometric Rationale", styles['CoverSubtitle']))
    story.append(divider())

    story.append(p(
        "In this chapter, we dissect every single code cell in `macro_stress_testing.ipynb` from Phase 1 through Phase 4. "
        "For each cell, we provide the exact Python code, an exhaustive line-by-line breakdown of what every statement executes, "
        "the engineering/econometric reasoning behind it, and an analysis of the cell's actual output."
    ))

    # =========================================================================
    # PHASE 1
    # =========================================================================
    story.append(Paragraph("5.1 Phase 1: Environment Setup & Toolchain Initialization (Cell 2)", styles['SectionTitle']))
    story.append(p("<b>Notebook Cell Index:</b> Cell 2 | <b>Type:</b> Code | <b>Execution Count:</b> 10"))
    
    cell2_code = """import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    brier_score_loss, 
    classification_report, 
    confusion_matrix
)
from sklearn.calibration import calibration_curve

# Set plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', 50)
print('All primary libraries imported successfully.')"""
    story.append(code_block(cell2_code))

    story.append(Paragraph("Line-by-Line Breakdown & Technical Justification:", styles['SubsectionTitle']))
    story.append(bullet("<code>import os, json, numpy as np, pandas as pd</code>: Imports foundational operating system file path handlers, JSON serialization tools, vector math operations, and high-performance tabular data structures (DataFrames)."))
    story.append(bullet("<code>import matplotlib.pyplot as plt, seaborn as sns</code>: Configures data visualization engines used for drawing receiver operating curves, calibration diagrams, and feature correlation heatmaps."))
    story.append(bullet("<code>import statsmodels.api as sm</code>: <b>Critical institutional library.</b> Unlike Scikit-Learn which focuses purely on point predictions, `statsmodels` is an econometric suite that calculates asymptotic p-values, z-statistics, standard errors, and allows custom covariance estimators (such as <b>month-clustered standard errors</b>)."))
    story.append(bullet("<code>from sklearn.linear_model import LogisticRegression</code>: Machine learning baseline classifier used for fast cross-validation iterations."))
    story.append(bullet("<code>from sklearn.preprocessing import StandardScaler, SimpleImputer</code>: Transformers for zero-mean/unit-variance feature scaling and median imputation."))
    story.append(bullet("<code>from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss</code>: Metric evaluation functions specifically chosen for imbalanced credit default prediction and probability calibration assessment."))
    story.append(bullet("<code>plt.style.use(...) & pd.set_option('display.max_columns', 50)</code>: Formats clean visual outputs and prevents Pandas from truncating wide 25-column tables in notebook displays."))

    story.append(callout(
        "Cell 2 Output & Verification",
        [
            "<b>Output:</b> <code>All primary libraries imported successfully.</code><br/>"
            "<b>Verification:</b> Confirms that all statistical, econometric, and machine learning packages are correctly resolved in Python 3.12 without library conflicts."
        ],
        kind='tip'
    ))

    # =========================================================================
    # PHASE 2
    # =========================================================================
    story.append(Paragraph("5.2 Phase 2: Data Procurement (Cells 4 & 5)", styles['SectionTitle']))
    story.append(p("<b>Notebook Cell Index:</b> Cell 4 | <b>Type:</b> Code | <b>Execution Count:</b> 11"))

    cell4_code = """# Phase 2: Load Curated Datasets
# 1. Micro-Data (LendingClub filtered to 15 relevant features)
microdata_path = 'data/processed/microdataprocessed.csv'
print(f'Loading micro-data from {microdata_path}...')
microdata = pd.read_csv(microdata_path, low_memory=False)
print(f'Micro-data loaded: {microdata.shape[0]:,} rows, {microdata.shape[1]} columns.')

# 2. Macroeconomic Series (FRED monthly indicators 2007-2018)
macrodata_path = 'data/processed/macrodata.csv'
print(f'Loading macro-data from {macrodata_path}...')
macrodata = pd.read_csv(macrodata_path)
print(f'Macro-data loaded: {macrodata.shape[0]} months, {macrodata.shape[1]} columns.')"""
    story.append(code_block(cell4_code))

    story.append(Paragraph("Line-by-Line Breakdown & Technical Justification:", styles['SubsectionTitle']))
    story.append(bullet("<code>microdata_path = 'data/processed/microdataprocessed.csv'</code>: Points to the curated micro-data file containing the 15 pre-origination borrower features isolated from LendingClub's raw 2.26 million loan dump."))
    story.append(bullet("<code>pd.read_csv(..., low_memory=False)</code>: Loads the large CSV table into memory. Setting `low_memory=False` instructs Pandas to read the entire file in a single internal buffer, preventing mixed-type column warnings across 2.26M rows."))
    story.append(bullet("<code>macrodata_path = 'data/processed/macrodata.csv'</code>: Points to the 144-month FRED macroeconomic dataset covering the historical period from January 2007 to December 2018."))
    story.append(bullet("<code>f'{microdata.shape[0]:,} rows'</code>: Formats large row counts with readable comma thousand-separators."))

    story.append(callout(
        "Cell 4 Output & Diagnostic Significance",
        [
            "<b>Output:</b><br/>"
            "<code>Loading micro-data from data/processed/microdataprocessed.csv...</code><br/>"
            "<code>Micro-data loaded: 2,260,701 rows, 15 columns.</code><br/>"
            "<code>Loading macro-data from data/processed/macrodata.csv...</code><br/>"
            "<code>Macro-data loaded: 144 months, 7 columns.</code><br/><br/>"
            "<b>Diagnostic Meaning:</b> Confirms that the full population of 2.26 million retail loans has been loaded into memory without data corruption, paired with 12 complete years (144 months) of macroeconomic observations."
        ],
        kind='note'
    ))

    # =========================================================================
    # PHASE 3
    # =========================================================================
    story.append(Paragraph("5.3 Phase 3: Data Cleaning & Point-in-Time Merging (Cells 7 & 8)", styles['SectionTitle']))
    story.append(p("<b>Notebook Cell Index:</b> Cell 7 | <b>Type:</b> Code | <b>Execution Count:</b> 13"))

    cell7_code = """# 1. Clean missing origination dates, filter completed outcomes, and convert issue_d to datetime
print('Preparing micro-data dates and filtering terminal outcomes...')
micro_df = microdata.dropna(subset=['issue_d']).copy()

# Filter completed loan outcomes to prevent target censoring (Fully Paid vs Charged Off)
terminal_statuses = ['Fully Paid', 'Charged Off']
micro_df = micro_df[micro_df['loan_status'].isin(terminal_statuses)].copy()

# Encode binary target: 1 for Charged Off (Default), 0 for Fully Paid
micro_df['default'] = (micro_df['loan_status'] == 'Charged Off').astype(int)

# Standardize origination date to monthly datetime (start of month)
micro_df['issue_date'] = pd.to_datetime(micro_df['issue_d'], format='%b-%Y')
micro_df = micro_df.sort_values('issue_date').reset_index(drop=True)

# 2. Prepare macro-data with 1-month publication lag
print('Applying 1-month publication lag to macro indicators...')
macro_df = macrodata.copy()
macro_df['macro_date'] = pd.to_datetime(macro_df['date'])
macro_df = macro_df.sort_values('macro_date').reset_index(drop=True)

macro_indicators = ['UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'UMCSENT', 'DGS10', 'CPI_YoY_Inflation']
macro_df[macro_indicators] = macro_df[macro_indicators].shift(1)

# 3. Point-in-Time (PIT) Merge: Match each loan to prevailing macro climate
print('Merging micro-data and macro-data into unified dataframe df...')
df = pd.merge(
    micro_df,
    macro_df.drop(columns=['date']),
    left_on='issue_date',
    right_on='macro_date',
    how='inner'
).drop(columns=['macro_date'])

# Drop any edge records with missing macro data resulting from the 1-month lag
df = df.dropna(subset=macro_indicators).reset_index(drop=True)

print(f'Successfully created unified dataframe df: {df.shape[0]:,} rows, {df.shape[1]} columns.')"""
    story.append(code_block(cell7_code))

    story.append(Paragraph("Line-by-Line Breakdown & Technical Justification:", styles['SubsectionTitle']))
    story.append(bullet("<code>micro_df = microdata.dropna(subset=['issue_d']).copy()</code>: Removes any malformed records lacking an origination date. Calling `.copy()` creates an explicit memory buffer, preventing Pandas `SettingWithCopyWarning` errors."))
    story.append(bullet("<code>terminal_statuses = ['Fully Paid', 'Charged Off']</code>: <b>Critical survival filter.</b> Retains only completed contracts. Excludes unresolved states (`Current`, `Late`, `In Grace Period`) that are still active and have not revealed their ultimate outcome."))
    story.append(bullet("<code>(micro_df['loan_status'] == 'Charged Off').astype(int)</code>: Encodes the primary ground-truth target: 1 for Default, 0 for Non-Default."))
    story.append(bullet("<code>pd.to_datetime(micro_df['issue_d'], format='%b-%Y')</code>: Parses date strings formatted as 'Jun-2012' into standard monthly timestamps (`2012-06-01`)."))
    story.append(bullet("<code>macro_df[macro_indicators].shift(1)</code>: Applies the historical reporting delay so that loans originated in month $t$ are merged with macroeconomic information published at $t-1$."))
    story.append(bullet("<code>pd.merge(..., left_on='issue_date', right_on='macro_date', how='inner')</code>: Executes the <b>Point-in-Time (PIT) chronological join</b>. Every loan is strictly matched to the exact macro climate prevailing on its origination date."))

    story.append(callout(
        "Cell 7 Output & Diagnostics",
        [
            "<b>Output:</b><br/>"
            "<code>Preparing micro-data dates and filtering terminal outcomes...</code><br/>"
            "<code>Applying 1-month publication lag to macro indicators...</code><br/>"
            "<code>Merging micro-data and macro-data into unified dataframe df...</code><br/>"
            "<code>Successfully created unified dataframe df: 1,345,310 rows, 23 columns.</code><br/><br/>"
            "<b>Analysis:</b> Out of 2.26 million total records, exactly 1,345,310 loans had fully resolved terminal outcomes across the entire 2007–2018 period. All 1.345M loans were successfully matched to macroeconomic time-series data with zero missing values."
        ],
        kind='tip'
    ))

    # =========================================================================
    # PHASE 4
    # =========================================================================
    story.append(Paragraph("5.4 Phase 4: Cohort Isolation, Feature Engineering & Pipelines (Cell 10)", styles['SectionTitle']))
    story.append(p("<b>Notebook Cell Index:</b> Cell 10 | <b>Type:</b> Code | <b>Execution Count:</b> 15"))

    story.append(p(
        "Cell 10 is one of the most sophisticated engineering components in the pipeline (160 lines). "
        "It applies the <b>Maturity Cutoff Filter (Change 1)</b>, constructs pre-origination features, and designs "
        "Scikit-Learn `ColumnTransformer` pipelines that remain strictly unfitted until cross-validation folds."
    ))

    story.append(Paragraph("Key Code Snippets & Architectural Components in Cell 10:", styles['SubsectionTitle']))

    cell10_snippet1 = """# Change 1: Maturity Filter (<= 2015-12-31)
maturity_mask = df['issue_date'] <= pd.to_datetime('2015-12-31')
df = df[maturity_mask].copy().reset_index(drop=True)
print(f'Matured analytical cohort `df` ready: {df.shape[0]:,} loans x {df.shape[1]} columns.')
print(f'Overall cohort default rate: {df[\"default\"].mean() * 100:.2f}%')"""
    story.append(code_block(cell10_snippet1))
    story.append(p(
        "<b>Rationale:</b> This executes our core discovery from Chapter 4. By truncating the cohort at December 31, 2015, "
        "we discard the heavily censored 2016–2018 loans, leaving exactly <b>620,860 fully matured loans</b> with an overall base default rate of <b>13.94%</b>."
    ))

    cell10_snippet2 = """# Pre-Origination Feature Engineering
df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2.0
df['annual_inc_log'] = np.log1p(np.maximum(df['annual_inc'], 0))
df['loan_amnt_log'] = np.log1p(df['loan_amnt'])

# Macro Velocity (6-month momentum) & Interaction Ablations
macro_df_sorted = macrodata.sort_values('date').copy()
macro_df_sorted['macro_date'] = pd.to_datetime(macro_df_sorted['date'])
macro_df_sorted['unemp_change_6m'] = macro_df_sorted['UNRATE'].diff(6)
df['dti_x_unemp'] = df['dti'] * df['UNRATE']
df['fico_x_unemp'] = df['fico_avg'] * df['UNRATE']"""
    story.append(code_block(cell10_snippet2))
    story.append(p(
        "<b>Rationale:</b> "
        "1) `fico_avg` averages the 4-point FICO reporting range into a single continuous score. "
        "2) `np.log1p` computes $\\ln(1 + x)$ to normalize the extreme positive right-skew of annual income and loan amount. "
        "3) `unemp_change_6m` calculates the 6-month first difference ($\Delta U_t = U_t - U_{t-6}$) to measure acute shock velocity. "
        "4) `dti_x_unemp` and `fico_x_unemp` are interaction terms constructed exclusively for the controlled Phase 5/6 Ablation study."
    ))

    cell10_snippet3 = """# Leakage-Free Pipeline Construction
borrower_numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

borrower_cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', borrower_numeric_pipe, borrower_numeric_cols),
    ('cat', borrower_cat_pipe, borrower_cat_cols),
    ('macro', StandardScaler(), macro_cols)
])"""
    story.append(code_block(cell10_snippet3))
    story.append(p(
        "<b>The Zero-Leakage Architecture:</b> Notice that `preprocessor.fit()` is NOT called here! "
        "If a researcher fits a `StandardScaler` or `SimpleImputer` on the entire dataset before splitting, "
        "the test set's mean, variance, and median leak into the training features. "
        "Here, the pipelines are left <b>unfitted</b>. They will be fit strictly inside training folds in Phase 5."
    ))
    story.append(PageBreak())
    return story
