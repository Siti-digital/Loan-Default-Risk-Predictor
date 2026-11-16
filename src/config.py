"""
Project Configuration and Feature Group Definitions
Project: Macro-Aware Credit Risk (ES 335, IIT Gandhinagar)
"""

from typing import List, Dict, Any

# ==============================================================================
# 1. Feature Group Definitions
# ==============================================================================

# Model B: Primary Borrower-Only Feature Set (Strictly pre-origination)
# Note: Excludes lender risk-pricing outputs ('int_rate', 'installment', 'grade', 'sub_grade')
BORROWER_NUMERICAL_FEATURES: List[str] = [
    'annual_inc',          # log1p-transformed inside the pipeline (see LOG_TRANSFORM_FEATURES)
    'dti',
    'fico_avg',            # midpoint of fico_range_low / fico_range_high (the two differ by a constant 4 points)
    'open_acc',
    'pub_rec',
    'revol_util',
    'mort_acc',
    'loan_amnt'
]

BORROWER_CATEGORICAL_FEATURES: List[str] = [
    'home_ownership'
]

# Heavy right-skewed features that are log1p-transformed before imputation/scaling.
# Chosen on the 2013/2014 CV folds only (raw vs log income check in Phase 5), never on the 2015 test set.
LOG_TRANSFORM_FEATURES: List[str] = [
    'annual_inc'
]

# Primary Macroeconomic Indicators (Stationary, interpretable, non-redundant)
# Lags applied via conservative 2-month publication schedule
PRIMARY_MACRO_FEATURES: List[str] = [
    'UNRATE',              # National Civilian Unemployment Rate (%)
    'UNRATE_change_6m',    # 6-Month Change in National Unemployment Rate (bps/pts)
    'CPI_YoY_Inflation'    # 12-Month Year-over-Year Inflation Rate (%)
]

# Full Feature Sets for each Model
FEATURE_GROUPS: Dict[str, Dict[str, Any]] = {
    'MODEL_A': {
        'name': 'Historical Baseline',
        'description': 'Predicts empirical training base-rate default probability.',
        'features': []
    },
    'MODEL_B': {
        'name': 'Borrower-Only Logistic Regression',
        'description': 'Borrower and loan application characteristics at origination. Excludes lender pricing.',
        'numerical': BORROWER_NUMERICAL_FEATURES,
        'categorical': BORROWER_CATEGORICAL_FEATURES,
        'macro': []
    },
    'MODEL_C': {
        'name': 'Borrower + Macro Logistic Regression',
        'description': 'Borrower characteristics combined with conservative 2-month lagged macro indicators.',
        'numerical': BORROWER_NUMERICAL_FEATURES,
        'categorical': BORROWER_CATEGORICAL_FEATURES,
        'macro': PRIMARY_MACRO_FEATURES
    },
    'MODEL_D': {
        'name': 'Borrower + Macro LightGBM',
        'description': 'Identical information set to Model C evaluated with gradient boosted decision trees for non-linear relationships.',
        'numerical': BORROWER_NUMERICAL_FEATURES,
        'categorical': BORROWER_CATEGORICAL_FEATURES,
        'macro': PRIMARY_MACRO_FEATURES
    },
    'MODEL_D_BASE': {
        'name': 'Borrower-Only LightGBM',
        'description': 'Same information set as Model B, fitted with gradient boosted trees (non-linear borrower benchmark).',
        'numerical': BORROWER_NUMERICAL_FEATURES,
        'categorical': BORROWER_CATEGORICAL_FEATURES,
        'macro': []
    },
    'LENDER_BENCHMARK': {
        'name': 'Lender-Information Benchmark',
        'description': 'Adds lender-determined pricing variables (int_rate, installment) to borrower features. '
                       'grade/sub_grade are not in the processed extract, so they are not used.',
        'numerical': BORROWER_NUMERICAL_FEATURES + ['int_rate', 'installment'],
        'categorical': BORROWER_CATEGORICAL_FEATURES,
        'macro': []
    },
    'MODEL_M': {
        'name': 'Macro-Only Logistic Regression',
        'description': 'Extra experiment: the three lagged macro features alone (how much signal does macro carry by itself?).',
        'numerical': [],
        'categorical': [],
        'macro': PRIMARY_MACRO_FEATURES
    }
}

# Optional Economic Interaction Ablation Experiments
OPTIONAL_ABLATIONS = {
    'dti_x_unemp': ('dti', 'UNRATE'),
    'fico_x_unemp': ('fico_avg', 'UNRATE')
}

# ==============================================================================
# 2. Maturity and Cohort Cutoff Configuration
# ==============================================================================
COHORT_TERM: str = '36 months'
MATURITY_CUTOFF_DATE: str = '2015-12-31'
UNRESOLVED_THRESHOLD_PCT: float = 0.5  # Max allowable % of ongoing/unresolved loans

# Status mappings
RESOLVED_DEFAULT_STATUSES = [
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off'
]

RESOLVED_PAID_STATUSES = [
    'Fully Paid',
    'Does not meet the credit policy. Status:Fully Paid'
]

UNRESOLVED_STATUSES = [
    'Current',
    'In Grace Period',
    'Late (31-120 days)',
    'Late (16-30 days)'
]

# ==============================================================================
# 3. Chronological Evaluation Partitioning
# ==============================================================================
TRAIN_START_DATE: str = '2007-01-01'
TRAIN_END_DATE: str = '2014-12-31'

# Expanding window cross-validation within training window
CV_SPLITS = [
    {'fold': 1, 'train_end': '2012-12-31', 'val_start': '2013-01-01', 'val_end': '2013-12-31'},
    {'fold': 2, 'train_end': '2013-12-31', 'val_start': '2014-01-01', 'val_end': '2014-12-31'}
]

# Untouched Holdout Test Partition
TEST_START_DATE: str = '2015-01-01'
TEST_END_DATE: str = '2015-12-31'

# ==============================================================================
# 4. Macroeconomic Publication Alignment
# ==============================================================================
MACRO_PUBLICATION_LAG_MONTHS: int = 2  # Conservative 2-month reporting lag (shift(2))

# ==============================================================================
# 5. Model Selection (expanding-window CV only) and Extra Experiments
# ==============================================================================
RANDOM_SEED: int = 42

# Logistic regression L2 strength, selected per model by mean CV PR-AUC on the 2013 and 2014 folds
LR_C_GRID: List[float] = [0.01, 0.1, 1.0]

# LightGBM: fixed regularisation + a small grid selected by mean CV PR-AUC
LGBM_FIXED_PARAMS: Dict[str, Any] = {
    'min_child_samples': 200, 'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8,
    'reg_lambda': 10.0, 'deterministic': True, 'force_col_wise': True,
    'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbose': -1
}
LGBM_GRID: List[Dict[str, Any]] = [
    {'num_leaves': 15, 'max_depth': 5, 'n_estimators': 200, 'learning_rate': 0.03},
    {'num_leaves': 7,  'max_depth': 3, 'n_estimators': 300, 'learning_rate': 0.05},
    {'num_leaves': 31, 'max_depth': 6, 'n_estimators': 200, 'learning_rate': 0.03},
    {'num_leaves': 15, 'max_depth': 5, 'n_estimators': 600, 'learning_rate': 0.02},
]

# Label-availability ("embargo") experiment: on 2015-01-01 only loans issued up to 2011-12
# have a fully observed 36-month outcome, so a model deployed then could only train on them.
EMBARGO_TRAIN_END_DATE: str = '2011-12-31'

# Bootstrap repetitions for month-level block bootstrap on the 2015 holdout
BOOTSTRAP_REPS: int = 500

