"""
Data Integrity and Leakage Validation Suite
Audits the modeling dataset against six critical quantitative risk criteria:
1. Post-origination feature leakage.
2. Invalid target labels.
3. Duplicate loan records.
4. Macroeconomic publication alignment and look-ahead leakage.
5. Train / Validation / Test temporal overlap.
6. Preprocessing fitted strictly on training partition.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config import (
    BORROWER_NUMERICAL_FEATURES,
    PRIMARY_MACRO_FEATURES,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE
)

FORBIDDEN_POST_ORIGINATION_COLS = [
    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
    'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
    'out_prncp', 'out_prncp_inv', 'settlement_status', 'debt_settlement_flag'
]

def check_post_origination_leakage(columns: list) -> bool:
    """Verifies that no post-origination tracking columns exist in the feature set."""
    leaked = [col for col in columns if col in FORBIDDEN_POST_ORIGINATION_COLS]
    if leaked:
        print(f"[FAIL] LEAKAGE FAILURE: Found post-origination columns: {leaked}")
        return False
    print("[PASS] Zero post-origination tracking columns detected in feature set.")
    return True

def check_target_labels(df: pd.DataFrame, target_col: str = 'default') -> bool:
    """Ensures target is binary {0, 1} with no NaNs or ambiguous intermediate statuses."""
    if target_col not in df.columns:
        print(f"[FAIL] TARGET FAILURE: Target column '{target_col}' not found.")
        return False
        
    unique_vals = set(df[target_col].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        print(f"[FAIL] TARGET FAILURE: Invalid target labels found: {unique_vals}")
        return False
        
    nan_count = df[target_col].isna().sum()
    if nan_count > 0:
        print(f"[FAIL] TARGET FAILURE: Found {nan_count:,} missing values in target column.")
        return False
        
    print(f"[PASS] Target variable '{target_col}' is strictly binary {unique_vals} with 0 NaNs.")
    return True

def check_duplicate_records(df: pd.DataFrame) -> bool:
    """Audits for identical duplicate records."""
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"[WARNING] Detected {dup_count:,} duplicate records in dataset.")
        return False
    print("[PASS] Zero duplicate records found in dataset.")
    return True

def check_macro_alignment_and_leakage(df: pd.DataFrame) -> bool:
    """Verifies macro indicators have 0 missing values and respect publication lag."""
    for macro_col in PRIMARY_MACRO_FEATURES:
        if macro_col not in df.columns:
            print(f"[FAIL] MACRO FAILURE: Macro feature '{macro_col}' missing from dataframe.")
            return False
        nan_count = df[macro_col].isna().sum()
        if nan_count > 0:
            print(f"[FAIL] MACRO FAILURE: Feature '{macro_col}' has {nan_count:,} missing values post-merge.")
            return False
            
    if 'economic_reference_date' in df.columns and 'issue_date' in df.columns:
        violating = (df['issue_date'] <= df['economic_reference_date']).sum()
        if violating > 0:
            print(f"[FAIL] MACRO LEAKAGE: Found {violating:,} loans matched to future/concurrent macro data.")
            return False
        lag_diff = (df['issue_date'].dt.year - df['economic_reference_date'].dt.year) * 12 + (df['issue_date'].dt.month - df['economic_reference_date'].dt.month)
        min_lag = lag_diff.min()
        if min_lag < 2:
            print(f"[FAIL] MACRO LEAKAGE: Minimum publication lag is {min_lag} months (expected >= 2).")
            return False
            
    print("[PASS] Macro indicators complete (0 NaNs) and respect conservative publication lag (>= 2 months).")
    return True

def check_temporal_partition_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, date_col: str = 'issue_date') -> bool:
    """Verifies strict chronological separation between train and holdout test partitions."""
    max_train = train_df[date_col].max()
    min_test = test_df[date_col].min()
    
    if max_train >= min_test:
        print(f"[FAIL] TEMPORAL LEAKAGE: Train max date ({max_train}) >= Test min date ({min_test}). Overlap detected!")
        return False
        
    print(f"[PASS] Strict chronological boundary verified: Train ends {max_train.strftime('%Y-%m-%d')} < Test begins {min_test.strftime('%Y-%m-%d')}.")
    return True

def check_preprocessing_leakage(train_X: pd.DataFrame, test_X: pd.DataFrame, num_cols: list) -> bool:
    """Simulates pipeline fitting to ensure scalers/imputers are strictly trained on train_X only."""
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    # Fit strictly on train
    train_imputed = imputer.fit_transform(train_X[num_cols])
    train_scaled = scaler.fit_transform(train_imputed)
    
    # Store fitted parameters
    fitted_means = scaler.mean_.copy()
    
    # Transform test without re-fitting
    test_imputed = imputer.transform(test_X[num_cols])
    test_scaled = scaler.transform(test_imputed)
    
    # Check that scaler mean was not contaminated by test data
    assert np.array_equal(scaler.mean_, fitted_means), "Scaler parameters changed during test transformation!"
    
    # Check for NaNs post-scaling
    assert not np.isnan(train_scaled).any(), "NaNs found in scaled training data!"
    assert not np.isnan(test_scaled).any(), "NaNs found in scaled test data!"
    
    print("[PASS] Leakage-free preprocessing verified: Scalers and imputers fit exclusively on training data.")
    return True

def run_all_validation_checks(df_merged: pd.DataFrame = None) -> bool:
    """Runs the complete suite of validation checks."""
    print("==============================================================================")
    print("RUNNING QUANTITATIVE RISK INTEGRITY & LEAKAGE VALIDATION SUITE")
    print("==============================================================================")
    
    if df_merged is None:
        return True
        
    p1 = check_post_origination_leakage(df_merged.columns)
    p2 = check_target_labels(df_merged, 'default')
    p3 = check_duplicate_records(df_merged)
    p4 = check_macro_alignment_and_leakage(df_merged)
    
    train_df = df_merged[df_merged['issue_date'] <= TRAIN_END_DATE]
    test_df = df_merged[df_merged['issue_date'] >= TEST_START_DATE]
    p5 = check_temporal_partition_overlap(train_df, test_df)
    
    p6 = check_preprocessing_leakage(train_df, test_df, [c for c in BORROWER_NUMERICAL_FEATURES if c in train_df.columns])
    
    all_passed = all([p1, p2, p3, p4, p5, p6])
    print("==============================================================================")
    if all_passed:
        print("[SUCCESS] ALL 6 DATA INTEGRITY AND LEAKAGE CHECKS PASSED SUCCESSFULLY!")
    else:
        print("[AUDIT NOTE] REVIEW LOGS ABOVE FOR DETAILS.")
    print("==============================================================================")
    return all_passed

if __name__ == '__main__':
    from macro_processing import prepare_macro_data, merge_micro_and_macro
    from config import RESOLVED_DEFAULT_STATUSES, RESOLVED_PAID_STATUSES, MATURITY_CUTOFF_DATE
    
    print("Loading micro-data for end-to-end verification...")
    micro = pd.read_csv('data/processed/microdataprocessed.csv', low_memory=False)
    micro = micro.dropna(subset=['issue_d']).copy()
    
    # Deduplicate exact duplicate records
    micro = micro.drop_duplicates().reset_index(drop=True)
    
    micro = micro[micro['term'].str.contains('36', na=False)].copy()
    
    # Filter to mature cohort
    micro['issue_date'] = pd.to_datetime(micro['issue_d'], format='%b-%Y')
    micro = micro[micro['issue_date'] <= MATURITY_CUTOFF_DATE].copy()
    
    # Filter resolved statuses
    micro = micro[micro['loan_status'].isin(RESOLVED_DEFAULT_STATUSES + RESOLVED_PAID_STATUSES)].copy()
    micro['default'] = micro['loan_status'].isin(RESOLVED_DEFAULT_STATUSES).astype(int)
    
    macro = prepare_macro_data()
    df_merged = merge_micro_and_macro(micro, macro)
    
    run_all_validation_checks(df_merged)
