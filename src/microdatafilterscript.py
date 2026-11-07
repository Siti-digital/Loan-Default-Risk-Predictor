"""
Curates raw LendingClub micro-data down to a strictly pre-origination allowlist,
eliminating all post-origination columns (recoveries, payment tracking, collection fees)
to prevent look-ahead target leakage.
"""

import os
import pandas as pd

def filter_microdata(
    raw_path: str = os.path.join('data', 'raw', 'microdata.csv'),
    processed_path: str = os.path.join('data', 'processed', 'microdataprocessed.csv')
):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{raw_path}'.\n"
            "Please ensure raw LendingClub data (e.g., accepted_2007_to_2018Q4.csv) is placed in 'data/raw/'."
        )

    # Pre-origination allowlist:
    # 1. Target & Date: issue_d, loan_status
    # 2. Borrower Profile: annual_inc, dti, fico_range_low, fico_range_high, open_acc, pub_rec, revol_util, mort_acc, home_ownership, loan_amnt, term
    # 3. Candidate Extended Attributes (if available in raw file): addr_state, purpose, emp_length, verification_status, earliest_cr_line, revol_bal
    # 4. Lender-determined Benchmark: int_rate, installment, grade, sub_grade
    full_allowlist = [
        'issue_d',           # Origination date for chronological PIT macro merging
        'loan_status',       # Target: Fully Paid vs Charged Off
        'loan_amnt',         # Requested and approved principal amount ($)
        'term',              # Loan duration (36 vs 60 months)
        'home_ownership',    # RENT, OWN, MORTGAGE, OTHER
        'annual_inc',        # Self-reported borrower annual income ($)
        'dti',               # Debt-to-Income ratio
        'fico_range_low',    # Application FICO lower bound
        'fico_range_high',   # Application FICO upper bound
        'open_acc',          # Number of active credit lines
        'pub_rec',           # Derogatory public records
        'revol_util',        # Revolving credit utilization rate (%)
        'mort_acc',          # Number of mortgage accounts
        # Extended candidate columns
        'addr_state',        # Borrower state of residence (for state macro matching)
        'purpose',           # Stated loan purpose (debt_consolidation, credit_card, etc.)
        'emp_length',        # Employment length in years
        'verification_status', # Income verification status
        'earliest_cr_line',  # Month borrower opened earliest credit line
        'revol_bal',         # Total credit revolving balance ($)
        # Lender-determined risk pricing benchmark features
        'int_rate',          # Contractual loan interest rate (%)
        'installment',       # Monthly repayment payment ($)
        'grade',             # LendingClub assigned risk grade (A-G)
        'sub_grade'          # LendingClub assigned sub-grade (A1-G5)
    ]
    
    # Read sample to detect which columns actually exist in raw CSV
    sample_df = pd.read_csv(raw_path, nrows=10)
    available_cols = [c for c in full_allowlist if c in sample_df.columns]
    missing_from_raw = [c for c in full_allowlist if c not in sample_df.columns]
    
    if missing_from_raw:
        print(f"Note: The following candidate columns were not present in raw file and will be skipped: {missing_from_raw}")
        
    print(f"Loading raw micro-data with {len(available_cols)} pre-origination allowlist columns...")
    df = pd.read_csv(raw_path, usecols=available_cols, low_memory=False)
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    print(f"Dataset loaded: {len(df):,} rows and {len(df.columns)} columns.")
    print(f"Writing curated dataset to: {processed_path}")
    
    df.to_csv(processed_path, index=False)
    print("Export completed successfully!")

if __name__ == '__main__':
    filter_microdata()
