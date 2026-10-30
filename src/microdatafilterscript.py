import os
import pandas as pd

def filter_microdata():
    raw_path = os.path.join('data', 'raw', 'microdata.csv')
    processed_dir = os.path.join('data', 'processed')
    processed_path = os.path.join(processed_dir, 'microdataprocessed.csv')
    relevant_columns = [
        'issue_d',           # Origination date for chronological PIT macro merging
        'loan_status',       # Target: Fully Paid vs Charged Off (drops ongoing Current loans)
        'loan_amnt',         # Requested and approved principal amount ($)
        'term',              # Loan duration (36 vs 60 months)
        'int_rate',          # Contractual loan interest rate (%)
        'installment',       # Monthly repayment payment ($)
        'home_ownership',    # RENT, OWN, MORTGAGE
        'annual_inc',        # Self-reported borrower annual income ($)
        'dti',               # Debt-to-Income ratio
        'fico_range_low',    # Application FICO lower bound
        'fico_range_high',   # Application FICO upper bound
        'open_acc',          # Number of active credit lines
        'pub_rec',           # Derogatory public records
        'revol_util',        # Revolving credit utilization rate (%)
        'mort_acc'           # Number of mortgage accounts
    ]
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Read only the selected columns with chunking / low_memory for RAM efficiency
    df = pd.read_csv(raw_path, usecols=relevant_columns, low_memory=False)
    
    print(f"Dataset loaded: {len(df):,} rows and {len(df.columns)} columns.")
    print(f"Writing filtered dataset to: {processed_path}")
    
    df.to_csv(processed_path, index=False)
    print("Export completed successfully!")

if __name__ == '__main__':
    filter_microdata()

