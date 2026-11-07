"""
Macroeconomic Data Processing and Publication Lag Alignment
Implements conservative publication lag alignment (shift(2)),
calculates stationary inflation and momentum metrics, and
performs leak-free Point-in-Time (PIT) chronological merging.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple

from config import (
    PRIMARY_MACRO_FEATURES,
    MACRO_PUBLICATION_LAG_MONTHS
)

def prepare_macro_data(
    macro_path: str = 'data/processed/macrodata.csv',
    lag_months: int = MACRO_PUBLICATION_LAG_MONTHS
) -> pd.DataFrame:
    """
    Loads FRED macroeconomic series, computes stationary transformations
    (CPI YoY Inflation, 6-Month Unemployment Change) BEFORE applying publication lag,
    and then applies a conservative publication lag to prevent look-ahead bias.
    """
    if not os.path.exists(macro_path):
        raise FileNotFoundError(f"Macro dataset not found at {macro_path}")
        
    macro = pd.read_csv(macro_path)
    macro['macro_date'] = pd.to_datetime(macro['date'])
    macro = macro.sort_values('macro_date').reset_index(drop=True)

    # 1. Compute stationary transformations prior to lagging
    # Note: CPI_YoY_Inflation is calculated on raw unshifted CPIAUCSL
    if 'CPI_YoY_Inflation' not in macro.columns and 'CPIAUCSL' in macro.columns:
        macro['CPI_YoY_Inflation'] = (macro['CPIAUCSL'].pct_change(12) * 100).round(2)
        
    if 'UNRATE_change_6m' not in macro.columns and 'UNRATE' in macro.columns:
        # 6-month momentum: current rate minus rate 6 months prior
        macro['UNRATE_change_6m'] = (macro['UNRATE'] - macro['UNRATE'].shift(6)).round(2)

    # 2. Apply publication lag to all numerical indicators
    # We shift by lag_months (default 2) so loan issued at start of month t matches
    # data published and verified available prior to the 1st of month t.
    indicators_to_lag = [c for c in macro.columns if c not in ['date', 'macro_date']]
    
    # Store reference date before shifting to track which economic month is matched
    macro['economic_reference_date'] = macro['macro_date']
    
    # Apply chronological lag shift
    macro[indicators_to_lag] = macro[indicators_to_lag].shift(lag_months)
    
    # Reference date also shifted to verify exact economic vintage
    macro['economic_reference_date'] = macro['economic_reference_date'].shift(lag_months)

    # Drop early unaligned rows resulting from the shift
    macro_lagged = macro.dropna(subset=['UNRATE', 'UNRATE_change_6m', 'CPI_YoY_Inflation']).copy()
    macro_lagged = macro_lagged.reset_index(drop=True)
    
    return macro_lagged

def merge_micro_and_macro(
    micro_df: pd.DataFrame,
    macro_lagged_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Executes leak-free Point-in-Time (PIT) chronological merge between
    cleaned micro-data and lagged macroeconomic indicators.
    """
    merged = pd.merge(
        micro_df,
        macro_lagged_df.drop(columns=['date']),
        left_on='issue_date',
        right_on='macro_date',
        how='inner'
    ).drop(columns=['macro_date'])
    
    # Validation assertion: No loan origination date should be earlier than or equal to
    # the publication of its matched macro indicators
    assert (merged['issue_date'] > merged['economic_reference_date']).all(), (
        "Integrity Violation: Found loans matched with future or concurrent macro reference dates!"
    )
    
    return merged

if __name__ == '__main__':
    macro_clean = prepare_macro_data()
    print("Prepared lagged macro dataset preview:")
    print(macro_clean[['macro_date', 'economic_reference_date'] + PRIMARY_MACRO_FEATURES].head(8))
    print(f"\nTotal lagged macro months available: {len(macro_clean)}")

