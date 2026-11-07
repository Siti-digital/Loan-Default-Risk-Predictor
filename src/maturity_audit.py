"""
Loan Maturity and Censoring Audit Script
Audits unresolved loan proportions across origination cohorts,
identifies the empirical maturity cutoff (<= 0.5% unresolved),
and outputs the maturity audit table and visualization plot.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    RESOLVED_DEFAULT_STATUSES,
    RESOLVED_PAID_STATUSES,
    UNRESOLVED_STATUSES,
    COHORT_TERM,
    MATURITY_CUTOFF_DATE,
    UNRESOLVED_THRESHOLD_PCT
)

def run_maturity_audit(data_path='data/processed/microdataprocessed.csv', output_dir='reports'):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    total_raw = len(df)
    print(f"Total raw observations: {total_raw:,}")

    # Clean missing dates
    df = df.dropna(subset=['issue_d']).copy()
    df['issue_date'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df['issue_year'] = df['issue_date'].dt.year

    # Audit term formatting
    print("\nTerm distribution in dataset:")
    print(df['term'].value_counts(dropna=False))

    # Tag resolution status
    df['is_unresolved'] = df['loan_status'].isin(UNRESOLVED_STATUSES)
    df['is_default'] = df['loan_status'].isin(RESOLVED_DEFAULT_STATUSES)
    df['is_paid'] = df['loan_status'].isin(RESOLVED_PAID_STATUSES)

    # 1. 36-Month Cohort Audit
    df_36 = df[df['term'].str.contains('36', na=False)].copy()
    total_36 = len(df_36)

    annual_36 = df_36.groupby('issue_year').agg(
        total=('loan_status', 'count'),
        unresolved=('is_unresolved', 'sum'),
        default=('is_default', 'sum'),
        paid=('is_paid', 'sum')
    )
    annual_36['pct_unresolved'] = (annual_36['unresolved'] / annual_36['total']) * 100
    annual_36['resolved_total'] = annual_36['default'] + annual_36['paid']
    annual_36['default_rate_resolved'] = (annual_36['default'] / annual_36['resolved_total']) * 100

    print("\n=== 36-Month Loans Annual Maturity Audit ===")
    print(annual_36.to_string())

    # 2. Monthly Audit around transition (2015 - 2016)
    monthly_36 = df_36[df_36['issue_date'] >= '2015-01-01'].groupby('issue_date').agg(
        total=('loan_status', 'count'),
        unresolved=('is_unresolved', 'sum'),
        default=('is_default', 'sum'),
        paid=('is_paid', 'sum')
    )
    monthly_36['pct_unresolved'] = (monthly_36['unresolved'] / monthly_36['total']) * 100
    monthly_36['default_rate'] = (monthly_36['default'] / (monthly_36['default'] + monthly_36['paid'])) * 100

    print("\n=== 36-Month Loans Monthly Maturity Audit (2015 - 2016 Transition) ===")
    print(monthly_36.head(15).to_string())

    # Find latest month with <= UNRESOLVED_THRESHOLD_PCT
    valid_months = monthly_36[monthly_36['pct_unresolved'] <= UNRESOLVED_THRESHOLD_PCT]
    latest_mature_month = valid_months.index.max()
    print(f"\nEmpirical Maturity Cutoff: Latest month with <= {UNRESOLVED_THRESHOLD_PCT}% unresolved is {latest_mature_month.strftime('%B %Y')} ({valid_months.loc[latest_mature_month, 'pct_unresolved']:.3f}% unresolved).")

    # 3. Retained vs Excluded Counts
    # Retained: 36-month loans up to MATURITY_CUTOFF_DATE that are resolved
    retained_mask = (df_36['issue_date'] <= MATURITY_CUTOFF_DATE) & (df_36['is_default'] | df_36['is_paid'])
    retained_count = retained_mask.sum()
    
    # Excluded breakdown
    excluded_60_month = len(df[df['term'].str.contains('60', na=False)])
    excluded_post_cutoff_36 = len(df_36[df_36['issue_date'] > MATURITY_CUTOFF_DATE])
    excluded_unresolved_in_period = len(df_36[(df_36['issue_date'] <= MATURITY_CUTOFF_DATE) & df_36['is_unresolved']])
    excluded_na = total_raw - len(df)

    print("\n=== Cohort Selection: Retained vs Excluded Summary ===")
    print(f"Total Raw Loans: {total_raw:,}")
    print(f"  [-] Excluded 60-Month Loans: {excluded_60_month:,} (due to severe multi-year censoring)")
    print(f"  [-] Excluded Post-2015 36-Month Loans: {excluded_post_cutoff_36:,} (unresolved ongoing cohorts)")
    print(f"  [-] Excluded Unresolved within 2007-2015: {excluded_unresolved_in_period:,} (Current/Late states)")
    print(f"  [-] Excluded Empty / Missing issue_d rows: {excluded_na:,}")
    print(f"  [=] Retained Fully Matured Analytical Cohort (2007-2015): {retained_count:,}")

    # 4. Save Audit Table to CSV
    table_path = os.path.join(output_dir, 'maturity_audit_table.csv')
    annual_36.to_csv(table_path)
    print(f"\nAudit table saved to: {table_path}")

    # 5. Generate and Save Publication-Quality Audit Plot
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # All monthly cohorts from 2007 to 2018
    all_monthly = df_36.groupby('issue_date').agg(
        total=('loan_status', 'count'),
        unresolved=('is_unresolved', 'sum')
    )
    all_monthly['pct_unresolved'] = (all_monthly['unresolved'] / all_monthly['total']) * 100

    ax1.plot(all_monthly.index, all_monthly['pct_unresolved'], color='#dc2626', linewidth=2.5, label='% Unresolved (Current / Late)')
    ax1.axhline(y=UNRESOLVED_THRESHOLD_PCT, color='#2563eb', linestyle='--', linewidth=1.5, label=f'Maturity Threshold ({UNRESOLVED_THRESHOLD_PCT}%)')
    ax1.axvline(pd.to_datetime(MATURITY_CUTOFF_DATE), color='#16a34a', linestyle='-', linewidth=2, label=f'Proposed Maturity Cutoff ({MATURITY_CUTOFF_DATE})')

    ax1.set_title('Loan Maturity & Censoring Audit: Proportion of Unresolved Loans by Origination Cohort\n(36-Month LendingClub Installment Loans)', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel('Origination Date (Cohort Vintage)', fontsize=11, fontweight='medium')
    ax1.set_ylabel('Unresolved Loans (%)', fontsize=11, fontweight='medium', color='#dc2626')
    ax1.set_ylim(-2, 100)
    ax1.tick_params(axis='y', labelcolor='#dc2626')

    # Shade the mature vs censored regions
    ax1.axvspan(all_monthly.index.min(), pd.to_datetime(MATURITY_CUTOFF_DATE), alpha=0.12, color='#16a34a', label='Retained Mature Cohort (0.0% - 0.3% Unresolved)')
    ax1.axvspan(pd.to_datetime(MATURITY_CUTOFF_DATE), all_monthly.index.max(), alpha=0.12, color='#dc2626', label='Excluded Right-Censored Cohorts (>28% - 88% Unresolved)')

    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    fig.tight_layout()

    plot_path = os.path.join(fig_dir, 'loan_maturity_censoring.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Maturity audit plot saved to: {plot_path}")

    return annual_36, monthly_36

if __name__ == '__main__':
    run_maturity_audit()

