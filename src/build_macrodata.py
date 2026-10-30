"""
Download, process, and filter relevant FRED macroeconomic indicators
strictly aligned with the LendingClub loan origination window (2007 - 2018).

Indicators curated for Macro Stress Testing on Retail Credit Defaults:
1. UNRATE   - Civilian Unemployment Rate (%) [Labor market stress]
2. CPIAUCSL - Consumer Price Index for All Urban Consumers [Price inflation]
3. FEDFUNDS - Federal Funds Effective Rate (%) [Monetary policy benchmark]
4. UMCSENT  - University of Michigan Consumer Sentiment Index [Borrower confidence & spending expectations]
5. DGS10    - 10-Year Treasury Constant Maturity Rate (%) [Long-term benchmark borrowing cost, aggregated monthly]

Output:
data/processed/macrodata.csv (restricted to 2007-01-01 through 2018-12-31)
"""
import os
import json
import urllib.request
import pandas as pd

def get_fred_api_key():
    key = os.getenv('FRED_API_KEY')
    if key:
        return key
    if os.path.exists('.env'):
        with open('.env', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('FRED_API_KEY') and '=' in line:
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
    return None

def fetch_monthly_series(series_id, api_key, frequency='m', aggregation_method='avg'):
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
        f"&frequency={frequency}&aggregation_method={aggregation_method}"
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    print(f"Fetching '{series_id}' (monthly {aggregation_method})...")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        
    records = []
    for obs in data.get('observations', []):
        val_str = obs.get('value')
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            val = None
        records.append({'date': obs.get('date'), series_id: val})
        
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df.dropna()

def main():
    api_key = get_fred_api_key()
    if not api_key:
        raise ValueError("FRED_API_KEY not found in .env file or environment.")

    # Curated indicator set
    indicators = [
        ('UNRATE', 'm', 'avg'),     # Unemployment Rate (%)
        ('CPIAUCSL', 'm', 'avg'),   # Consumer Price Index
        ('FEDFUNDS', 'm', 'avg'),   # Fed Funds Rate (%)
        ('UMCSENT', 'm', 'avg'),    # Consumer Sentiment Index
        ('DGS10', 'm', 'avg')       # 10-Year Treasury Yield (%)
    ]

    macro_df = None
    for series_id, freq, agg in indicators:
        df_series = fetch_monthly_series(series_id, api_key, frequency=freq, aggregation_method=agg)
        if macro_df is None:
            macro_df = df_series
        else:
            macro_df = pd.merge(macro_df, df_series, on='date', how='outer')

    macro_df = macro_df.sort_values('date').reset_index(drop=True)

    # Compute YoY Inflation Rate (%) from CPIAUCSL across the full history first
    macro_df['CPI_YoY_Inflation'] = (macro_df['CPIAUCSL'].pct_change(12) * 100).round(2)

    # Filter strictly to the loan origination window: 2007 through 2018 (144 months)
    start_date = '2007-01-01'
    end_date = '2018-12-31'
    macro_df = macro_df[(macro_df['date'] >= start_date) & (macro_df['date'] <= end_date)].reset_index(drop=True)

    out_dir = os.path.join('data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'macrodata.csv')

    macro_df.to_csv(out_path, index=False)
    print(f"\n[Success] Processed macro dataset created at: {out_path}")
    print(f"Total months retained (2007-2018): {len(macro_df)}")
    print(f"Columns: {macro_df.columns.tolist()}")
    print("\nSample records:")
    print(macro_df.head())
    print("\nLatest records (End of 2018):")
    print(macro_df.tail())

if __name__ == '__main__':
    main()
