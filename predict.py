#!/usr/bin/env python3
"""
Loan Default Risk Predictor & Macroeconomic Stress-Testing CLI
Evaluates calibrated Probability of Default (PD) using trained models from loan_default_predictor.bin.

Usage Examples:
  1. Default Demo Run:
     python predict.py

  2. Single Borrower CLI Evaluation:
     python predict.py --fico 700 --dti 18.5 --income 75000 --loan 15000 --home RENT --unemp 5.2 --cpi 2.1 --unemp-chg 0.0

  3. Non-linear LightGBM Evaluation:
     python predict.py --model MODEL_D --fico 640 --dti 28.0 --income 45000 --unemp 9.5

  4. Batch CSV Scoring:
     python predict.py --csv path/to/borrowers.csv --output path/to/scored_borrowers.csv
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd


MODEL_BIN_PATH = "loan_default_predictor.bin"


def load_model_package(bin_path=MODEL_BIN_PATH):
    """Load serialized model package and metadata from binary file."""
    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Binary model file '{bin_path}' not found! "
            f"Please ensure '{bin_path}' is present in the working directory."
        )
    return joblib.load(bin_path)


def assign_risk_category(pd_value, threshold):
    """Categorize continuous default probability into actionable risk tiers."""
    if pd_value >= 0.22:
        return "Severe Risk", "FLAG_HIGH_RISK (Auto-Decline / Senior Credit Underwriter Review)"
    elif pd_value >= threshold:
        return "Elevated Risk", "FLAG_MODERATE_RISK (Manual Verification Required)"
    elif pd_value >= 0.10:
        return "Moderate Risk", "APPROVE (Standard Terms)"
    else:
        return "Low Risk", "APPROVE (Preferred Prime Terms)"


def score_dataframe(df_input, pkg, model_key="MODEL_C"):
    """Transform input DataFrame and predict calibrated Probability of Default."""
    if model_key not in pkg["fitted_models"]:
        available = list(pkg["fitted_models"].keys())
        raise ValueError(f"Model '{model_key}' not found in package. Available models: {available}")

    prep = pkg["fitted_preprocessors"][model_key]
    model = pkg["fitted_models"][model_key]

    # Fill default baseline values for missing columns
    df_eval = df_input.copy()

    # Handle fico alias
    if "fico" in df_eval.columns:
        if "fico_range_low" not in df_eval.columns:
            df_eval["fico_range_low"] = df_eval["fico"]
        if "fico_range_high" not in df_eval.columns:
            df_eval["fico_range_high"] = df_eval["fico"] + 4

    # Default values for borrower and macro traits if omitted
    defaults = {
        "annual_inc": 70000.0,
        "dti": 18.0,
        "fico_range_low": 690.0,
        "fico_range_high": 694.0,
        "open_acc": 11.0,
        "pub_rec": 0.0,
        "revol_util": 50.0,
        "mort_acc": 1.0,
        "loan_amnt": 15000.0,
        "home_ownership": "RENT",
        "UNRATE": 5.0,
        "UNRATE_change_6m": 0.0,
        "CPI_YoY_Inflation": 2.0,
        "dti_x_unemp": 18.0 * 5.0,
        "fico_x_unemp": 692.0 * 5.0,
        "int_rate": 13.0,
        "installment": 350.0
    }

    for col, default_val in defaults.items():
        if col not in df_eval.columns:
            df_eval[col] = default_val

    # Re-calculate interaction terms if relevant
    if "fico_avg" not in df_eval.columns:
        df_eval["fico_avg"] = (df_eval["fico_range_low"] + df_eval["fico_range_high"]) / 2.0
    df_eval["dti_x_unemp"] = df_eval["dti"] * df_eval["UNRATE"]
    df_eval["fico_x_unemp"] = df_eval["fico_avg"] * df_eval["UNRATE"]

    X_trans = prep.transform(df_eval)
    probabilities = model.predict_proba(X_trans)[:, 1]
    return probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Loan Default Risk Predictor & Macro Stress Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--bin", type=str, default=MODEL_BIN_PATH, help="Path to binary model file (.bin)")
    parser.add_argument("--model", type=str, default="MODEL_C", choices=["MODEL_B", "MODEL_C", "MODEL_C_ABLATION", "MODEL_D_BASE", "MODEL_D", "LENDER_BENCHMARK"], help="Model variant to evaluate (default: MODEL_C)")
    parser.add_argument("--csv", type=str, default=None, help="Path to input CSV for batch scoring")
    parser.add_argument("--output", type=str, default=None, help="Path to save scored CSV output")

    # Borrower financial arguments
    parser.add_argument("--fico", type=float, default=None, help="Borrower FICO Score (e.g. 690)")
    parser.add_argument("--dti", type=float, default=None, help="Debt-to-Income ratio % (e.g. 18.5)")
    parser.add_argument("--income", type=float, default=None, help="Annual Income $ (e.g. 75000)")
    parser.add_argument("--loan", type=float, default=None, help="Loan Amount $ (e.g. 15000)")
    parser.add_argument("--home", type=str, default=None, choices=["RENT", "MORTGAGE", "OWN", "OTHER"], help="Housing status")

    # Macroeconomic scenario arguments
    parser.add_argument("--unemp", type=float, default=None, help="National Unemployment Rate % (UNRATE, e.g. 5.2)")
    parser.add_argument("--cpi", type=float, default=None, help="CPI Inflation Rate % YoY (e.g. 2.1)")
    parser.add_argument("--unemp-chg", type=float, default=None, help="6-Month Unemployment Change % (e.g. 0.0)")

    args = parser.parse_args()

    # Load binary model package
    try:
        pkg = load_model_package(args.bin)
    except Exception as e:
        print(f"Error loading model binary: {e}")
        sys.exit(1)

    threshold = pkg.get("prevalence_threshold", 0.148863)

    # 1. BATCH CSV SCORING MODE
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: Input CSV file '{args.csv}' not found.")
            sys.exit(1)

        print(f"Loading batch data from '{args.csv}'...")
        df_batch = pd.read_csv(args.csv)
        print(f"Scoring {len(df_batch):,} rows using {args.model}...")
        probs = score_dataframe(df_batch, pkg, model_key=args.model)
        df_batch["predicted_default_prob"] = np.round(probs, 5)
        df_batch["decision"] = np.where(probs >= threshold, "FLAGGED_HIGH_RISK", "APPROVED")

        out_path = args.output or "scored_output.csv"
        df_batch.to_csv(out_path, index=False)
        print(f"[SUCCESS] Scored {len(df_batch):,} loans. Saved to: '{out_path}'.")
        return

    # 2. SINGLE BORROWER EVALUATION MODE
    has_custom_input = any(
        x is not None for x in [args.fico, args.dti, args.income, args.loan, args.home, args.unemp, args.cpi, args.unemp_chg]
    )

    fico = args.fico if args.fico is not None else 690.0
    dti = args.dti if args.dti is not None else 18.0
    income = args.income if args.income is not None else 70000.0
    loan = args.loan if args.loan is not None else 15000.0
    home = args.home if args.home is not None else "RENT"
    unemp = args.unemp if args.unemp is not None else 5.0
    cpi = args.cpi if args.cpi is not None else 2.0
    unemp_chg = args.unemp_chg if args.unemp_chg is not None else 0.0

    borrower_data = pd.DataFrame([{
        "annual_inc": income,
        "dti": dti,
        "fico_range_low": fico,
        "fico_range_high": fico + 4,
        "loan_amnt": loan,
        "home_ownership": home,
        "UNRATE": unemp,
        "CPI_YoY_Inflation": cpi,
        "UNRATE_change_6m": unemp_chg
    }])

    prob = score_dataframe(borrower_data, pkg, model_key=args.model)[0]
    tier, recommendation = assign_risk_category(prob, threshold)

    print("=" * 72)
    print("LOAN DEFAULT RISK PREDICTOR & MACRO STRESS EVALUATOR")
    print("=" * 72)
    print(f"Model Evaluated     : {args.model} ({pkg.get('model_name', 'Fitted ML Model')})")
    print(f"Trained Partition   : {pkg.get('training_cohort', '2007-2014 Cohort')}")
    print(f"Prevalence Threshold: {threshold:.2%}")
    print("-" * 72)
    print("Borrower Financial Profile:")
    print(f"  * FICO Score       : {fico:.0f}")
    print(f"  * Debt-to-Income   : {dti:.1f}%")
    print(f"  * Annual Income    : ${income:,.0f}")
    print(f"  * Requested Loan   : ${loan:,.0f}")
    print(f"  * Housing Status   : {home}")
    print("-" * 72)
    print("Macroeconomic Conditions at Origination:")
    print(f"  * National UNRATE  : {unemp:.1f}%")
    print(f"  * CPI Inflation YoY: {cpi:.1f}%")
    print(f"  * 6M Unemp Change  : {unemp_chg:+.1f}%")
    print("=" * 72)
    print(f"PREDICTED PROBABILITY OF DEFAULT : {prob:.2%}")
    print(f"RISK TIER                        : {tier}")
    print(f"CREDIT UNDERWRITING ACTION       : {recommendation}")
    print("=" * 72)

    if not has_custom_input:
        print("\n[NOTE] Ran default sample borrower. You can specify custom inputs via:")
        print("  python predict.py --fico 720 --dti 15 --income 90000 --unemp 6.5 --cpi 3.0\n")


if __name__ == "__main__":
    main()
