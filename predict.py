#!/usr/bin/env python3
"""
Interactive Menu-Driven Loan Default Risk Predictor & Macroeconomic Stress-Testing System
ES 335 Machine Learning | Quantitative Risk & Credit Analytics

Loads trained model artifacts from 'loan_default_predictor.bin' and provides an
interactive, menu-based evaluation workflow with line-by-line input validation.
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
    """Categorize continuous default probability into actionable credit risk tiers."""
    if pd_value >= 0.22:
        return "Severe Risk", "FLAG_HIGH_RISK (Auto-Decline / Senior Underwriter Review)", "#ef4444"
    elif pd_value >= threshold:
        return "Elevated Risk", "FLAG_MODERATE_RISK (Manual Verification Required)", "#f59e0b"
    elif pd_value >= 0.10:
        return "Moderate Risk", "APPROVE (Standard Underwriting Terms)", "#3b82f6"
    else:
        return "Low Risk", "APPROVE (Preferred Prime Rate)", "#10b981"


def score_dataframe(df_input, pkg, model_key="MODEL_C"):
    """Transform input DataFrame and predict calibrated Probability of Default."""
    if model_key not in pkg["fitted_models"]:
        available = list(pkg["fitted_models"].keys())
        raise ValueError(f"Model '{model_key}' not found in package. Available models: {available}")

    prep = pkg["fitted_preprocessors"][model_key]
    model = pkg["fitted_models"][model_key]

    df_eval = df_input.copy()

    # Handle FICO aliases
    if "fico" in df_eval.columns:
        if "fico_range_low" not in df_eval.columns:
            df_eval["fico_range_low"] = df_eval["fico"]
        if "fico_range_high" not in df_eval.columns:
            df_eval["fico_range_high"] = df_eval["fico"] + 4.0

    # Fill defaults for omitted features
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
        "int_rate": 13.0,
        "installment": 350.0
    }

    for col, default_val in defaults.items():
        if col not in df_eval.columns:
            df_eval[col] = default_val

    # Re-calculate engineered interaction features if needed
    if "fico_avg" not in df_eval.columns:
        df_eval["fico_avg"] = (df_eval["fico_range_low"] + df_eval["fico_range_high"]) / 2.0
    df_eval["dti_x_unemp"] = df_eval["dti"] * df_eval["UNRATE"]
    df_eval["fico_x_unemp"] = df_eval["fico_avg"] * df_eval["UNRATE"]

    X_trans = prep.transform(df_eval)
    probabilities = model.predict_proba(X_trans)[:, 1]
    return probabilities


def prompt_input(param_name, example_text, default_val, val_type=float, min_val=None, max_val=None):
    """
    Robust interactive input prompt with type validation, range checking,
    default fallbacks, and '-1' cancel detection.
    """
    while True:
        try:
            prompt_str = f"  -> Enter {param_name} [{example_text}] (Default: {default_val}, -1 to cancel): "
            user_raw = input(prompt_str).strip()

            if user_raw == "-1":
                return -1

            if user_raw == "":
                return default_val

            if val_type == str:
                clean_str = user_raw.upper()
                return clean_str

            val = val_type(user_raw)

            if min_val is not None and val < min_val:
                print(f"     [!] Value too low. Must be >= {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"     [!] Value too high. Must be <= {max_val}. Try again.")
                continue

            return val
        except ValueError:
            print(f"     [!] Invalid format. Please enter a valid {val_type.__name__} (e.g. {example_text}).")


def run_interactive_prediction(pkg):
    """Conducts a line-by-line interactive assessment of a new loan applicant."""
    threshold = pkg.get("prevalence_threshold", 0.148863)

    print("\n" + "=" * 80)
    print("  INTERACTIVE CREDIT RISK & MACRO STRESS ASSESSMENT")
    print("=" * 80)
    print("This system will collect borrower attributes and macroeconomic conditions at issue.")
    print("Please enter values line-by-line. Press [Enter] to use the suggested default.")
    print("Type -1 at any prompt to cancel and return to the main menu.\n")

    print("[SECTION 1: Individual Borrower Fundamentals]")
    fico = prompt_input("FICO Credit Score", "e.g. 710 or 685 (Integer 600-850)", 690, int, 600, 850)
    if fico == -1:
        print("\n[Assessment Cancelled] Returning to main menu...")
        return

    dti = prompt_input("Debt-to-Income Ratio (DTI)", "e.g. 18.52 or 24.3 (Float % 0-50)", 18.0, float, 0.0, 60.0)
    if dti == -1:
        print("\n[Assessment Cancelled] Returning to main menu...")
        return

    income = prompt_input("Annual Borrower Income", "e.g. 75000.0 or 120000.0 (Float $)", 70000.0, float, 5000.0, 5000000.0)
    if income == -1:
        print("\n[Assessment Cancelled] Returning to main menu...")
        return

    loan = prompt_input("Requested Loan Amount", "e.g. 15000.0 or 25000.0 (Float $)", 15000.0, float, 1000.0, 50000.0)
    if loan == -1:
        print("\n[Assessment Cancelled] Returning to main menu...")
        return

    home = prompt_input("Home Ownership Status", "RENT, MORTGAGE, OWN, OTHER", "RENT", str)
    if home == -1:
        print("\n[Assessment Cancelled] Returning to main menu...")
        return
    if home not in ["RENT", "MORTGAGE", "OWN", "OTHER"]:
        print(f"     [i] Unrecognized ownership status '{home}'. Defaulting to 'RENT'.")
        home = "RENT"

    print("\n[SECTION 2: Macroeconomic Environment at Origination]")
    print("  Note: Sliders are calibrated for standard to extreme economic stress testing.")
    unemp = prompt_input("National Unemployment Rate (UNRATE)", "e.g. 5.2 or 15.0 (Float % 3.0-20.0)", 5.0, float, 3.0, 20.0)
    if unemp == -1:
        print("\n[Assessment Cancelled] Returning to main menu...")
        return

    cpi = prompt_input("CPI Inflation Rate (YoY)", "e.g. 2.1 or 8.5 (Float % -2.0 to 15.0)", 2.0, float, -2.0, 15.0)
    if cpi == -1:
        print("\n[Assessment Cancelled] Returning to main menu...")
        return

    unemp_chg = prompt_input("6-Month Change in Unemployment", "e.g. 0.0 or 1.8 (Float % -1.5 to 2.5)", 0.0, float, -1.5, 2.5)
    if unemp_chg == -1:
        print("\n[Assessment Cancelled] Returning to main menu...")
        return

    print("\n[SECTION 3: Model Architecture Selection]")
    print("  [1] MODEL_C: Borrower + Macro Logistic Regression (Default / Regulatory Standard)")
    print("  [2] MODEL_D: Borrower + Macro LightGBM (Gradient Boosted Trees)")
    print("  [3] MODEL_B: Borrower-Only Logistic Regression (Excludes Macroeconomic Features)")
    m_choice = prompt_input("Select Model [1, 2, or 3]", "1, 2, or 3", 1, int, 1, 3)
    if m_choice == -1:
        print("\n[Assessment Cancelled] Returning to main menu...")
        return

    model_key_map = {1: "MODEL_C", 2: "MODEL_D", 3: "MODEL_B"}
    chosen_model = model_key_map.get(m_choice, "MODEL_C")

    # Build evaluation DataFrame
    borrower_df = pd.DataFrame([{
        "annual_inc": income,
        "dti": dti,
        "fico_range_low": float(fico),
        "fico_range_high": float(fico + 4),
        "loan_amnt": loan,
        "home_ownership": home,
        "UNRATE": unemp,
        "CPI_YoY_Inflation": cpi,
        "UNRATE_change_6m": unemp_chg
    }])

    # Score current loan
    pd_pred = score_dataframe(borrower_df, pkg, model_key=chosen_model)[0]
    tier, recommendation, _ = assign_risk_category(pd_pred, threshold)

    # Score benign baseline for sensitivity comparison
    benign_df = borrower_df.copy()
    benign_df["UNRATE"] = 5.0
    benign_df["CPI_YoY_Inflation"] = 2.0
    benign_df["UNRATE_change_6m"] = 0.0
    pd_benign = score_dataframe(benign_df, pkg, model_key=chosen_model)[0]

    odds_curr = pd_pred / (1.0 - pd_pred)
    odds_benign = pd_benign / (1.0 - pd_benign)
    odds_delta_pct = ((odds_curr / odds_benign) - 1.0) * 100.0

    print("\n" + "=" * 80)
    print("                     LOAN RISK ASSESSMENT & DECISION REPORT")
    print("=" * 80)
    print(f"Model Evaluated     : {chosen_model}")
    print(f"Training Baseline PD: {pkg.get('baseline_pd', 0.13155):.2%} (2007-2014 Training Cohort)")
    print(f"Prevalence Threshold: {threshold:.2%} (2015 Out-of-Time Base Rate)")
    print("-" * 80)
    print(f"Borrower Summary    : FICO {fico} | DTI {dti:.1f}% | Income ${income:,.0f} | Loan ${loan:,.0f} | {home}")
    print(f"Macro Environment   : UNRATE {unemp:.1f}% | CPI {cpi:.1f}% | 6M Unemp Change {unemp_chg:+.1f}%")
    print("-" * 80)
    print(f"CALIBRATED PROBABILITY OF DEFAULT : {pd_pred:.2%}")
    print(f"QUALITATIVE RISK TIER             : {tier}")
    print(f"UNDERWRITING DECISION ACTION      : {recommendation}")
    print("-" * 80)
    sign = "+" if odds_delta_pct >= 0 else ""
    print(f"Macro Sensitivity Delta           : {sign}{odds_delta_pct:.1f}% shift in default odds relative to benign baseline.")
    print("=" * 80 + "\n")


def run_macro_stress_simulation(pkg):
    """Simulates how an applicant's default risk evolves under varying macroeconomic shock severities."""
    print("\n" + "=" * 80)
    print("           MACROECONOMIC SHOCK SENSITIVITY & STRESS SIMULATION")
    print("=" * 80)
    print("Evaluates how the default probability of a single applicant shifts across economic scenarios.")

    fico = prompt_input("Borrower FICO Score", "e.g. 680 (Integer 600-850)", 680, int, 600, 850)
    if fico == -1:
        return
    dti = prompt_input("Debt-to-Income (DTI %)", "e.g. 20.0 (Float %)", 20.0, float, 0.0, 50.0)
    if dti == -1:
        return
    income = prompt_input("Annual Income ($)", "e.g. 65000.0 (Float $)", 65000.0, float, 10000.0, 1000000.0)
    if income == -1:
        return

    scenarios = [
        {"name": "1. Economic Boom (Low Unemp, Mild Inflation)", "unemp": 4.2, "cpi": 1.5, "chg": -0.8},
        {"name": "2. Calm Expansion (Historical Median Baseline)", "unemp": 5.0, "cpi": 2.0, "chg": 0.0},
        {"name": "3. Mild Economic Slowdown", "unemp": 6.5, "cpi": 2.8, "chg": 0.6},
        {"name": "4. Stagflationary Pressure", "unemp": 7.8, "cpi": 5.2, "chg": 1.1},
        {"name": "5. Severe Recessionary Shock (2009 Peak)", "unemp": 10.0, "cpi": 3.5, "chg": 2.2}
    ]

    print("\nEvaluating across 5 standardized economic stress scenarios (Model C vs Model D)...\n")
    results = []
    threshold = pkg.get("prevalence_threshold", 0.148863)

    for sc in scenarios:
        row_df = pd.DataFrame([{
            "annual_inc": income,
            "dti": dti,
            "fico_range_low": float(fico),
            "fico_range_high": float(fico + 4),
            "loan_amnt": 15000.0,
            "home_ownership": "RENT",
            "UNRATE": sc["unemp"],
            "CPI_YoY_Inflation": sc["cpi"],
            "UNRATE_change_6m": sc["chg"]
        }])
        pd_c = score_dataframe(row_df, pkg, model_key="MODEL_C")[0]
        pd_d = score_dataframe(row_df, pkg, model_key="MODEL_D")[0]
        results.append({
            "Scenario": sc["name"],
            "UNRATE": f"{sc['unemp']:.1f}%",
            "CPI": f"{sc['cpi']:.1f}%",
            "Model_C_PD": f"{pd_c:.2%}",
            "Model_D_PD": f"{pd_d:.2%}",
            "Decision_C": "DECLINE/REVIEW" if pd_c >= threshold else "APPROVE"
        })

    table_df = pd.DataFrame(results)
    print(table_df.to_string(index=False))
    print("=" * 80 + "\n")


def display_model_info(pkg):
    """Displays training cohort boundaries, holdout metrics, and feature lists."""
    print("\n" + "=" * 80)
    print("                    MODEL SPECIFICATION & HOLDOUT SCORECARD")
    print("=" * 80)
    print(f"Training Partition   : {pkg.get('training_cohort', '2007-2014 Window')}")
    print(f"Holdout Test Cohort  : {pkg.get('holdout_test_cohort', '2015 Window (283,026 loans)')}")
    print(f"Historical Base PD   : {pkg.get('baseline_pd', 0.13155):.4%}")
    print(f"Decision Threshold   : {pkg.get('prevalence_threshold', 0.148863):.4%}")
    print("-" * 80)
    print("Features Required:")
    print(f"  * Numerical  : {pkg['features_required']['numerical']}")
    print(f"  * Categorical: {pkg['features_required']['categorical']}")
    print("-" * 80)
    print("Available Model Architectures in Binary File:")
    for k in pkg.get("fitted_models", {}).keys():
        print(f"  [✓] {k}")
    print("=" * 80 + "\n")


def run_batch_scoring(pkg):
    """Prompt-based batch scoring for an external CSV file."""
    print("\n" + "=" * 80)
    print("                           BATCH CSV SCORING")
    print("=" * 80)
    csv_path = prompt_input("Path to input CSV file", "e.g. new_loans.csv", "sample.csv", str)
    if csv_path == "-1":
        return

    if not os.path.exists(csv_path):
        print(f"[!] File '{csv_path}' does not exist. Returning to main menu.")
        return

    out_path = prompt_input("Path for scored output CSV", "e.g. scored_loans.csv", "scored_loans.csv", str)
    if out_path == "-1":
        return

    try:
        df_batch = pd.read_csv(csv_path)
        print(f"Loaded {len(df_batch):,} rows. Scoring using Model C...")
        probs = score_dataframe(df_batch, pkg, model_key="MODEL_C")
        threshold = pkg.get("prevalence_threshold", 0.148863)
        df_batch["predicted_default_prob"] = np.round(probs, 5)
        df_batch["underwriting_decision"] = np.where(probs >= threshold, "FLAGGED_HIGH_RISK", "APPROVED")
        df_batch.to_csv(out_path, index=False)
        print(f"[SUCCESS] Scored {len(df_batch):,} loans. Output saved to '{out_path}'.\n")
    except Exception as e:
        print(f"[ERROR] Batch scoring failed: {e}\n")


def main_interactive_menu():
    """Main menu loop driven by user selection until -1 is entered."""
    try:
        pkg = load_model_package()
    except Exception as e:
        print(f"[FATAL ERROR] Could not load binary model package: {e}")
        sys.exit(1)

    while True:
        print("=" * 80)
        print("       LOAN DEFAULT RISK PREDICTOR & MACRO STRESS EXPLORER (ES 335)")
        print("=" * 80)
        print("Please choose an option:")
        print("  [1] Score Single Loan Applicant (Line-by-Line Interactive Input)")
        print("  [2] Run Macroeconomic Shock Simulation (Stress Testing)")
        print("  [3] Batch Score a CSV File")
        print("  [4] View Model Architecture & Holdout Benchmark Performance")
        print("  [-1] Exit Program")
        print("-" * 80)

        choice = input("Enter choice [1, 2, 3, 4, or -1 to exit]: ").strip()

        if choice in ["-1", "exit", "quit", "q"]:
            print("\nExiting Loan Default Risk Predictor. Goodbye!\n")
            break
        elif choice == "1":
            run_interactive_prediction(pkg)
        elif choice == "2":
            run_macro_stress_simulation(pkg)
        elif choice == "3":
            run_batch_scoring(pkg)
        elif choice == "4":
            display_model_info(pkg)
        else:
            print("\n[!] Unrecognized choice. Please enter 1, 2, 3, 4, or -1.\n")


def parse_cli_args():
    """Fallback parser to preserve non-interactive CLI scripting if flags are passed."""
    if len(sys.argv) > 1:
        # User supplied CLI arguments directly
        parser = argparse.ArgumentParser(description="Loan Default Risk Predictor CLI")
        parser.add_argument("--bin", type=str, default=MODEL_BIN_PATH)
        parser.add_argument("--model", type=str, default="MODEL_C")
        parser.add_argument("--csv", type=str, default=None)
        parser.add_argument("--output", type=str, default=None)
        parser.add_argument("--fico", type=float, default=690.0)
        parser.add_argument("--dti", type=float, default=18.0)
        parser.add_argument("--income", type=float, default=70000.0)
        parser.add_argument("--loan", type=float, default=15000.0)
        parser.add_argument("--home", type=str, default="RENT")
        parser.add_argument("--unemp", type=float, default=5.0)
        parser.add_argument("--cpi", type=float, default=2.0)
        parser.add_argument("--unemp-chg", type=float, default=0.0)

        args = parser.parse_args()
        pkg = load_model_package(args.bin)
        threshold = pkg.get("prevalence_threshold", 0.148863)

        if args.csv:
            df_b = pd.read_csv(args.csv)
            probs = score_dataframe(df_b, pkg, model_key=args.model)
            df_b["predicted_default_prob"] = np.round(probs, 5)
            df_b["decision"] = np.where(probs >= threshold, "FLAGGED_HIGH_RISK", "APPROVED")
            out_p = args.output or "scored_output.csv"
            df_b.to_csv(out_p, index=False)
            print(f"[SUCCESS] Scored {len(df_b):,} loans saved to '{out_p}'.")
            return

        row = pd.DataFrame([{
            "annual_inc": args.income,
            "dti": args.dti,
            "fico_range_low": args.fico,
            "fico_range_high": args.fico + 4.0,
            "loan_amnt": args.loan,
            "home_ownership": args.home,
            "UNRATE": args.unemp,
            "CPI_YoY_Inflation": args.cpi,
            "UNRATE_change_6m": args.unemp_chg
        }])
        prob = score_dataframe(row, pkg, model_key=args.model)[0]
        tier, rec, _ = assign_risk_category(prob, threshold)
        print(f"Predicted Probability of Default: {prob:.2%} | Tier: {tier} | Action: {rec}")
    else:
        # Launch interactive menu loop
        main_interactive_menu()


if __name__ == "__main__":
    parse_cli_args()
