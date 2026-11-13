from reportlab.platypus import Paragraph, Spacer, PageBreak, Table, TableStyle
from styles import styles, p, bullet, num_bullet, callout, divider, GRAY_BG, GRAY_BORDER, BLUE, NAVY, WHITE

def get_chapter3_story():
    story = []
    story.append(Paragraph("Chapter 3: Exhaustive Variable Dictionary — Every Feature Demystified", styles['ChapterTitle']))
    story.append(Paragraph("Definitions, Mathematical Formulations, Economic Intuition, and Empirical Beta Weights", styles['CoverSubtitle']))
    story.append(divider())

    story.append(p(
        "A critical test during an oral defense is whether you can explain every single feature in your model down to its official source, "
        "its mathematical transformation, its economic meaning, and why its coefficient had a specific sign and magnitude. "
        "This chapter provides an exhaustive breakdown of every variable utilized across our econometric pipeline."
    ))

    story.append(Paragraph("3.1 Systematic Macroeconomic Levers (FRED Time Series)", styles['SectionTitle']))
    
    story.append(Paragraph("1. National Civilian Unemployment Rate (UNRATE)", styles['SubsectionTitle']))
    story.append(p(
        "<b>Source:</b> Federal Reserve Bank of St. Louis (FRED) / U.S. Bureau of Labor Statistics (BLS).<br/>"
        "<b>Definition:</b> The percentage of the civilian labor force that is unemployed and actively seeking work within the past 4 weeks (CPS Household Survey).<br/>"
        "<b>Range in Data:</b> 3.8% to 10.0% (observed between 2007 and 2015).<br/>"
        "<b>Standardized Beta in Model C:</b> <code>β = -0.0350</code> (z = -1.63)."
    ))
    story.append(callout(
        "The Counter-Intuitive Negative Sign of Static UNRATE (The Flight-to-Quality Underwriting Defense)",
        [
            "In your oral interview, a professor might challenge you: <i>'Why is the coefficient for unemployment rate negative (β = -0.035)? Doesn't higher unemployment cause more defaults?'</i><br/><br/>"
            "<b>The Winning Economic Defense:</b> When static UNRATE was high (9–10% during 2009–2011), bank underwriters became extremely terrified! LendingClub drastically tightened approval standards (known as the <b>Flight-to-Quality effect</b>). "
            "Only the most pristine, high-cash-buffer borrowers were approved during those years. Consequently, loans originated during peak unemployment actually performed better than expected. "
            "Meanwhile, loan defaults are truly triggered by the <i>acute change</i> in unemployment, which is captured by our velocity variable below!"
        ],
        kind='amber'
    ))

    story.append(Paragraph("2. 6-Month Change in Unemployment (UNRATE_change_6m — Shock Velocity)", styles['SubsectionTitle']))
    story.append(p(
        "<b>Formula:</b> $\Delta U_t = U_t - U_{t-6}$<br/>"
        "<b>Definition:</b> The 6-month acceleration (first difference) in the national unemployment rate. Captures acute recessionary momentum.<br/>"
        "<b>Range in Data:</b> -1.2% to +2.4% (Great Recession shock velocity).<br/>"
        "<b>Standardized Beta in Model C:</b> <code>β = +0.0758</code> (z = +2.61, statistically significant at p < 0.01)."
    ))
    story.append(p(
        "<b>Economic Rationale:</b> A sudden jump in unemployment (+2.0% in 6 months) indicates mass corporate layoffs. "
        "Households have no time to adjust their expenditures, cut discretionary spending, or refinance debts. "
        "This positive coefficient confirms that <b>macroeconomic shock velocity</b> is the primary systematic driver of retail default spikes."
    ))

    story.append(Paragraph("3. Consumer Price Index Year-over-Year Inflation (CPI_YoY_Inflation)", styles['SubsectionTitle']))
    story.append(p(
        "<b>Source:</b> Bureau of Labor Statistics Series `CPIAUCSL` (Consumer Price Index for All Urban Consumers: All Items).<br/>"
        "<b>Mathematical Transformation:</b> Raw CPI is non-stationary (unit root $I(1)$ with a permanent upward drift). We transform raw CPI into stationary year-over-year percentage change:<br/>"
        "$$\\text{CPI\\_YoY\\_Inflation}_t = \\left(\\frac{\\text{CPIAUCSL}_t - \\text{CPIAUCSL}_{t-12}}{\\text{CPIAUCSL}_{t-12}}\\right) \\times 100$$<br/>"
        "<b>Standardized Beta in Model C:</b> <code>β = +0.0580</code> (z = +2.34, p < 0.05)."
    ))
    story.append(p(
        "<b>Economic Rationale:</b> Fixed installment loans require fixed monthly dollar payments. When inflation rises, "
        "the cost of essential survival goods (groceries, gasoline, heating, healthcare) increases. "
        "This directly compresses the household's <b>residual disposable income</b>, leaving less cash to service unsecured consumer debt."
    ))

    story.append(Paragraph("4. Point-in-Time (PIT) Publication Lag Shift Policy", styles['SubsectionTitle']))
    story.append(p(
        "Macroeconomic data is never available in real time. For example, the Bureau of Labor Statistics releases June unemployment data in mid-July, "
        "and preliminary CPI numbers are routinely revised months later. If a borrower originates a loan on June 2nd, the bank does NOT have June macro data."
    ))
    story.append(callout(
        "The 2-Month Lag Policy (`shift(2)`)",
        [
            "To strictly guarantee zero look-ahead bias, all macro series in our pipeline are shifted backwards by a conservative <b>2-month publication lag</b> (`shift(2)`). "
            "A loan originated in June is merged with official macro data published in April. This aligns with institutional Basel III Point-in-Time standards."
        ],
        kind='tip'
    ))

    story.append(Paragraph("3.2 Borrower Micro-Data Levers (LendingClub Micro-Data)", styles['SectionTitle']))

    story.append(Paragraph("1. FICO Credit Score (fico_avg)", styles['SubsectionTitle']))
    story.append(p(
        "<b>Definition:</b> The credit score developed by Fair Isaac Corporation, ranging from 300 to 850. In our data, LendingClub reports `fico_range_low` and `fico_range_high` (a 4-point band); we calculate the midpoint `fico_avg`.<br/>"
        "<b>FICO Mathematical Composition:</b>"
    ))
    story.append(bullet("<b>Payment History (35%):</b> Presence of late payments, delinquencies, charge-offs, or collections."))
    story.append(bullet("<b>Amounts Owed / Credit Utilization (30%):</b> Total debt outstanding and revolving balance utilization."))
    story.append(bullet("<b>Length of Credit History (15%):</b> Age of oldest account, average age of all accounts."))
    story.append(bullet("<b>New Credit & Inquiries (10%):</b> Number of recently opened accounts and hard credit checks."))
    story.append(bullet("<b>Credit Mix (10%):</b> Blend of installment loans, revolving cards, and retail accounts."))
    story.append(p(
        "<b>Standardized Beta in Model C:</b> <code>β = -0.4011</code> (z = -18.72, p < 0.0001). "
        "This is by far the single most powerful negative predictor of default in the entire model. For every standard deviation increase in FICO (~30 points), the log-odds of default decline by 0.40."
    ))

    story.append(Paragraph("2. Debt-to-Income Ratio (dti)", styles['SubsectionTitle']))
    story.append(p(
        "<b>Definition:</b> Total monthly debt payments (excluding mortgage and the requested loan) divided by gross monthly income:<br/>"
        "$$\\text{DTI} = \\frac{\\sum \\text{Monthly Non-Housing Debt Payments}}{\\text{Gross Monthly Income}} \\times 100$$<br/>"
        "<b>Standardized Beta in Model C:</b> <code>β = +0.1458</code> (z = +8.41, p < 0.0001)."
    ))
    story.append(p(
        "<b>Economic Rationale:</b> A high DTI indicates a borrower who is already over-leveraged. If a borrower has a DTI of 35%, more than a third of their pre-tax income goes directly to debt servicing before food, rent, or utilities."
    ))

    story.append(Paragraph("3. Annual Income (annual_inc & annual_inc_log)", styles['SubsectionTitle']))
    story.append(p(
        "<b>Definition:</b> Self-reported gross annual earnings.<br/>"
        "<b>Mathematical Transformation:</b> Raw income exhibits extreme positive right-skew (Pareto-tail distribution with values exceeding $1,000,000). "
        "Fitting linear or logistic models on raw income causes high-earning outliers to exert massive leverage on the gradient. "
        "We apply the log transformation:<br/>"
        "$$\\text{annual\\_inc\\_log} = \\ln(1 + \\text{annual\\_inc})$$<br/>"
        "<b>Standardized Beta in Model C:</b> <code>β = -0.2454</code> (z = -12.30, p < 0.0001). Strong negative predictor; higher earning capacity provides deep emergency reserves."
    ))

    story.append(Paragraph("4. Other Pre-Origination Application Features", styles['SubsectionTitle']))
    story.append(bullet("<b>Revolving Credit Utilization (`revol_util`):</b> The percentage of available revolving credit lines currently drawn. High utilization (>70%) signals liquidity strain."))
    story.append(bullet("<b>Public Derogatory Records (`pub_rec`):</b> Number of legal derogatory events, including civil judgments, tax liens, or bankruptcies. Strong positive default predictor."))
    story.append(bullet("<b>Number of Open Accounts (`open_acc`):</b> Number of active credit lines."))
    story.append(bullet("<b>Mortgage Accounts (`mort_acc`):</b> Number of mortgage accounts; strong indicator of homeownership stability and residential equity."))
    story.append(bullet("<b>Home Ownership (`home_ownership`):</b> Categorical indicator (MORTGAGE, RENT, OWN). In Model C, RENT is the reference category; homeowners with mortgages have significantly lower default propensity."))
    story.append(PageBreak())
    return story
