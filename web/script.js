// Client-side Logistic Sigmoid Inference Engine for Macroeconomic Sensitivity Explorer
// Strictly evaluates Model C (Borrower + Macro Logistic Regression)
// Formula: P(Default) = 1 / (1 + exp(-(beta_0 + sum(beta_i * (X_i - mu_i) / sigma_i))))
// Note: Model D (LightGBM) is evaluated offline in Python; tree ensembles are NOT approximated by this sigmoid formula.

// Default baseline parameters (populated dynamically from model_coefficients.json once Model C is trained)
let modelParams = {
  model_id: "Model_C_Logit",
  description: "Fitted Model C (Borrower + Macro) Logistic Regression Parameters",
  intercept: -1.75,
  weights: {
    fico_score: -0.45,
    dti: 0.32,
    annual_income_log: -0.28,
    unemployment_rate: 0.25,
    cpi_inflation_rate: 0.12,
    unemp_change_6m: 0.18
  },
  variable_means: {
    fico_score: 698.0,
    dti: 17.5,
    annual_income_log: 11.12,
    unemployment_rate: 6.8,
    cpi_inflation_rate: 1.8,
    unemp_change_6m: 0.05
  },
  variable_stds: {
    fico_score: 32.0,
    dti: 8.2,
    annual_income_log: 0.55,
    unemployment_rate: 1.9,
    cpi_inflation_rate: 1.4,
    unemp_change_6m: 0.65
  }
};

// Benign economic baseline values for comparison
const BENIGN_BASELINE = {
  unemployment: 5.0,
  cpi: 2.0,
  unemp_change: 0.0,
  fico: 690,
  dti: 18,
  income: 70000
};

// Attempt to fetch dynamically exported coefficients from model_coefficients.json
fetch("model_coefficients.json")
  .then((res) => {
    if (res.ok) return res.json();
    throw new Error("Local JSON fetch skipped (using embedded fallback)");
  })
  .then((data) => {
    if (data.intercept && data.weights) {
      modelParams = data;
      calculateRisk();
    }
  })
  .catch((err) => {
    console.log("Using built-in model parameters:", err.message);
  });

// DOM Elements
const unemploymentSlider = document.getElementById("unemployment-slider");
const unemploymentVal = document.getElementById("unemployment-val");
const cpiSlider = document.getElementById("cpi-slider");
const cpiVal = document.getElementById("cpi-val");
const unempChangeSlider = document.getElementById("unemp-change-slider");
const unempChangeVal = document.getElementById("unemp-change-val");

const ficoInput = document.getElementById("fico-input");
const dtiInput = document.getElementById("dti-input");
const incomeInput = document.getElementById("income-input");

const pdPercentage = document.getElementById("pd-percentage");
const riskGauge = document.getElementById("risk-gauge");
const riskStatus = document.getElementById("risk-status");
const deltaSummary = document.getElementById("delta-summary");

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function computePD(unemp, cpi, unempChange, fico, dti, income) {
  const logIncome = Math.log(Math.max(income, 1000));
  
  // Standardized inputs using training partition parameters
  const z_fico = (fico - modelParams.variable_means.fico_score) / (modelParams.variable_stds?.fico_score || 32);
  const z_dti = (dti - modelParams.variable_means.dti) / (modelParams.variable_stds?.dti || 8.2);
  const z_income = (logIncome - modelParams.variable_means.annual_income_log) / (modelParams.variable_stds?.annual_income_log || 0.55);
  const z_unemp = (unemp - modelParams.variable_means.unemployment_rate) / (modelParams.variable_stds?.unemployment_rate || 1.9);
  const z_cpi = (cpi - modelParams.variable_means.cpi_inflation_rate) / (modelParams.variable_stds?.cpi_inflation_rate || 1.4);
  const z_unemp_change = (unempChange - modelParams.variable_means.unemp_change_6m) / (modelParams.variable_stds?.unemp_change_6m || 0.65);

  const z =
    modelParams.intercept +
    modelParams.weights.fico_score * z_fico +
    modelParams.weights.dti * z_dti +
    modelParams.weights.annual_income_log * z_income +
    modelParams.weights.unemployment_rate * z_unemp +
    modelParams.weights.cpi_inflation_rate * z_cpi +
    modelParams.weights.unemp_change_6m * z_unemp_change;

  return sigmoid(z);
}

function calculateRisk() {
  const unemp = parseFloat(unemploymentSlider.value);
  const cpi = parseFloat(cpiSlider.value);
  const unempChange = parseFloat(unempChangeSlider.value);

  const fico = parseFloat(ficoInput.value) || 690;
  const dti = parseFloat(dtiInput.value) || 18;
  const income = parseFloat(incomeInput.value) || 70000;

  // Update slider text displays
  unemploymentVal.textContent = `${unemp.toFixed(1)}%`;
  cpiVal.textContent = `${cpi.toFixed(1)}%`;
  unempChangeVal.textContent = `${unempChange >= 0 ? "+" : ""}${unempChange.toFixed(1)}%`;

  // Compute current PD
  const currentPD = computePD(unemp, cpi, unempChange, fico, dti, income);
  const pdFormatted = (currentPD * 100).toFixed(1);
  pdPercentage.textContent = `${pdFormatted}%`;

  // Compute baseline PD under benign economic conditions
  const baselinePD = computePD(
    BENIGN_BASELINE.unemployment,
    BENIGN_BASELINE.cpi,
    BENIGN_BASELINE.unemp_change,
    fico,
    dti,
    income
  );

  // Update Risk Status Badge and Gauge Appearance
  let statusText = "Low Risk";
  let statusClass = "low";
  let gaugeColor = "#10b981"; // green

  if (currentPD >= 0.22) {
    statusText = "Severe Default Risk";
    statusClass = "severe";
    gaugeColor = "#ef4444"; // red
  } else if (currentPD >= 0.16) {
    statusText = "Elevated Default Risk";
    statusClass = "high";
    gaugeColor = "#f59e0b"; // amber
  } else if (currentPD >= 0.11) {
    statusText = "Moderate Default Risk";
    statusClass = "moderate";
    gaugeColor = "#3b82f6"; // blue
  }

  riskStatus.textContent = statusText;
  riskStatus.className = `risk-badge ${statusClass}`;
  riskGauge.style.borderColor = gaugeColor;

  // Scenario Delta Summary
  const oddsCurrent = currentPD / (1 - currentPD);
  const oddsBaseline = baselinePD / (1 - baselinePD);
  const oddsDeltaPct = ((oddsCurrent / oddsBaseline - 1) * 100).toFixed(1);
  const sign = oddsDeltaPct >= 0 ? "+" : "";

  deltaSummary.innerHTML = `Under these macroeconomic origination conditions, estimated default odds change by <strong>${sign}${oddsDeltaPct}%</strong> relative to a benign baseline.`;
}

// Attach Event Listeners
[unemploymentSlider, cpiSlider, unempChangeSlider].forEach((slider) => {
  slider.addEventListener("input", calculateRisk);
});

[ficoInput, dtiInput, incomeInput].forEach((input) => {
  input.addEventListener("input", calculateRisk);
});

// Initial calculation on page load
calculateRisk();
