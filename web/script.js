// Client-side Logistic Sigmoid Inference Engine for Macroeconomic Stress Testing
// Formula: P(Default) = 1 / (1 + exp(-(beta_0 + sum(beta_i * X_i))))

// Default baseline coefficients (can also be loaded dynamically from model_coefficients.json)
let modelParams = {
  intercept: -1.85,
  weights: {
    fico_score: -0.0042,
    dti: 0.0315,
    annual_income_log: -0.22,
    unemployment_rate: 0.145,
    cpi_inflation_rate: 0.082,
    fed_funds_rate: 0.051,
    dti_x_inflation: 0.0075
  }
};

// Benign economic baseline values for comparison
const BENIGN_BASELINE = {
  unemployment: 4.5,
  cpi: 2.0,
  fedfunds: 1.5,
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
      modelParams.intercept = data.intercept;
      modelParams.weights = data.weights;
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
const fedfundsSlider = document.getElementById("fedfunds-slider");
const fedfundsVal = document.getElementById("fedfunds-val");

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

function computePD(unemp, cpi, fed, fico, dti, income) {
  const logIncome = Math.log(Math.max(income, 1000));
  const interaction = (dti * cpi) / 100;

  const z =
    modelParams.intercept +
    modelParams.weights.fico_score * (fico - 600) +
    modelParams.weights.dti * (dti - 15) +
    modelParams.weights.annual_income_log * (logIncome - 11.0) +
    modelParams.weights.unemployment_rate * (unemp - 4.5) +
    modelParams.weights.cpi_inflation_rate * (cpi - 2.0) +
    modelParams.weights.fed_funds_rate * (fed - 1.5) +
    modelParams.weights.dti_x_inflation * interaction;

  return sigmoid(z);
}

function calculateRisk() {
  const unemp = parseFloat(unemploymentSlider.value);
  const cpi = parseFloat(cpiSlider.value);
  const fed = parseFloat(fedfundsSlider.value);

  const fico = parseFloat(ficoInput.value) || 690;
  const dti = parseFloat(dtiInput.value) || 18;
  const income = parseFloat(incomeInput.value) || 70000;

  // Update slider text displays
  unemploymentVal.textContent = `${unemp.toFixed(1)}%`;
  cpiVal.textContent = `${cpi.toFixed(1)}%`;
  fedfundsVal.textContent = `${fed.toFixed(2)}%`;

  // Compute current PD
  const currentPD = computePD(unemp, cpi, fed, fico, dti, income);
  const pdFormatted = (currentPD * 100).toFixed(1);
  pdPercentage.textContent = `${pdFormatted}%`;

  // Compute benign baseline PD for delta
  const baselinePD = computePD(
    BENIGN_BASELINE.unemployment,
    BENIGN_BASELINE.cpi,
    BENIGN_BASELINE.fedfunds,
    fico,
    dti,
    income
  );

  const deltaPct = ((currentPD - baselinePD) / baselinePD) * 100;
  const sign = deltaPct >= 0 ? "+" : "";
  deltaSummary.innerHTML = `Default odds changed by <strong>${sign}${deltaPct.toFixed(1)}%</strong> relative to benign baseline.`;

  // Update Gauge and Risk Badge styling
  if (currentPD < 0.12) {
    riskGauge.style.borderColor = "#10b981";
    riskStatus.style.backgroundColor = "#10b981";
    riskStatus.textContent = "Low Risk (Grade A/B)";
  } else if (currentPD < 0.22) {
    riskGauge.style.borderColor = "#f59e0b";
    riskStatus.style.backgroundColor = "#f59e0b";
    riskStatus.textContent = "Moderate Risk (Grade C/D)";
  } else {
    riskGauge.style.borderColor = "#ef4444";
    riskStatus.style.backgroundColor = "#ef4444";
    riskStatus.textContent = "Elevated Stress / High Risk";
  }
}

// Event Listeners
[unemploymentSlider, cpiSlider, fedfundsSlider].forEach((el) => {
  el.addEventListener("input", calculateRisk);
});

[ficoInput, dtiInput, incomeInput].forEach((el) => {
  el.addEventListener("input", calculateRisk);
});

// Initial calculation on load
calculateRisk();

