// Client-side Logistic Sigmoid Inference Engine for Macroeconomic Sensitivity Explorer
// Strictly evaluates Model C (Borrower + Macro Logistic Regression)
// Formula: P(Default) = 1 / (1 + exp(-(effective_intercept + sum(beta_i * (X_i - mu_i) / sigma_i))))
// Note: Model D (LightGBM) is evaluated offline in Python; tree ensembles are NOT approximated by this sigmoid formula.

// Default fallback parameters (populated dynamically from model_coefficients.json once exported)
let modelParams = {
  model_id: "Model_C_Logit",
  description: "Fitted Model C (Borrower + Macro) Logistic Regression Parameters (Trained on 2007-2014)",
  effective_intercept: -1.89598,
  weights: {
    fico_score: -0.401139,
    dti: 0.145846,
    annual_income: -0.245381,
    unemployment_rate: -0.034996,
    cpi_inflation_rate: 0.057986,
    unemp_change_6m: 0.075759
  },
  variable_means: {
    fico_score: 693.385,
    dti: 17.525,
    annual_income: 72836.4,
    unemployment_rate: 7.738,
    cpi_inflation_rate: 1.865,
    unemp_change_6m: -0.198
  },
  variable_stds: {
    fico_score: 30.134,
    dti: 7.892,
    annual_income: 55595.2,
    unemployment_rate: 1.637,
    cpi_inflation_rate: 1.346,
    unemp_change_6m: 0.584
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

// Fetch dynamically exported coefficients from model_coefficients.json
fetch("model_coefficients.json")
  .then((res) => {
    if (res.ok) return res.json();
    throw new Error("Local JSON fetch skipped (using embedded fallback)");
  })
  .then((data) => {
    if (data.weights) {
      modelParams = data;
      calculateRisk();
    }
  })
  .catch((err) => {
    console.log("Using built-in model parameters:", err.message);
  });

// Fetch holdout benchmark results to populate the model comparison table
fetch("benchmark_results.json")
  .then((res) => {
    if (res.ok) return res.json();
    throw new Error("Local benchmark JSON not found");
  })
  .then((data) => {
    const tbody = document.getElementById("benchmark-table-body");
    if (!tbody || !data.models) return;
    tbody.innerHTML = "";
    data.models.forEach((m) => {
      const tr = document.createElement("tr");
      tr.style.borderBottom = "1px solid #e2e8f0";
      tr.innerHTML = `
        <td style="padding: 0.5rem 0.75rem; font-weight: 600;">${m.Model}</td>
        <td style="padding: 0.5rem 0.75rem; color: #64748b;">${m.Calibration || "unweighted"}</td>
        <td style="padding: 0.5rem 0.75rem; font-family: monospace;">${m["PR-AUC"].toFixed(4)}</td>
        <td style="padding: 0.5rem 0.75rem; font-family: monospace;">${m["ROC-AUC"].toFixed(4)}</td>
        <td style="padding: 0.5rem 0.75rem; font-family: monospace;">${m["Brier_Score"].toFixed(4)}</td>
        <td style="padding: 0.5rem 0.75rem; font-family: monospace;">${(m["Mean_Pred_PD"] * 100).toFixed(2)}%</td>
        <td style="padding: 0.5rem 0.75rem; font-family: monospace;">${(m["Actual_DR"] * 100).toFixed(2)}%</td>
      `;
      tbody.appendChild(tr);
    });
  })
  .catch((err) => {
    console.log("Benchmark table loader:", err.message);
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
  const effectiveIntercept =
    modelParams.effective_intercept !== undefined
      ? modelParams.effective_intercept
      : modelParams.intercept;

  // Standardized inputs using training partition parameters
  const z_fico = (fico - modelParams.variable_means.fico_score) / (modelParams.variable_stds?.fico_score || 30.134);
  const z_dti = (dti - modelParams.variable_means.dti) / (modelParams.variable_stds?.dti || 7.892);

  const incMean =
    modelParams.variable_means.annual_income !== undefined
      ? modelParams.variable_means.annual_income
      : (modelParams.variable_means.annual_income_log || 72836.4);
  const incStd =
    modelParams.variable_stds?.annual_income !== undefined
      ? modelParams.variable_stds.annual_income
      : (modelParams.variable_stds?.annual_income_log || 55595.2);
  const z_income = (income - incMean) / incStd;
  const w_income =
    modelParams.weights.annual_income !== undefined
      ? modelParams.weights.annual_income
      : modelParams.weights.annual_income_log;

  const z_unemp = (unemp - modelParams.variable_means.unemployment_rate) / (modelParams.variable_stds?.unemployment_rate || 1.637);
  const z_cpi = (cpi - modelParams.variable_means.cpi_inflation_rate) / (modelParams.variable_stds?.cpi_inflation_rate || 1.346);
  const z_unemp_change = (unempChange - modelParams.variable_means.unemp_change_6m) / (modelParams.variable_stds?.unemp_change_6m || 0.584);

  const z =
    effectiveIntercept +
    modelParams.weights.fico_score * z_fico +
    modelParams.weights.dti * z_dti +
    w_income * z_income +
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
