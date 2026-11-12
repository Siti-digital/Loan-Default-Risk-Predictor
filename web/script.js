// Client-side Logistic Sigmoid Inference Engine with Speedometer Gauge
// Evaluates Model C (Borrower + Macro Logistic Regression)
// Formula: P(Default) = 1 / (1 + exp(-(effective_intercept + sum(beta_i * (X_i - mu_i) / sigma_i))))

// Default fallback parameters (derived from 2007-2014 training window)
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
const predictBtn = document.getElementById("predict-btn");

const needle = document.getElementById("speedometer-needle");
const needleTip = document.getElementById("needle-tip");
const pdPercentage = document.getElementById("pd-percentage");
const riskStatus = document.getElementById("risk-status");
const deltaSummary = document.getElementById("delta-summary");

function sigmoid(z) {
  return 1.0 / (1.0 + Math.exp(-z));
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
  const unemp = parseFloat(unemploymentSlider ? unemploymentSlider.value : 5.0);
  const cpi = parseFloat(cpiSlider ? cpiSlider.value : 2.0);
  const unempChange = parseFloat(unempChangeSlider ? unempChangeSlider.value : 0.0);

  const fico = parseFloat(ficoInput ? ficoInput.value : 690) || 690;
  const dti = parseFloat(dtiInput ? dtiInput.value : 18) || 18;
  const income = parseFloat(incomeInput ? incomeInput.value : 70000) || 70000;

  // Update slider label texts
  updateSliderLabels();

  // Compute current PD
  const currentPD = computePD(unemp, cpi, unempChange, fico, dti, income);
  const pdFormatted = (currentPD * 100).toFixed(1);
  if (pdPercentage) {
    pdPercentage.textContent = `${pdFormatted}%`;
  }

  // Compute baseline PD under benign economic conditions
  const baselinePD = computePD(
    BENIGN_BASELINE.unemployment,
    BENIGN_BASELINE.cpi,
    BENIGN_BASELINE.unemp_change,
    fico,
    dti,
    income
  );

  // Speedometer Needle Angle Calculation:
  // Green Zone:   0.0% to 10.0% PD -> -90 deg to -45 deg
  // Blue Zone:   10.0% to 16.0% PD -> -45 deg to   0 deg
  // Orange Zone: 16.0% to 22.0% PD ->   0 deg to +45 deg
  // Red Zone:    22.0%+ PD         -> +45 deg to +90 deg
  let needleAngle = -90;
  let statusText = "Low Risk";
  let statusClass = "low";
  let gaugeColor = "#10b981"; // green

  if (currentPD <= 0.10) {
    needleAngle = -90 + (currentPD / 0.10) * 45;
    statusText = "Low Risk";
    statusClass = "low";
    gaugeColor = "#10b981"; // green
  } else if (currentPD <= 0.16) {
    needleAngle = -45 + ((currentPD - 0.10) / 0.06) * 45;
    statusText = "Moderate Risk";
    statusClass = "moderate";
    gaugeColor = "#3b82f6"; // blue
  } else if (currentPD <= 0.22) {
    needleAngle = 0 + ((currentPD - 0.16) / 0.06) * 45;
    statusText = "Elevated Default Risk";
    statusClass = "high";
    gaugeColor = "#f59e0b"; // amber
  } else {
    const extra = Math.min((currentPD - 0.22) / 0.18, 1.0);
    needleAngle = 45 + extra * 45;
    statusText = "Severe Default Risk";
    statusClass = "severe";
    gaugeColor = "#ef4444"; // red
  }

  // Clamp needle angle to [-90, +90]
  needleAngle = Math.max(-90, Math.min(90, needleAngle));

  // Rotate speedometer needle
  if (needle) {
    needle.style.transform = `rotate(${needleAngle.toFixed(1)}deg)`;
  }
  if (needleTip) {
    needleTip.setAttribute("fill", gaugeColor);
  }

  // Update Status Badge - matching the exact color the pointer is pointing to
  if (riskStatus) {
    riskStatus.textContent = statusText;
    riskStatus.className = `risk-badge ${statusClass}`;
    riskStatus.style.backgroundColor = gaugeColor;
  }

  // Update Scenario Delta Summary
  const oddsCurrent = currentPD / (1.0 - currentPD);
  const oddsBaseline = baselinePD / (1.0 - baselinePD);
  const oddsDeltaPct = ((oddsCurrent / oddsBaseline - 1.0) * 100.0).toFixed(1);
  const sign = oddsDeltaPct >= 0 ? "+" : "";

  if (deltaSummary) {
    deltaSummary.innerHTML = `Under these macroeconomic origination conditions, estimated default odds change by <strong>${sign}${oddsDeltaPct}%</strong> relative to a benign baseline.`;
  }
}

function updateSliderLabels() {
  if (unemploymentSlider && unemploymentVal) {
    const unemp = parseFloat(unemploymentSlider.value);
    unemploymentVal.textContent = `${unemp.toFixed(1)}%`;
  }
  if (cpiSlider && cpiVal) {
    const cpi = parseFloat(cpiSlider.value);
    cpiVal.textContent = `${cpi.toFixed(1)}%`;
  }
  if (unempChangeSlider && unempChangeVal) {
    const chg = parseFloat(unempChangeSlider.value);
    unempChangeVal.textContent = `${chg >= 0 ? "+" : ""}${chg.toFixed(1)}%`;
  }
}

// Live update labels on drag for all sliders
if (unemploymentSlider) {
  unemploymentSlider.addEventListener("input", updateSliderLabels);
  unemploymentSlider.addEventListener("change", updateSliderLabels);
}

if (cpiSlider) {
  cpiSlider.addEventListener("input", updateSliderLabels);
  cpiSlider.addEventListener("change", updateSliderLabels);
}

if (unempChangeSlider) {
  unempChangeSlider.addEventListener("input", updateSliderLabels);
  unempChangeSlider.addEventListener("change", updateSliderLabels);
}

// Predict button click listener
if (predictBtn) {
  predictBtn.addEventListener("click", () => {
    predictBtn.style.transform = "scale(0.97)";
    setTimeout(() => {
      predictBtn.style.transform = "";
    }, 120);
    calculateRisk();
  });
}

// Clear highlight from all preset buttons
function clearPresetHighlights() {
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.classList.remove("active");
  });
}

// Quick Stress Scenario Presets Listener
document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    clearPresetHighlights();
    btn.classList.add("active");

    if (unemploymentSlider) unemploymentSlider.value = btn.dataset.unemp;
    if (cpiSlider) cpiSlider.value = btn.dataset.cpi;
    if (unempChangeSlider) unempChangeSlider.value = btn.dataset.chg;
    updateSliderLabels();
    calculateRisk();
  });
});

// Remove preset highlight if the user adjusts ANY individual parameter
const parameterInputs = [
  unemploymentSlider,
  cpiSlider,
  unempChangeSlider,
  ficoInput,
  dtiInput,
  incomeInput
];

parameterInputs.forEach((input) => {
  if (input) {
    input.addEventListener("input", clearPresetHighlights);
    input.addEventListener("change", clearPresetHighlights);
  }
});

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

// Fetch holdout benchmark results to populate/refresh the model comparison table if served via HTTP/S
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
      const isModelC = m.Model === "MODEL_C";
      const tr = document.createElement("tr");
      tr.style.borderBottom = "1px solid #e2e8f0";
      if (isModelC) {
        tr.style.backgroundColor = "#f0fdf4";
      }
      tr.innerHTML = `
        <td style="padding: 0.5rem 0.75rem; font-weight: ${isModelC ? "700" : "600"}; ${isModelC ? "color: #15803d;" : ""}">${m.Model}${isModelC ? ' <span style="font-size: 0.7rem; background: #22c55e; color: #fff; padding: 1px 5px; border-radius: 4px; margin-left: 4px;">CHOSEN</span>' : ''}</td>
        <td style="padding: 0.5rem 0.75rem; color: ${isModelC ? "#166534" : "#64748b"}; font-weight: ${isModelC ? "500" : "normal"};">${m.Calibration || "unweighted"}</td>
        <td style="padding: 0.5rem 0.75rem; font-family: monospace; font-weight: ${isModelC ? "600" : "normal"};">${m["PR-AUC"].toFixed(4)}</td>
        <td style="padding: 0.5rem 0.75rem; font-family: monospace; font-weight: ${isModelC ? "600" : "normal"};">${m["ROC-AUC"].toFixed(4)}</td>
        <td style="padding: 0.5rem 0.75rem; font-family: monospace; font-weight: ${isModelC ? "600" : "normal"};">${m["Brier_Score"].toFixed(4)}</td>
        <td style="padding: 0.5rem 0.75rem; font-family: monospace; font-weight: ${isModelC ? "600" : "normal"};">${(m["Mean_Pred_PD"] * 100).toFixed(2)}%</td>
        <td style="padding: 0.5rem 0.75rem; font-family: monospace; font-weight: ${isModelC ? "600" : "normal"};">${(m["Actual_DR"] * 100).toFixed(2)}%</td>
      `;
      tbody.appendChild(tr);
    });
  })
  .catch((err) => {
    // Falls back seamlessly to the statically embedded rows in index.html
    console.log("Using static benchmark table rows:", err.message);
  });

// Initial calculation on page load
updateSliderLabels();
calculateRisk();
