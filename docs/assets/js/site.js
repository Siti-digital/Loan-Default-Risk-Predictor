/* Project site renderer. All numbers come from assets/data/site-data.js, written by the notebook (Phase 7). */
(function () {
  'use strict';
  var D = window.SITE_DATA;
  if (!D) { console.error('site-data.js missing'); return; }

  // ---------------------------------------------------------------- helpers
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var css = function (v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); };
  var nf = function (x, d) { return Number(x).toLocaleString('en-US', { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 }); };
  var pct = function (x, d) { return (100 * x).toFixed(d == null ? 1 : d) + '%'; };
  var fx = function (x, d) { return Number(x).toFixed(d == null ? 4 : d); };
  var sgn = function (x, d) { var s = Number(x).toFixed(d == null ? 4 : d); return (x > 0 ? '+' : x < 0 ? '−' : '') + s.replace('-', ''); };
  var by = function (arr, key) { var o = {}; arr.forEach(function (r) { o[r[key]] = r; }); return o; };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };

  var NAMES = {
    MODEL_A: 'A · Baseline', MODEL_B: 'B · Borrower (LR)', MODEL_C: 'C · Borrower + macro (LR)',
    MODEL_D_BASE: 'D-base · Borrower (LightGBM)', MODEL_D: 'D · Borrower + macro (LightGBM)',
    LENDER_BENCHMARK: 'Lender benchmark', MODEL_C_ABLATION: 'C + interactions', MODEL_M: 'Macro only',
    MODEL_C_BALANCED_CHECK: 'C, balanced weights'
  };
  var SHORT = { MODEL_A: 'A', MODEL_B: 'B', MODEL_C: 'C', MODEL_D_BASE: 'D-base', MODEL_D: 'D', LENDER_BENCHMARK: 'Lender', MODEL_M: 'Macro only', MODEL_C_ABLATION: 'C + inter.', MODEL_C_BALANCED_CHECK: 'C balanced' };
  var CVAR = { MODEL_A: '--s-a', MODEL_B: '--s-b', MODEL_C: '--s-c', MODEL_D_BASE: '--s-dbase', MODEL_D: '--s-d', LENDER_BENCHMARK: '--s-l', ACTUAL: '--s-actual' };
  var col = function (k) { return css(CVAR[k] || '--muted'); };
  var FEAT = {
    fico_avg: 'FICO score', dti: 'Debt-to-income (%)', annual_inc: 'Annual income ($)', loan_amnt: 'Loan amount ($)',
    revol_util: 'Revolving utilisation (%)', open_acc: 'Open credit lines', mort_acc: 'Mortgage accounts', pub_rec: 'Public derogatory records',
    UNRATE: 'Unemployment rate (%)', UNRATE_change_6m: '6-month change in unemployment (pp)', CPI_YoY_Inflation: 'CPI inflation, YoY (%)'
  };
  var MAIN = ['MODEL_A', 'MODEL_B', 'MODEL_C', 'MODEL_D_BASE', 'MODEL_D', 'LENDER_BENCHMARK'];

  var H = by(D.holdout, 'Model');
  var CVM = by(D.cv_mean, 'Model');
  var BOOT = D.bootstrap;
  var LOGIT = by(D.logit, 'term');

  // ---------------------------------------------------------------- bound numbers
  var bootRow = function (i) { return BOOT[i]; };
  var selTag = function (k) { var s = D.meta.selected[k]; if (!s) return '–'; return Object.keys(s).map(function (p) { return p + '=' + s[p]; }).join(', '); };
  var pre = D.preproc_check;
  var logGain = (function () {
    var g = 0, n = 0;
    ['MODEL_B', 'MODEL_C'].forEach(function (m) {
      var raw = pre.filter(function (r) { return r.Model === m && r['Income transform'].indexOf('raw') === 0; })[0];
      var lg = pre.filter(function (r) { return r.Model === m && r['Income transform'].indexOf('log') === 0; })[0];
      if (raw && lg) { g += lg['PR-AUC'] - raw['PR-AUC']; n++; }
    });
    return n ? g / n : 0;
  })();
  var macroOR = Object.keys(LOGIT).filter(function (k) { return /UNRATE|CPI/.test(k); }).map(function (k) { return LOGIT[k].odds_ratio; });
  var ficoKey = Object.keys(LOGIT).filter(function (k) { return /fico/.test(k); })[0];
  var K = {
    raw_loans: nf(D.meta.raw_loans), cohort_loans: nf(D.meta.cohort_loans), train_loans: nf(D.meta.train_loans), test_loans: nf(D.meta.test_loans),
    train_dr: pct(D.meta.train_dr), test_dr: pct(D.meta.test_dr), train_months: String(D.meta.train_months), tau: pct(D.meta.tau),
    boot_reps: String(D.meta.bootstrap_reps), generated: D.generated,
    a_pr: fx(H.MODEL_A['PR-AUC'], 3), b_pr4: fx(H.MODEL_B['PR-AUC'], 4), c_pr4: fx(H.MODEL_C['PR-AUC'], 4), b_pr: fx(H.MODEL_B['PR-AUC'], 3), c_pr: fx(H.MODEL_C['PR-AUC'], 3), l_pr: fx(H.LENDER_BENCHMARK['PR-AUC'], 3),
    b_lift: Math.round(100 * (H.MODEL_B['PR-AUC'] / H.MODEL_A['PR-AUC'] - 1)) + '%',
    d_pr_cb: sgn(H.MODEL_C['PR-AUC'] - H.MODEL_B['PR-AUC']), d_roc_cb: sgn(H.MODEL_C['ROC-AUC'] - H.MODEL_B['ROC-AUC']),
    ci_pr: '[' + sgn(bootRow(0)['CI_2.5%'], 4) + ', ' + sgn(bootRow(0)['CI_97.5%'], 4) + ']',
    ci_pr_text: sgn(bootRow(0)['CI_2.5%'], 4) + ' to ' + sgn(bootRow(0)['CI_97.5%'], 4),
    corr: sgn(D.meta.macro_shift_corr, 2),
    bal_mean: pct(H.MODEL_C_BALANCED_CHECK.Mean_Pred_PD, 0), bal_brier: fx(H.MODEL_C_BALANCED_CHECK.Brier_Score, 3),
    sel_b: selTag('MODEL_B'), sel_c: selTag('MODEL_C'), sel_l: selTag('LENDER_BENCHMARK'), sel_d: selTag('MODEL_D'), sel_dbase: selTag('MODEL_D_BASE'),
    skew_raw: Math.round(D.quality_notes.annual_inc_skew_raw), skew_log: fx(D.quality_notes.annual_inc_skew_log, 1), log_gain: sgn(logGain, 4),
    or_fico: ficoKey ? fx(LOGIT[ficoKey].odds_ratio, 2) : '–',
    or_macro_range: macroOR.length ? fx(Math.min.apply(null, macroOR), 2) + '–' + fx(Math.max.apply(null, macroOR), 2) : '–'
  };
  $$('[data-k]').forEach(function (el) { var k = el.getAttribute('data-k'); if (K[k] != null) el.textContent = K[k]; });

  // ---------------------------------------------------------------- Chart.js plumbing
  var charts = {};
  var builders = [];
  function register(fn) { builders.push(fn); fn(); }
  function make(id, cfg) {
    if (charts[id]) charts[id].destroy();
    var el = document.getElementById(id); if (!el || !window.Chart) return null;
    charts[id] = new Chart(el, cfg); return charts[id];
  }
  function baseOpts(extra) {
    var o = {
      responsive: true, maintainAspectRatio: false, animation: { duration: 350 },
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: css('--surface'), titleColor: css('--ink'), bodyColor: css('--text'), borderColor: css('--border'), borderWidth: 1,
          padding: 10, cornerRadius: 6, boxPadding: 4, titleFont: { family: 'Inter', weight: '600' }, bodyFont: { family: 'Inter' }
        }
      },
      scales: {}
    };
    return deepMerge(o, extra || {});
  }
  function axis(extra) {
    return deepMerge({ grid: { color: css('--grid'), drawTicks: false }, border: { display: false }, ticks: { color: css('--muted'), font: { family: 'Inter', size: 11 }, padding: 6 }, title: { color: css('--muted'), font: { family: 'Inter', size: 11 } } }, extra || {});
  }
  function deepMerge(a, b) {
    Object.keys(b).forEach(function (k) {
      if (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object') deepMerge(a[k], b[k]); else a[k] = b[k];
    });
    return a;
  }
  // vertical or horizontal reference line(s): options.plugins.refLines = [{axis:'x'|'y', value, color, dash}]
  var refLinePlugin = {
    id: 'refLines',
    afterDatasetsDraw: function (chart, args, opts) {
      if (!opts || !opts.lines) return;
      var ctx = chart.ctx, a = chart.chartArea;
      opts.lines.forEach(function (l) {
        var sc = chart.scales[l.axis === 'x' ? 'x' : 'y']; if (!sc) return;
        var p = sc.getPixelForValue(l.value);
        ctx.save(); ctx.strokeStyle = l.color || css('--muted'); ctx.lineWidth = l.width || 1.25; ctx.setLineDash(l.dash || [4, 4]);
        ctx.beginPath();
        if (l.axis === 'x') { ctx.moveTo(p, a.top); ctx.lineTo(p, a.bottom); } else { ctx.moveTo(a.left, p); ctx.lineTo(a.right, p); }
        ctx.stroke();
        if (l.label) { ctx.setLineDash([]); ctx.fillStyle = l.color || css('--muted'); ctx.font = '11px Inter'; ctx.textAlign = l.axis === 'x' ? 'left' : 'right';
          if (l.axis === 'x') ctx.fillText(l.label, p + 5, a.top + 12); else ctx.fillText(l.label, a.right - 4, p - 5); }
        ctx.restore();
      });
    }
  };
  // shaded x-ranges for category axes: options.plugins.bands = {ranges:[{from,to,color}]}
  var bandPlugin = {
    id: 'bands',
    beforeDatasetsDraw: function (chart, args, opts) {
      if (!opts || !opts.ranges) return;
      var ctx = chart.ctx, a = chart.chartArea, x = chart.scales.x;
      var labels = chart.data.labels, half = (x.getPixelForValue(1) - x.getPixelForValue(0)) / 2;
      opts.ranges.forEach(function (r) {
        var i0 = typeof r.from === 'string' ? labels.indexOf(r.from) : r.from, i1 = typeof r.to === 'string' ? labels.indexOf(r.to) : r.to;
        var x0 = Math.max(a.left, x.getPixelForValue(i0) - half);
        var x1 = Math.min(a.right, x.getPixelForValue(i1) + half);
        ctx.save(); ctx.fillStyle = r.color; ctx.fillRect(x0, a.top, Math.max(1, x1 - x0), a.bottom - a.top); ctx.restore();
      });
    }
  };
  if (window.Chart) {
    Chart.register(refLinePlugin, bandPlugin);
    Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
    Chart.defaults.elements.line.borderWidth = 2;
    Chart.defaults.elements.point.radius = 0;
    Chart.defaults.elements.point.hoverRadius = 5;
    Chart.defaults.elements.bar.borderRadius = 4;
  }
  function mix(cssVar, alpha) { var c = css(cssVar); return 'color-mix(in srgb, ' + c + ' ' + Math.round(alpha * 100) + '%, transparent)'; }
  function alphaColor(hex, a) {
    var h = hex.replace('#', ''); if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join('');
    var n = parseInt(h, 16); return 'rgba(' + (n >> 16 & 255) + ',' + (n >> 8 & 255) + ',' + (n & 255) + ',' + a + ')';
  }

  // ---------------------------------------------------------------- 02 data
  (function waterfall() {
    var rows = D.waterfall, max = rows[0].loans_remaining, html = '<div class="table-wrap" style="border:none"><table class="data fit"><tbody>';
    rows.forEach(function (r, i) {
      var w = (100 * r.loans_remaining / max).toFixed(1);
      html += '<tr' + (i === rows.length - 1 ? ' class="hl"' : '') + '><td class="wrap-cell">' + esc(r.step) + (r.removed ? ' <span class="tag neutral">−' + nf(r.removed) + '</span>' : '') +
        '</td><td class="dbar"><div class="in"><span class="b" style="width:' + (w * 0.6) + '%"></span><em>' + nf(r.loans_remaining) + '</em></div></td></tr>';
    });
    $('#waterfall').innerHTML = html + '</tbody></table></div>';
  })();

  (function quality() {
    var q = D.quality_notes, rows = [
      ['<code>mort_acc</code> missing: not recorded for 2007–2011 vintages, partly in 2012', pct(q.mort_acc_missing_pct / 100, 1), 'Median imputation fitted on training rows. It is missing by vintage, not at random, so its coefficient mostly reflects post-2012 loans'],
      ['Income above $1M', nf(q.annual_inc_over_1m) + ' loans', '<code>log1p</code> transform limits their leverage'],
      ['Income recorded as $0', nf(q.annual_inc_zero) + ' loans', 'Kept (log1p(0) = 0); too few to matter'],
      ['DTI ≥ 100 (999 used as a placeholder)', nf(q.dti_at_or_above_100) + ' loans', 'All in 2015; kept, noted as a limitation'],
      ['Revolving utilisation &gt; 100%', nf(q.revol_util_over_100) + ' loans', 'Plausible (over-limit cards); kept'],
      ['Rare <code>home_ownership</code> values (ANY / NONE / OTHER)', '214 loans', 'One-hot encoded; ANY and NONE (&lt; 50 rows) are dropped from the statsmodels fit']
    ];
    $('#quality-body').innerHTML = rows.map(function (r) { return '<tr><td class="wrap-cell">' + r[0] + '</td><td class="num">' + r[1] + '</td><td class="wrap-cell">' + r[2] + '</td></tr>'; }).join('');
  })();

  register(function censor() {
    var v = D.vintage_audit, labels = v.map(function (r) { return String(r.year); });
    var common = function (yTitle, max) {
      return baseOpts({ scales: { x: axis({ grid: { display: false } }), y: axis({ beginAtZero: true, max: max, title: { display: true, text: yTitle } }) },
        plugins: { tooltip: { callbacks: { label: function (c) { return c.parsed.y.toFixed(1) + '%'; } } } } });
    };
    var cens = labels.map(function (y) { return Number(y) >= 2016 ? alphaColor(css('--s-c'), 1) : alphaColor(css('--s-c'), 0.35); });
    make('ch-censor', { type: 'bar', data: { labels: labels, datasets: [{ data: v.map(function (r) { return r.pct_unresolved; }), backgroundColor: cens, borderRadius: 4, maxBarThickness: 34 }] }, options: common('% of loans unresolved', 100) });
    var drc = labels.map(function (y) { return Number(y) >= 2016 ? alphaColor(css('--s-b'), 0.35) : alphaColor(css('--s-b'), 1); });
    make('ch-censor-dr', { type: 'bar', data: { labels: labels, datasets: [{ data: v.map(function (r) { return r.resolved_only_default_rate_pct; }), backgroundColor: drc, borderRadius: 4, maxBarThickness: 34 }] }, options: common('% defaulted (resolved loans)', 30) });
  });

  var sigFeature = 'FICO (avg)';
  (function sigSeg() {
    var feats = []; D.borrower_signal.forEach(function (r) { if (feats.indexOf(r.feature) < 0) feats.push(r.feature); });
    $('#sig-seg').innerHTML = feats.map(function (f, i) { return '<button type="button" aria-pressed="' + (i === 0) + '" data-f="' + esc(f) + '">' + esc(f.replace(' (avg)', '').replace(' (%)', '')) + '</button>'; }).join('');
    $$('#sig-seg button').forEach(function (b) { b.addEventListener('click', function () { sigFeature = b.dataset.f; $$('#sig-seg button').forEach(function (x) { x.setAttribute('aria-pressed', x === b); }); drawSignal(); }); });
  })();
  function drawSignal() {
    var rows = D.borrower_signal.filter(function (r) { return r.feature === sigFeature; });
    make('ch-signal', { type: 'bar', data: { labels: rows.map(function (r) { return r.level; }), datasets: [{ data: rows.map(function (r) { return 100 * r.default_rate; }), backgroundColor: css('--s-b'), maxBarThickness: 46, loans: rows.map(function (r) { return r.loans; }) }] },
      options: baseOpts({ scales: { x: axis({ grid: { display: false } }), y: axis({ beginAtZero: true, title: { display: true, text: 'Default rate, %' } }) },
        plugins: { refLines: { lines: [{ axis: 'y', value: 100 * D.meta.train_dr, label: 'base ' + pct(D.meta.train_dr) }] },
          tooltip: { callbacks: { label: function (c) { return c.parsed.y.toFixed(1) + '% of ' + nf(c.dataset.loans[c.dataIndex]) + ' loans'; } } } } }) });
  }
  register(drawSignal);

  register(function monthly() {
    var m = D.monthly, labels = m.map(function (r) { return r.month; });
    var idx = function (mm) { return labels.indexOf(mm); };
    var bands = { ranges: [
      { from: labels[0], to: '2012-12', color: alphaColor(css('--s-b'), 0.08) },
      { from: '2013-01', to: '2014-12', color: alphaColor(css('--accent'), 0.1) },
      { from: '2015-01', to: labels[labels.length - 1], color: alphaColor(css('--s-c'), 0.13) }] };
    var xAxis = function (show) { return axis({ grid: { display: false }, ticks: { display: show, autoSkip: false, maxRotation: 0, callback: function (v, i) { var l = labels[i]; return l && l.slice(5) === '01' ? l.slice(0, 4) : ''; } } }); };
    make('ch-month-dr', { type: 'line', data: { labels: labels, datasets: [{ label: 'Default rate', data: m.map(function (r) { return 100 * r.default_rate; }), borderColor: col('ACTUAL'), loans: m.map(function (r) { return r.loans; }) }] },
      options: baseOpts({ interaction: { mode: 'index', intersect: false }, scales: { x: xAxis(true), y: axis({ beginAtZero: true, title: { display: true, text: 'Default rate, %' } }) }, plugins: { bands: bands,
        tooltip: { callbacks: { label: function (c) { return 'Default rate ' + c.parsed.y.toFixed(1) + '% (' + nf(c.dataset.loans[c.dataIndex]) + ' loans)'; } } } } }) });
    [['ch-month-u', 'UNRATE', '%'], ['ch-month-du', 'UNRATE_change_6m', ' pp'], ['ch-month-cpi', 'CPI_YoY_Inflation', '%']].forEach(function (s) {
      make(s[0], { type: 'line', data: { labels: labels, datasets: [{ data: m.map(function (r) { return r[s[1]]; }), borderColor: css('--s-b'), borderWidth: 1.6 }] },
        options: baseOpts({ interaction: { mode: 'index', intersect: false }, scales: { x: xAxis(false), y: axis({ ticks: { maxTicksLimit: 3 } }) }, plugins: { bands: bands,
          tooltip: { callbacks: { label: function (c) { return c.parsed.y.toFixed(2) + s[2]; } } } } }) });
    });
    void idx;
  });

  // ---------------------------------------------------------------- 04 tuning
  var tuneFam = 'lr';
  (function tuneSeg() {
    $('#tune-seg').innerHTML = '<button type="button" data-t="lr" aria-pressed="true">Logistic regression</button><button type="button" data-t="gbm" aria-pressed="false">LightGBM</button><button type="button" data-t="pre" aria-pressed="false">Income transform</button>';
    $$('#tune-seg button').forEach(function (b) { b.addEventListener('click', function () { tuneFam = b.dataset.t; $$('#tune-seg button').forEach(function (x) { x.setAttribute('aria-pressed', x === b); }); drawTune(); }); });
  })();
  function drawTune() {
    var rows;
    if (tuneFam === 'pre') {
      var best = {}; D.preproc_check.forEach(function (r) { if (!best[r.Model] || r['PR-AUC'] > best[r.Model]) best[r.Model] = r['PR-AUC']; });
      rows = D.preproc_check.map(function (r) { return '<tr' + (r['PR-AUC'] === best[r.Model] ? ' class="hl"' : '') + '><td>' + SHORT[r.Model] + '</td><td class="mono">' + esc(r['Income transform']) + '</td><td class="num">' + fx(r['PR-AUC'], 5) + '</td><td class="num">' + fx(r['ROC-AUC'], 5) + '</td><td class="num">' + fx(r.Brier_Score, 5) + '</td><td>' + (r['PR-AUC'] === best[r.Model] ? '<span class="tag good">chosen</span>' : '') + '</td></tr>'; });
    } else {
      var keep = tuneFam === 'lr' ? ['MODEL_B', 'MODEL_C', 'LENDER_BENCHMARK', 'MODEL_C_ABLATION', 'MODEL_M'] : ['MODEL_D_BASE', 'MODEL_D'];
      rows = D.tuning.filter(function (r) { return keep.indexOf(r.Model) >= 0; }).map(function (r) {
        return '<tr' + (r.Selected ? ' class="hl"' : '') + '><td>' + SHORT[r.Model] + '</td><td class="mono">' + esc(r.Params) + '</td><td class="num">' + fx(r['PR-AUC'], 5) + '</td><td class="num">' + fx(r['ROC-AUC'], 5) + '</td><td class="num">' + fx(r.Brier_Score, 5) + '</td><td>' + (r.Selected ? '<span class="tag good">selected</span>' : '') + '</td></tr>';
      });
    }
    $('#tune-body').innerHTML = rows.join('');
  }
  drawTune();

  // ---------------------------------------------------------------- 06 results
  (function holdoutTable() {
    $('#holdout-body').innerHTML = MAIN.map(function (k) {
      var r = H[k], cv = CVM[k];
      return '<tr' + (k === 'MODEL_C' ? ' class="hl"' : '') + '><td><span class="swatch" style="background:var(' + CVAR[k] + ')"></span>' + NAMES[k] + '</td><td class="num">' + fx(r['PR-AUC']) + '</td><td class="num">' + fx(r['ROC-AUC']) + '</td><td class="num">' + fx(r.Brier_Score) + '</td><td class="num">' + pct(r.Mean_Pred_PD, 2) + '</td><td class="num">' + (cv ? fx(cv['PR-AUC']) : '–') + '</td></tr>';
    }).join('') + '<tr class="muted"><td>Actual 2015 default rate</td><td class="num"></td><td class="num"></td><td class="num"></td><td class="num">' + pct(D.meta.test_dr, 2) + '</td><td class="num"></td></tr>';
  })();

  var metric = 'PR-AUC';
  $$('#metric-seg button').forEach(function (b) { b.addEventListener('click', function () { metric = b.dataset.m; $$('#metric-seg button').forEach(function (x) { x.setAttribute('aria-pressed', x === b); }); drawMetric(); }); });
  function drawMetric() {
    var isPD = metric === 'Mean_Pred_PD', mult = isPD ? 100 : 1;
    var vals = MAIN.map(function (k) { return H[k][metric] * mult; });
    var lines = [];
    if (metric === 'PR-AUC') lines.push({ axis: 'x', value: H.MODEL_A['PR-AUC'], label: 'no skill' });
    if (metric === 'ROC-AUC') lines.push({ axis: 'x', value: 0.5, label: 'no skill' });
    if (isPD) lines.push({ axis: 'x', value: 100 * D.meta.test_dr, label: 'actual ' + pct(D.meta.test_dr), color: css('--ink') });
    make('ch-metric', { type: 'bar', data: { labels: MAIN.map(function (k) { return SHORT[k]; }), datasets: [{ data: vals, backgroundColor: MAIN.map(col), maxBarThickness: 26 }] },
      options: baseOpts({ indexAxis: 'y', scales: { y: axis({ grid: { display: false } }), x: axis({ beginAtZero: true, title: { display: true, text: isPD ? 'Mean predicted PD, %' : metric.replace('_', ' ') + (metric === 'Brier_Score' ? ' (lower is better)' : '') } }) },
        plugins: { refLines: { lines: lines }, tooltip: { callbacks: { title: function (c) { return NAMES[MAIN[c[0].dataIndex]]; }, label: function (c) { return isPD ? c.parsed.x.toFixed(2) + '%' : c.parsed.x.toFixed(4); } } } } }) });
    var d = H.MODEL_C[metric] - H.MODEL_B[metric];
    var notes = {
      'PR-AUC': 'C − B = ' + sgn(d) + '. LightGBM adds ' + sgn(H.MODEL_D_BASE['PR-AUC'] - H.MODEL_B['PR-AUC']) + ' over B; the lender\'s pricing adds ' + sgn(H.LENDER_BENCHMARK['PR-AUC'] - H.MODEL_B['PR-AUC']) + '.',
      'ROC-AUC': 'C − B = ' + sgn(d) + '. All borrower models sit between 0.63 and 0.65; the lender benchmark reaches ' + fx(H.LENDER_BENCHMARK['ROC-AUC'], 3) + '.',
      'Brier_Score': 'C − B = ' + sgn(d, 5) + ' (positive = C worse). The baseline scores ' + fx(H.MODEL_A.Brier_Score) + '.',
      'Mean_Pred_PD': MAIN.filter(function (k) { return H[k].Mean_Pred_PD < D.meta.test_dr; }).length + ' of ' + MAIN.length + ' models under-predict 2015, the riskiest vintage in the data. Model C under-predicts more than B (' + pct(H.MODEL_C.Mean_Pred_PD, 2) + ' vs ' + pct(H.MODEL_B.Mean_Pred_PD, 2) + ' against ' + pct(D.meta.test_dr, 2) + ').'
    };
    $('#metric-note').textContent = notes[metric];
  }
  register(drawMetric);

  (function ciPlot() {
    var labels = ['PR-AUC', 'ROC-AUC', 'Brier score'];
    var maxAbs = 0; BOOT.forEach(function (r) { maxAbs = Math.max(maxAbs, Math.abs(r['CI_2.5%']), Math.abs(r['CI_97.5%']), Math.abs(r.Point_Estimate)); });
    var lim = maxAbs * 1.25, pos = function (x) { return (50 + 50 * x / lim).toFixed(2) + '%'; };
    var html = '';
    BOOT.forEach(function (r, i) {
      var lo = r['CI_2.5%'], hi = r['CI_97.5%'], pt = r.Point_Estimate, isBrier = i === 2;
      var verdict = (lo <= 0 && hi >= 0) ? '<span class="tag neutral">no detectable difference</span>' : (((isBrier ? hi < 0 : lo > 0)) ? '<span class="tag good">C better</span>' : '<span class="tag warn">B better</span>');
      html += '<div class="ci-row"><div><strong>Δ ' + labels[i] + '</strong>' + (isBrier ? ' <span class="chart-note">(lower is better)</span>' : '') + '<br>' + verdict + '</div>' +
        '<div class="ci-track" title="95% interval"><div class="zero" style="left:50%"></div><div class="iv" style="left:' + pos(lo) + ';width:calc(' + pos(hi) + ' - ' + pos(lo) + ')"></div><div class="pt" style="left:' + pos(pt) + '"></div></div>' +
        '<div class="ci-val">' + sgn(pt, 5) + '<br><span style="color:var(--muted)">[' + sgn(lo, 5) + ', ' + sgn(hi, 5) + ']</span></div></div>';
    });
    $('#ci-plot').innerHTML = html;
    $('#ci-note').textContent = 'All three differences are tiny next to the gap between B and the no-skill baseline (' + sgn(H.MODEL_B['PR-AUC'] - H.MODEL_A['PR-AUC'], 3) + ' PR-AUC). Where an interval excludes zero it favours the borrower-only model.';
  })();

  var curveKind = 'pr';
  $$('#curve-seg button').forEach(function (b) { b.addEventListener('click', function () { curveKind = b.dataset.c; $$('#curve-seg button').forEach(function (x) { x.setAttribute('aria-pressed', x === b); }); drawCurves(); }); });
  function drawCurves() {
    var keys = curveKind === 'cal' ? ['MODEL_B', 'MODEL_C', 'MODEL_D', 'MODEL_C_BALANCED_CHECK'] : ['MODEL_B', 'MODEL_C', 'MODEL_D', 'LENDER_BENCHMARK'];
    var ds = keys.map(function (k) {
      var pts = curveKind === 'pr' ? D.curves[k].map(function (p) { return { x: p.recall, y: p.precision }; })
        : curveKind === 'roc' ? D.curves[k].map(function (p) { return { x: p.fpr, y: p.recall }; })
          : D.calibration[k].map(function (p) { return { x: p.pred, y: p.obs }; });
      var c = k === 'MODEL_C_BALANCED_CHECK' ? css('--muted') : col(k);
      return { label: NAMES[k], data: pts, borderColor: c, backgroundColor: c, showLine: true, borderDash: k === 'MODEL_C_BALANCED_CHECK' ? [5, 4] : [], pointRadius: curveKind === 'cal' ? 4 : 0, borderWidth: 2 };
    });
    var ref = curveKind === 'pr' ? [{ x: 0, y: D.meta.test_dr }, { x: 1, y: D.meta.test_dr }] : curveKind === 'roc' ? [{ x: 0, y: 0 }, { x: 1, y: 1 }] : [{ x: 0, y: 0 }, { x: 0.7, y: 0.7 }];
    ds.push({ label: curveKind === 'cal' ? 'Perfect calibration' : 'No skill', data: ref, borderColor: css('--muted'), borderDash: [3, 4], borderWidth: 1.2, showLine: true, pointRadius: 0 });
    var t = { pr: ['Recall', 'Precision'], roc: ['False positive rate', 'True positive rate'], cal: ['Mean predicted PD (decile)', 'Observed default rate'] }[curveKind];
    make('ch-curves', { type: 'scatter', data: { datasets: ds }, options: baseOpts({ interaction: { mode: 'nearest', intersect: false },
      scales: { x: axis({ min: 0, max: curveKind === 'cal' ? 0.7 : 1, title: { display: true, text: t[0] } }), y: axis({ min: 0, max: curveKind === 'pr' ? 0.8 : curveKind === 'cal' ? 0.5 : 1, title: { display: true, text: t[1] } }) },
      plugins: { tooltip: { callbacks: { label: function (c) { return c.dataset.label + ': (' + c.parsed.x.toFixed(3) + ', ' + c.parsed.y.toFixed(3) + ')'; } } } } }) });
    $('#curve-legend').innerHTML = ds.map(function (d) { return '<span><i class="' + (d.borderDash && d.borderDash.length ? 'dash' : '') + '" style="background:' + d.borderColor + ';color:' + d.borderColor + '"></i>' + esc(d.label) + '</span>'; }).join('');
    $('#curve-note').textContent = curveKind === 'cal'
      ? 'Points near the diagonal are well calibrated. Unweighted models track it; balanced weights shift every decile far to the right.'
      : 'B and C lie on top of each other. The lender benchmark sits clearly above both.';
  }
  register(drawCurves);

  // threshold explorer
  var thrModel = 'MODEL_C';
  var thrIdx = 0;
  (function thrInit() {
    var keys = ['MODEL_B', 'MODEL_C', 'LENDER_BENCHMARK', 'MODEL_D'];
    $('#thr-model').innerHTML = keys.map(function (k) { return '<button type="button" data-k2="' + k + '" aria-pressed="' + (k === thrModel) + '">' + SHORT[k] + '</button>'; }).join('');
    $$('#thr-model button').forEach(function (b) { b.addEventListener('click', function () { thrModel = b.dataset.k2; $$('#thr-model button').forEach(function (x) { x.setAttribute('aria-pressed', x === b); }); drawThr(); updateThr(); }); });
    var grid = D.thresholds[thrModel], best = 0, bd = 1e9;
    grid.forEach(function (r, i) { var d = Math.abs(r.t - D.meta.tau); if (d < bd) { bd = d; best = i; } });
    thrIdx = best;
    var s = $('#thr'); s.max = grid.length - 1; s.value = best;
    s.addEventListener('input', function () { thrIdx = Number(s.value); updateThr(); });
  })();
  function thrRows() { return D.thresholds[thrModel].map(function (r) { var f = r.tp + r.fp, n = f + r.fn + r.tn; return { t: r.t, prec: f ? r.tp / f : 0, rec: r.tp / (r.tp + r.fn), flagged: f / n, r: r }; }); }
  function drawThr() {
    var rows = thrRows();
    make('ch-thr', { type: 'line', data: { labels: rows.map(function (r) { return (100 * r.t).toFixed(1); }), datasets: [
      { label: 'Precision', data: rows.map(function (r) { return 100 * r.prec; }), borderColor: css('--s-b') },
      { label: 'Recall', data: rows.map(function (r) { return 100 * r.rec; }), borderColor: css('--s-c') },
      { label: 'Flagged', data: rows.map(function (r) { return 100 * r.flagged; }), borderColor: css('--muted'), borderDash: [5, 4] }] },
      options: baseOpts({ interaction: { mode: 'index', intersect: false }, scales: { x: axis({ grid: { display: false }, title: { display: true, text: 'Threshold, % PD' }, ticks: { maxTicksLimit: 10 } }), y: axis({ min: 0, max: 100, title: { display: true, text: '%' } }) },
        plugins: { refLines: { lines: [{ axis: 'x', value: thrIdx, color: css('--accent'), dash: [], width: 2 }] }, tooltip: { callbacks: { title: function (c) { return 'Threshold ' + c[0].label + '%'; }, label: function (c) { return c.dataset.label + ': ' + c.parsed.y.toFixed(1) + '%'; } } } } }) });
  }
  function updateThr() {
    var rows = thrRows(); if (thrIdx >= rows.length) thrIdx = rows.length - 1;
    var r = rows[thrIdx], c = r.r, s = $('#thr');
    s.value = thrIdx; s.style.setProperty('--fill', (100 * thrIdx / (rows.length - 1)) + '%');
    $('#thr-out').textContent = (100 * r.t).toFixed(1) + '%';
    $('#cm').innerHTML = '<div></div><div class="h">Flagged (PD ≥ t)</div><div class="h">Approved (PD &lt; t)</div>' +
      '<div class="r">Defaulted</div><div class="cell tp"><b>' + nf(c.tp) + '</b><small>caught defaults</small></div><div class="cell fn"><b>' + nf(c.fn) + '</b><small>missed defaults</small></div>' +
      '<div class="r">Repaid</div><div class="cell fp"><b>' + nf(c.fp) + '</b><small>false alarms</small></div><div class="cell tn"><b>' + nf(c.tn) + '</b><small>correct approvals</small></div>';
    var f1 = 2 * c.tp / (2 * c.tp + c.fp + c.fn);
    $('#thr-stats').innerHTML = [['Precision', pct(r.prec)], ['Recall', pct(r.rec)], ['Loans flagged', pct(r.flagged)], ['F1', f1.toFixed(3)]]
      .map(function (x) { return '<div class="stat"><div class="v">' + x[1] + '</div><div class="l">' + x[0] + '</div></div>'; }).join('');
    var ch = charts['ch-thr']; if (ch) { ch.options.plugins.refLines.lines[0].value = thrIdx; ch.update('none'); }
  }
  register(function () { drawThr(); updateThr(); });

  register(function oddsRatios() {
    var terms = D.logit.filter(function (r) { return r.term !== 'const'; }).sort(function (a, b) { return a.odds_ratio - b.odds_ratio; });
    var pretty = function (t) { var n = t.split('__')[1] || t; return ({ annual_inc: 'Income (log)', dti: 'Debt-to-income', fico_avg: 'FICO', open_acc: 'Open lines', pub_rec: 'Public records', revol_util: 'Revolving util.', mort_acc: 'Mortgage accts', loan_amnt: 'Loan amount', UNRATE: 'Unemployment', UNRATE_change_6m: 'Δ unemployment 6m', CPI_YoY_Inflation: 'Inflation', home_ownership_MORTGAGE: 'Home: mortgage', home_ownership_OWN: 'Home: own', home_ownership_OTHER: 'Home: other' })[n] || n; };
    var isMacro = function (t) { return /UNRATE|CPI/.test(t); };
    var labels = terms.map(function (r) { return pretty(r.term); });
    make('ch-or', { type: 'bar', data: { labels: labels, datasets: [
      { type: 'bar', label: '95% CI', data: terms.map(function (r) { return [r.or_ci_low, r.or_ci_high]; }), backgroundColor: terms.map(function (r) { return alphaColor(isMacro(r.term) ? css('--s-c') : css('--s-b'), 0.3); }), borderRadius: 3, barThickness: 8, order: 2 },
      { type: 'scatter', label: 'Odds ratio', data: terms.map(function (r, i) { return { x: r.odds_ratio, y: labels[i] }; }), backgroundColor: terms.map(function (r) { return isMacro(r.term) ? css('--s-c') : css('--s-b'); }), pointRadius: 5, pointHoverRadius: 7, borderColor: css('--surface'), borderWidth: 2, order: 1 }] },
      options: baseOpts({ indexAxis: 'y', interaction: { mode: 'nearest', axis: 'y', intersect: false }, scales: { y: axis({ type: 'category', grid: { display: false } }), x: axis({ title: { display: true, text: 'Odds ratio per 1 SD (95% CI)' } }) },
        plugins: { refLines: { lines: [{ axis: 'x', value: 1, label: 'no effect' }] }, tooltip: { callbacks: { label: function (c) { var r = terms[c.dataIndex]; return 'OR ' + r.odds_ratio.toFixed(3) + '  [' + r.or_ci_low.toFixed(3) + ', ' + r.or_ci_high.toFixed(3) + ']  p=' + (r.p_value < 0.001 ? '<0.001' : r.p_value.toFixed(3)); } } } } }) });
  });

  // ---------------------------------------------------------------- 07 errors
  register(function vintage() {
    var m = D.monthly.filter(function (r) { return r.pred_B != null; });
    var labels = m.map(function (r) { return r.month; });
    make('ch-vintage', { type: 'line', data: { labels: labels, datasets: [
      { label: 'Actual', data: m.map(function (r) { return 100 * r.default_rate; }), borderColor: col('ACTUAL'), borderWidth: 2.4 },
      { label: 'Model B', data: m.map(function (r) { return 100 * r.pred_B; }), borderColor: col('MODEL_B') },
      { label: 'Model C', data: m.map(function (r) { return 100 * r.pred_C; }), borderColor: col('MODEL_C') }] },
      options: baseOpts({ interaction: { mode: 'index', intersect: false }, scales: { x: axis({ grid: { display: false }, ticks: { autoSkip: false, maxRotation: 0, callback: function (v, i) { var l = labels[i]; return l.slice(5) === '01' ? l.slice(0, 4) : (l.slice(5) === '07' ? 'Jul' : ''); } } }), y: axis({ title: { display: true, text: 'Default rate, %' } }) },
        plugins: { refLines: { lines: [{ axis: 'x', value: labels.indexOf('2014-01') }, { axis: 'x', value: labels.indexOf('2015-01'), label: '2015 test' }] }, tooltip: { callbacks: { label: function (c) { return c.dataset.label + ': ' + c.parsed.y.toFixed(2) + '%'; } } } } }) });
  });
  (function vintageTable() {
    var away = 0;
    $('#vintage-body').innerHTML = D.vintage_yearly.map(function (r) {
      var shift = r.pred_C - r.pred_B, needed = r.actual - r.pred_B, wrong = shift * needed < 0; if (wrong) away++;
      return '<tr' + (r.vintage === 2015 ? ' class="hl"' : '') + '><td>' + r.vintage + (r.vintage === 2015 ? ' (test)' : ' (CV)') + '</td><td class="num">' + nf(r.loans) + '</td><td class="num"><strong>' + pct(r.actual, 2) + '</strong></td><td class="num">' + pct(r.pred_A, 2) + '</td><td class="num">' + pct(r.pred_B, 2) + '</td><td class="num">' + pct(r.pred_C, 2) + '</td><td class="num">' + sgn(100 * shift, 2) + ' pp ' + (wrong ? '<span class="tag warn">wrong way</span>' : '<span class="tag good">right way</span>') + '</td></tr>';
    }).join('');
    $('#vintage-finding').textContent = 'In ' + away + ' of ' + D.vintage_yearly.length + ' out-of-time years the macro block moves the predicted default rate away from the realised rate. Month by month, the macro shift (C − B) is ' +
      (D.meta.macro_shift_corr < 0 ? 'negatively' : 'positively') + ' correlated with Model B\'s miss (r = ' + sgn(D.meta.macro_shift_corr, 2) + '). Falling unemployment after 2010 told the model that risk was falling, but LendingClub\'s later vintages defaulted more, most likely because of looser underwriting, which national macro data do not capture.';
  })();

  var segName = 'FICO (avg)';
  (function segInit() {
    var names = []; D.segments.forEach(function (r) { if (names.indexOf(r.segment) < 0) names.push(r.segment); });
    $('#segment-seg').innerHTML = names.map(function (n, i) { return '<button type="button" data-s="' + esc(n) + '" aria-pressed="' + (i === 0) + '">' + esc(n.replace(' (avg)', '').replace(' (%)', '')) + '</button>'; }).join('');
    $$('#segment-seg button').forEach(function (b) { b.addEventListener('click', function () { segName = b.dataset.s; $$('#segment-seg button').forEach(function (x) { x.setAttribute('aria-pressed', x === b); }); drawSeg(); }); });
  })();
  function drawSeg() {
    var rows = D.segments.filter(function (r) { return r.segment === segName; });
    make('ch-seg', { type: 'bar', data: { labels: rows.map(function (r) { return r.level; }), datasets: [
      { label: 'Actual', data: rows.map(function (r) { return 100 * r.actual; }), backgroundColor: col('ACTUAL'), maxBarThickness: 22 },
      { label: 'Model B', data: rows.map(function (r) { return 100 * r.pred_B; }), backgroundColor: col('MODEL_B'), maxBarThickness: 22 },
      { label: 'Model C', data: rows.map(function (r) { return 100 * r.pred_C; }), backgroundColor: col('MODEL_C'), maxBarThickness: 22 }] },
      options: baseOpts({ interaction: { mode: 'index', intersect: false }, datasets: { bar: { categoryPercentage: 0.7, barPercentage: 0.92 } }, scales: { x: axis({ grid: { display: false } }), y: axis({ beginAtZero: true, title: { display: true, text: 'Default rate, % (2015)' } }) },
        plugins: { tooltip: { callbacks: { afterTitle: function (c) { var r = rows[c[0].dataIndex]; return nf(r.loans) + ' loans · within-segment ROC-AUC (C) ' + r.roc_auc_C_within.toFixed(3); }, label: function (c) { return c.dataset.label + ': ' + c.parsed.y.toFixed(1) + '%'; } } } } }) });
    var worst = rows.slice().sort(function (a, b) { return a.gap_C_pp - b.gap_C_pp; })[0];
    $('#seg-note').textContent = 'Largest under-prediction in this view: ' + worst.level + ' (' + sgn(worst.gap_C_pp, 1) + ' pp). B and C make almost identical errors. Within a segment, ranking is weaker (ROC-AUC ' +
      Math.min.apply(null, rows.map(function (r) { return r.roc_auc_C_within; })).toFixed(2) + '–' + Math.max.apply(null, rows.map(function (r) { return r.roc_auc_C_within; })).toFixed(2) + ').';
  }
  register(drawSeg);

  (function errTables() {
    var order = ['TP: caught default', 'FN: missed default', 'FP: false alarm', 'TN: correct approval'];
    var g = by(D.error_groups, 'outcome');
    $('#err-groups').innerHTML = order.filter(function (o) { return g[o]; }).map(function (o) {
      var r = g[o], cls = o[0] === 'F' ? (o[1] === 'N' ? 'bad' : 'warn') : 'good';
      return '<tr><td><span class="tag ' + cls + '">' + o.split(':')[0] + '</span> ' + esc(o.split(': ')[1]) + '</td><td class="num">' + nf(r.loans) + '</td><td class="num">' + pct(r.mean_pd_C) + '</td><td class="num">' + nf(r.fico) + '</td><td class="num">' + fx(r.dti, 1) + '</td><td class="num">$' + nf(r.median_income) + '</td><td class="num">' + fx(r.int_rate_not_used, 1) + '%</td></tr>';
    }).join('');
    var fn = g['FN: missed default'], tn = g['TN: correct approval'], tp = g['TP: caught default'];
    if (fn && tn) {
      $('#err-note').innerHTML = 'Missed defaults look almost exactly like correct approvals on what the model sees (FICO ' + nf(fn.fico) + ' vs ' + nf(tn.fico) + ', DTI ' + fx(fn.dti, 1) + ' vs ' + fx(tn.dti, 1) +
        '), yet LendingClub charged them more interest (' + fx(fn.int_rate_not_used, 1) + '% vs ' + fx(tn.int_rate_not_used, 1) + '%). The lender had information our features lack, such as credit-history detail, recent inquiries and employment. That is the same gap the lender benchmark shows. ' +
        (tp ? 'Caught defaults are the visibly weak files (FICO ' + nf(tp.fico) + ', DTI ' + fx(tp.dti, 1) + ').' : '');
    }
    $('#err-examples').innerHTML = D.error_examples.map(function (r) {
      var isFN = r.outcome.indexOf('FN') === 0;
      return '<tr><td><span class="tag ' + (isFN ? 'bad' : 'warn') + '">' + (isFN ? 'missed default' : 'false alarm') + '</span></td><td>' + esc(r.issue_d) + '</td><td class="num">' + pct(r.pd_C) + '</td><td class="num">' + nf(r.fico_avg) + '</td><td class="num">' + fx(r.dti, 1) + '</td><td class="num">$' + nf(r.annual_inc) + '</td><td class="num">$' + nf(r.loan_amnt) + '</td><td>' + esc(r.home_ownership) + '</td><td class="num">' + (r.revol_util == null ? '–' : fx(r.revol_util, 1) + '%') + '</td><td class="num">' + fx(r.int_rate, 2) + '%</td></tr>';
    }).join('');
  })();

  (function extraTable() {
    var ex = {}; D.extra.forEach(function (r) { ex[(/Embargo/.test(r.Experiment) ? 'emb_' : 'main_') + r.Model] = r; });
    var eb = ex.emb_MODEL_B, ec = ex.emb_MODEL_C;
    if (eb && ec) {
      $('#extra-emb').innerHTML = '<strong>Embargoed training.</strong> On 1 Jan 2015 only loans issued up to Dec 2011 had a fully observed outcome (' + nf(eb.Train_loans) + ' loans). The main split also uses 2012–2014 loans whose outcomes were realised during 2015–2017, so it is optimistic. Trained the realistic way, PR-AUC drops to ' +
        fx(eb['PR-AUC']) + ' (B) and ' + fx(ec['PR-AUC']) + ' (C). The macro model is now clearly <em>worse</em> calibrated: it predicts a ' + pct(ec.Mean_Pred_PD) + ' default rate against ' + pct(D.meta.test_dr) + ' actual (B: ' + pct(eb.Mean_Pred_PD) + '), because the macro coefficients were learned mostly from recession-era loans. <code>mort_acc</code> is not recorded before 2012, so these models do without it.';
    }
    $('#extra-body').innerHTML = D.extra.map(function (r) {
      return '<tr' + (/Embargo/.test(r.Experiment) ? ' class="hl"' : '') + '><td class="wrap-cell">' + esc(r.Experiment) + '</td><td>' + (SHORT[r.Model] || r.Model) + '</td><td class="num">' + nf(r.Train_loans) + '</td><td class="num">' + fx(r['PR-AUC']) + '</td><td class="num">' + fx(r['ROC-AUC']) + '</td><td class="num">' + fx(r.Brier_Score) + '</td><td class="num">' + pct(r.Mean_Pred_PD, 1) + '</td></tr>';
    }).join('');
  })();

  // ---------------------------------------------------------------- 09 explorer
  var EX = D.explorer, MB = EX.models.MODEL_B, MC = EX.models.MODEL_C, R = EX.ranges;
  var state = {};
  var STEP = { fico_avg: 1, dti: 0.5, annual_inc: 1000, loan_amnt: 500, revol_util: 1, open_acc: 1, mort_acc: 1, pub_rec: 1, UNRATE: 0.1, UNRATE_change_6m: 0.1, CPI_YoY_Inflation: 0.1 };
  var FMT = {
    fico_avg: function (v) { return nf(v); }, dti: function (v) { return fx(v, 1) + '%'; }, annual_inc: function (v) { return '$' + nf(v); }, loan_amnt: function (v) { return '$' + nf(v); },
    revol_util: function (v) { return nf(v) + '%'; }, open_acc: function (v) { return nf(v); }, mort_acc: function (v) { return nf(v); }, pub_rec: function (v) { return nf(v); },
    UNRATE: function (v) { return fx(v, 1) + '%'; }, UNRATE_change_6m: function (v) { return sgn(v, 1) + ' pp'; }, CPI_YoY_Inflation: function (v) { return fx(v, 1) + '%'; }
  };
  var roundTo = function (v, s) { return Math.round(v / s) * s; };
  function slider(key, host) {
    var lo = R[key][0], hi = R[key][1], st = STEP[key];
    lo = Math.floor(lo / st) * st; hi = Math.ceil(hi / st) * st;
    if (key === 'annual_inc') { lo = Math.max(10000, lo); }
    var div = document.createElement('div'); div.className = 'range-row';
    div.innerHTML = '<div class="top"><label for="in-' + key + '">' + FEAT[key] + '</label><output id="out-' + key + '"></output></div><input type="range" id="in-' + key + '" min="' + lo + '" max="' + hi + '" step="' + st + '">' +
      '<div class="hint">training range ' + FMT[key](lo) + ' to ' + FMT[key](hi) + '</div>';
    host.appendChild(div);
    var inp = $('#in-' + key, div);
    inp.addEventListener('input', function () { state[key] = Number(inp.value); clearPresets(); computeExplorer(); });
    return inp;
  }
  var inputs = {};
  ['UNRATE', 'UNRATE_change_6m', 'CPI_YoY_Inflation'].forEach(function (k) { inputs[k] = slider(k, $('#macro-inputs')); });
  ['fico_avg', 'dti', 'annual_inc', 'loan_amnt', 'revol_util', 'open_acc', 'mort_acc', 'pub_rec'].forEach(function (k) { inputs[k] = slider(k, $('#borrower-inputs')); });
  Object.keys(EX.defaults).forEach(function (k) { state[k] = roundTo(EX.defaults[k], STEP[k]); });
  var monthBy = by(D.monthly, 'month');
  var PRESETS = { calm: '2015-06', peak: '2010-01', onset: '2009-03' };
  function setMacroFromMonth(mm) { var r = monthBy[mm]; if (!r) return; ['UNRATE', 'UNRATE_change_6m', 'CPI_YoY_Inflation'].forEach(function (k) { state[k] = Math.min(R[k][1], Math.max(R[k][0], roundTo(r[k], 0.1))); }); }
  setMacroFromMonth(PRESETS.calm);
  function clearPresets() { $$('#presets .chip').forEach(function (c) { c.setAttribute('aria-pressed', 'false'); }); }
  $$('#presets .chip').forEach(function (c) {
    c.addEventListener('click', function () { setMacroFromMonth(PRESETS[c.dataset.p]); clearPresets(); c.setAttribute('aria-pressed', 'true'); syncInputs(); computeExplorer(); });
  });
  $('#presets .chip[data-p="calm"]').setAttribute('aria-pressed', 'true');
  $('#in-home').addEventListener('change', computeExplorer);
  $('#presets .chip[data-p="calm"]').title = 'Macro values seen by a loan issued Jun 2015';
  $('#presets .chip[data-p="peak"]').title = 'Macro values seen by a loan issued Jan 2010';
  $('#presets .chip[data-p="onset"]').title = 'Macro values seen by a loan issued Mar 2009';
  function syncInputs() { Object.keys(inputs).forEach(function (k) { inputs[k].value = state[k]; }); }
  syncInputs();

  function logit(model, row) {
    var z = model.intercept, parts = [];
    Object.keys(model.numeric).forEach(function (c) {
      var s = model.numeric[c], x = s.transform === 'log1p' ? Math.log1p(row[c]) : row[c], v = s.coef * (x - s.mean) / s.scale;
      z += v; parts.push({ key: c, v: v });
    });
    Object.keys(model.categorical).forEach(function (c) {
      var m = model.categorical[c], v = m[row[c]] || 0; z += v; parts.push({ key: 'home', v: v - (m.RENT || 0) });
    });
    return { p: 1 / (1 + Math.exp(-z)), parts: parts };
  }
  function computeExplorer() {
    Object.keys(inputs).forEach(function (k) {
      var inp = inputs[k]; $('#out-' + k).textContent = FMT[k](state[k]);
      inp.style.setProperty('--fill', (100 * (state[k] - inp.min) / (inp.max - inp.min)) + '%');
    });
    var row = Object.assign({}, state, { home_ownership: $('#in-home').value });
    var b = logit(MB, row), c = logit(MC, row);
    $('#pd-b').textContent = pct(b.p); $('#pd-c').textContent = pct(c.p);
    $('#bar-b').style.width = Math.min(100, 100 * b.p / 0.5) + '%'; $('#bar-c').style.width = Math.min(100, 100 * c.p / 0.5) + '%';
    var macroPart = c.parts.filter(function (x) { return /UNRATE|CPI/.test(x.key); }).reduce(function (s, x) { return s + x.v; }, 0);
    var ficoPart = c.parts.filter(function (x) { return x.key === 'fico_avg'; })[0].v;
    $('#pd-delta').innerHTML = 'Model C differs from Model B by <strong>' + sgn(100 * (c.p - b.p), 1) + ' pp</strong>. Inside Model C the three macro inputs together contribute <strong>' + sgn(macroPart, 2) +
      '</strong> to the log-odds, versus <strong>' + sgn(ficoPart, 2) + '</strong> from FICO alone.';
    $('#pd-context').textContent = 'For scale: the 2015 default rate was ' + pct(D.meta.test_dr) + ' and the training rate ' + pct(D.meta.train_dr) + '. PD is the chance this loan is charged off over its 36-month life.';
    drawContrib(c.parts);
  }
  function drawContrib(parts) {
    var labels = { fico_avg: 'FICO', dti: 'Debt-to-income', annual_inc: 'Income (log)', loan_amnt: 'Loan amount', revol_util: 'Revolving util.', open_acc: 'Open lines', mort_acc: 'Mortgage accts', pub_rec: 'Public records', UNRATE: 'Unemployment', UNRATE_change_6m: 'Δ unemployment 6m', CPI_YoY_Inflation: 'Inflation', home: 'Home (vs rent)' };
    var p = parts.slice().sort(function (a, b) { return Math.abs(b.v) - Math.abs(a.v); });
    var data = p.map(function (x) { return x.v; }), names = p.map(function (x) { return labels[x.key] || x.key; });
    var colors = p.map(function (x) { return /UNRATE|CPI/.test(x.key) ? css('--s-c') : css('--s-b'); });
    var ch = charts['ch-contrib'];
    if (ch) { ch.data.labels = names; ch.data.datasets[0].data = data; ch.data.datasets[0].backgroundColor = colors; ch.update(); return; }
    make('ch-contrib', { type: 'bar', data: { labels: names, datasets: [{ data: data, backgroundColor: colors, maxBarThickness: 18 }] },
      options: baseOpts({ indexAxis: 'y', animation: { duration: 200 }, scales: { y: axis({ grid: { display: false } }), x: axis({ suggestedMin: -0.6, suggestedMax: 0.6, title: { display: true, text: 'Change in log-odds of default' } }) },
        plugins: { refLines: { lines: [{ axis: 'x', value: 0, dash: [] }] }, tooltip: { callbacks: { label: function (c) { return sgn(c.parsed.x, 3) + ' log-odds (odds × ' + Math.exp(c.parsed.x).toFixed(2) + ')'; } } } } }) });
  }
  register(function () { if (charts['ch-contrib']) { charts['ch-contrib'].destroy(); delete charts['ch-contrib']; } computeExplorer(); });

  // ---------------------------------------------------------------- 10 gallery
  (function gallery() {
    var figs = [
      ['loan_maturity_censoring.png', 'Right-censoring audit', 'Share of 36-month loans still unresolved by issue month; the cut-off at Dec 2015.'],
      ['eda_default_rate_and_macro.png', 'Default rate and macro inputs', 'Monthly default rate with the 2-month-lagged unemployment, its 6-month change and inflation.'],
      ['eda_borrower_signal.png', 'Borrower signal', 'Training-window default rate by FICO, DTI, income and home ownership.'],
      ['model_c_odds_ratios.png', 'Model C odds ratios', 'statsmodels Logit with month-clustered 95% intervals.'],
      ['holdout_evaluation_curves.png', 'Holdout curves', 'Reliability, precision–recall and ROC on 2015.'],
      ['vintage_calibration.png', 'Out-of-time calibration', 'Monthly actual vs predicted default rate, 2013–2015.'],
      ['segment_errors.png', 'Segment errors', 'Actual vs Model C predicted default rate by segment, 2015.']
    ];
    $('#gallery').innerHTML = figs.map(function (f) { return '<figure class="fig"><button type="button" data-src="assets/figures/' + f[0] + '" aria-label="Enlarge ' + esc(f[1]) + '"><img loading="lazy" src="assets/figures/' + f[0] + '" alt="' + esc(f[1] + ': ' + f[2]) + '"></button><figcaption><b>' + esc(f[1]) + '</b>' + esc(f[2]) + '</figcaption></figure>'; }).join('');
    var lb = $('#lightbox'), img = $('img', lb);
    $$('#gallery button').forEach(function (b) { b.addEventListener('click', function () { img.src = b.dataset.src; img.alt = b.getAttribute('aria-label'); lb.classList.add('open'); }); });
    lb.addEventListener('click', function () { lb.classList.remove('open'); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') lb.classList.remove('open'); });
  })();

  // ---------------------------------------------------------------- theme + nav
  $('#theme-toggle').addEventListener('click', function () {
    var root = document.documentElement, cur = root.getAttribute('data-theme');
    var dark = cur ? cur === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    var next = dark ? 'light' : 'dark'; root.setAttribute('data-theme', next);
    try { localStorage.setItem('site-theme', next); } catch (e) { /* storage unavailable */ }
    builders.forEach(function (fn) { fn(); });
  });
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onChange = function () { if (!document.documentElement.getAttribute('data-theme')) builders.forEach(function (fn) { fn(); }); };
    if (mq.addEventListener) mq.addEventListener('change', onChange);
  }
  if ('IntersectionObserver' in window) {
    var links = $$('#nav-links a'), map = {};
    links.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting && map[e.target.id]) { links.forEach(function (a) { a.classList.remove('active'); }); map[e.target.id].classList.add('active'); } });
    }, { rootMargin: '-45% 0px -50% 0px' });
    Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }
})();
