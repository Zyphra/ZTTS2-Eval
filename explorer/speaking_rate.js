/* ============================================================
   ZTTS2 · Eval — speaking-rate views (scripts/prosody/speaking_rate_dist.py, live)
   Interactive SVG twins of the char/s distributional metric, in the
   Zyphra theme. Reads /api/speaking_rate (built by
   scripts/results_doc/build_speaking_rate.py).
   Depends on globals from app.js ($, $$, el, svgEl, apiGet)
   and the chart kit + R_* palette + SMALL_N from
   chart_common.js (loaded first).
   ============================================================ */
"use strict";

const SR_GREEN = "#1f9d55";   // ceiling / reference "natural speech" accent (zyphra green)

// sort: row order of the distribution view — "pace" (fastest median first) or "dref"
const SR = { data: null, panel: null, type: "dref", sort: "pace", pin: null };

// Panels are one per (test-set × language) — 50-odd of them, which is too many for a
// single chip row. Split it the way the reliability view splits dataset from language:
// pick the test set first, then the language within it.
function srSets() {
  const out = [];
  SR.data.panels.forEach((p) => {
    if (!out.some((s) => s.key === p.set)) out.push({ key: p.set, label: p.set_label });
  });
  return out;
}
function srPanelsIn(set) { return SR.data.panels.filter((p) => p.set === set); }
function srSetOf(key) {
  const p = SR.data.panels.find((x) => x.key === key);
  return p ? p.set : null;
}

// nice 1-2-5 tick step (char/s distances are fractional, unlike edit counts)
function srNiceNum(x, round) {
  if (!(x > 0)) return 1;
  const exp = Math.floor(Math.log10(x)), f = x / Math.pow(10, exp);
  const nf = round ? (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10)
                   : (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10);
  return nf * Math.pow(10, exp);
}
function srTicks(lo, hi, n) {
  if (!(hi > lo)) return [lo || 0, (lo || 0) + 1];
  const step = srNiceNum((hi - lo) / (n || 5), true);
  const start = Math.floor(lo / step) * step;
  const ticks = [];
  for (let t = start; t <= hi + step * 0.5; t += step) ticks.push(+t.toFixed(6));
  return ticks;
}
const srFmt = (v, step) => (step && step < 1 ? v.toFixed(step < 0.1 ? 2 : 1) : String(Math.round(v)));

// ---- interactivity (chart_common kit, parametrized for this view) -----------
const srTipKit = makeTip("#sr-chart", "#sr-tip");

// Display label + focus key. A payload can hold two gt_* series: the GT system itself
// (split null) and the reference's split-half twin ("random 50/50" — the ceiling), so
// the twin needs its own label and key or the two read and highlight as one.
const srLabel = (r) => cleanName(r.name) + (r.split ? " · reference half" : "");
const srKey = (r) => (r.split ? `${r.name}#half` : r.name);

function srSummary(r, col) {
  const xc = r.x_ceiling != null ? ` · ${r.x_ceiling}× ceiling` : "";
  const head = `<span class="sw" style="background:${col || R_PRIMARY}"></span><b>${srLabel(r)}</b><br>`;
  if (!r.ci)    // an "all" macro row: no interval, no single median
    return head + `d<sub>ref</sub> <b>${r.d_ref.toFixed(2)}</b> char/s · macro over ${r.n_langs} languages<br>` +
      `<span class="meta">${xc ? xc.slice(3) + " · " : ""}n ${r.n.toLocaleString()}</span>`;
  return head +
    `d<sub>ref</sub> <b>${r.d_ref.toFixed(2)}</b> char/s · 95% CI [${r.ci[0].toFixed(2)}, ${r.ci[1].toFixed(2)}]<br>` +
    `<span class="meta">median ${r.median} char/s${r.heard_pct != null ? ` · scored ${r.heard_pct}% (WER ≤ 20%)` : ""} · ` +
    `trunc ${r.truncation_pct}%${xc} · n ${r.n.toLocaleString()}</span>`;
}

// focus emphasises the primary paths via stroke width (colour already encodes
// the system here, unlike the reliability plates' black recolour)
const setSrFocus = makeFocus("#sr-chart", {
  dim: "0.12",
  emph: (e, on) => {
    if (on) { e.style.strokeWidth = "3"; if (e.parentNode) e.parentNode.appendChild(e); }
    else { e.style.strokeWidth = ""; }
  },
});

const srBind = makeBind(setSrFocus, srTipKit, SR, srSummary);
function srReapply() { if (SR.pin) setSrFocus(SR.pin); }

// ---------------------------------------------------------------------------
function ensureSr() {
  return ensureCache(SR, "/api/speaking_rate",
    (d) => !!(d.panels && d.panels.length),
    (d) => {
      addSrMacroPanels(d);
      // open on the first set's "all" panel (the macro), like the board and the bins
      if (!SR.panel || !d.panels.some((x) => x.key === SR.panel))
        SR.panel = (d.panels.find((x) => x.aggregate) || d.panels.find((x) => x.lang === "en") || d.panels[0] || {}).key;
    });
}

// One "all" panel per test set: each system's d_ref (and ×ceiling) macro-averaged over
// the set's languages, only for systems scored on every one of them — a mean over a
// different language subset is a different quantity. It carries no interval (per-language
// bootstrap intervals don't combine into one) and no distribution: char/s is only
// comparable within a language, so the pace strips stay per-language.
const SR_ALL = "all";
function addSrMacroPanels(d) {
  const sets = [];
  d.panels.forEach((p) => { if (!sets.includes(p.set)) sets.push(p.set); });
  sets.forEach((set) => {
    const ps = d.panels.filter((p) => p.set === set);
    const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
    const series = [];
    ps[0].series.forEach((r0) => {
      const rs = ps.map((p) => p.series.find((r) => srKey(r) === srKey(r0)));
      if (rs.some((r) => !r)) return;
      const xcs = rs.map((r) => r.x_ceiling);
      series.push({
        name: r0.name, split: r0.split, seed: r0.seed, n_langs: ps.length,
        n: rs.reduce((s, r) => s + r.n, 0),
        d_ref: +mean(rs.map((r) => r.d_ref)).toFixed(3),
        x_ceiling: xcs.every((v) => v != null) ? +mean(xcs).toFixed(2) : null,
        ci: null,
      });
    });
    const agg = {
      key: `${set}|${SR_ALL}`, set, set_label: ps[0].set_label, lang: SR_ALL, aggregate: true,
      n_langs: ps.length, ceiling: mean(ps.map((p) => p.ceiling)),
      ref: { n: ps.reduce((s, p) => s + p.ref.n, 0) }, series,
    };
    d.panels.splice(d.panels.indexOf(ps[0]), 0, agg);   // heads its set's chip row
  });
}

function curPanel() {
  return SR.data.panels.find((p) => p.key === SR.panel) || SR.data.panels[0];
}

window.renderSpeaking = async function () {
  const chart = $("#sr-chart");
  const ok = await ensureSr();
  if (!ok) {
    chart.querySelectorAll("svg").forEach((n) => n.remove());
    chart.style.height = "auto";
    if (!chart.querySelector(".loading"))
      chart.append(el("div", "loading", "no speaking-rate cache on disk — run build_speaking_rate.py"));
    $("#sr-legend").innerHTML = ""; $("#sr-panels").innerHTML = ""; $("#sr-sets").innerHTML = "";
    $("#sr-source").innerHTML = "";
    $("#ro-channel").textContent = "speaking rate / —";
    return;
  }
  chart.querySelectorAll(".loading").forEach((n) => n.remove());
  const p = curPanel();
  $("#ro-channel").textContent = `speaking rate / ${p.lang}`;
  renderSrSets();
  renderSrPanels();
  bindSrTypeSeg(p);
  updateSrHead(p);
  drawSr(p);
  const src = $("#sr-source");
  if (src) src.innerHTML =
    `<span><b>Data:</b> ZTTS2-Eval · ${p.set_label} · ` +
    (p.aggregate ? `macro over ${p.n_langs} languages` : p.lang) + ` · characters per second over trimmed speech, ` +
    `2-Wasserstein vs the reference-voice distribution</span><span class="lic">CC BY</span>`;
};

// Selecting a panel resets the per-chart series state, which is scoped to one panel.
function pickSrPanel(key) {
  SR.panel = key; SR.pin = null;
  window.renderSpeaking();
}

function renderSrSets() {
  const box = $("#sr-sets");
  if (!box) return;
  const cur = srSetOf(SR.panel);
  box.innerHTML = "";
  srSets().forEach((s) => {
    const c = el("button", "lang-chip" + (s.key === cur ? " on" : ""), s.label);
    c.onclick = () => {
      // stay on the same language across sets where it exists — switching test set
      // to compare the same language is the common move
      const inSet = srPanelsIn(s.key);
      const lang = (SR.data.panels.find((p) => p.key === SR.panel) || {}).lang;
      const keep = inSet.find((p) => p.lang === lang) || inSet.find((p) => p.lang === "en") || inSet[0];
      if (keep) pickSrPanel(keep.key);
    };
    box.append(c);
  });
}

// Language picker for the active test set — chip is the bare language code, since
// the set is already named by the row above.
function renderSrPanels() {
  const box = $("#sr-panels");
  const inSet = srPanelsIn(srSetOf(SR.panel));
  if (inSet.length < 2) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  box.innerHTML = "";
  inSet.forEach((p) => {
    const c = el("button", "lang-chip" + (p.key === SR.panel ? " on" : ""), p.lang);
    c.onclick = () => pickSrPanel(p.key);
    box.append(c);
  });
}

let _srSegBound = false;
function bindSrTypeSeg(p) {
  // an "all" panel has no distributions to draw (char/s is only comparable within a
  // language): the switch stays, its distribution half disabled, so the option is
  // visibly there one language-pick away. SR.type is kept for when a language returns.
  const shown = p.aggregate ? "dref" : SR.type;
  $$("#sr-type-seg button").forEach((b) => {
    const off = !!p.aggregate && b.dataset.sr === "dist";
    b.disabled = off;
    b.title = off ? "per language only: characters per second aren't comparable across languages — pick a language" : "";
    b.classList.toggle("on", b.dataset.sr === shown);
  });
  $$("#sr-sort-seg button").forEach((b) => b.classList.toggle("on", b.dataset.sort === SR.sort));
  $("#sr-sort-seg").hidden = shown !== "dist";   // row order only applies to the strips
  if (_srSegBound) return;
  _srSegBound = true;
  $$("#sr-type-seg button").forEach((b) => {
    b.onclick = () => { SR.type = b.dataset.sr; window.renderSpeaking(); };
  });
  $$("#sr-sort-seg button").forEach((b) => {
    b.onclick = () => { SR.sort = b.dataset.sort; window.renderSpeaking(); };
  });
}

function updateSrHead(p) {
  if (p.aggregate) {
    $("#sr-title").textContent = "Speaking rate — distance from natural pacing, all languages";
    $("#sr-note").textContent =
      `Dot = d ref (2-Wasserstein between the system's and the reference voice's char/s) averaged over ` +
      `the set's ${p.n_langs} languages, for the ${p.series.length} systems scored on every one of them. ` +
      `No interval: per-language bootstrap intervals don't combine into one. Lower = closer to natural ` +
      `pace; the green rule is the mean split-half ceiling. Pace distributions are per language only.`;
    return;
  }
  const stat = `${p.series.length} system${p.series.length !== 1 ? "s" : ""} · ${p.lang} · ` +
    `reference n ${p.ref.n.toLocaleString()} · ceiling ${p.ceiling.toFixed(2)} char/s`;
  const H = {
    dref: ["Speaking rate — distance from natural pacing",
      `Dot = d ref (2-Wasserstein between the system's and the reference voice's char/s), ` +
      `line = bootstrap 95% CI. Lower = closer to natural pace; the green rule is the split-half ceiling, ` +
      `what a perfect system scores at finite n. ${stat}.`],
    dist: ["Pace distributions against natural speech",
      `One row per system: its distribution of speaking rate (characters per second of trimmed speech) ` +
      `over the test set. The shaded ridge is the shape, the bar the middle half (p25–p75) with whiskers to ` +
      `p10 and p90, the white tick the median. The green band and line are the reference voice's middle half ` +
      `and median, and its shape is outlined faintly in every row. "vs natural" is the median's difference ` +
      `from the reference median (+ = faster). ${stat}.`],
  };
  const [t, n] = H[SR.type] || H.dref;
  $("#sr-title").textContent = t;
  $("#sr-note").textContent = n;
}

function clearSrChart() {
  const chart = $("#sr-chart");
  chart.querySelectorAll("svg").forEach((n) => n.remove());
  $("#sr-tip").hidden = true;
  chart.style.overflow = "visible";
  return chart;
}

function drawSr(p) {
  if (SR.type === "dist" && !p.aggregate) return drawSrDist(p);
  return drawSrDref(p);
}

// ---------------------------------------------------------------------------
// d · CI  (horizontal dot + bootstrap CI, sorted by d_ref, ceiling rule)
// ---------------------------------------------------------------------------
function drawSrDref(p) {
  const chart = clearSrChart();
  $("#sr-legend").innerHTML = "";
  const rows = p.series.slice().sort((a, b) => a.d_ref - b.d_ref);

  const W = Math.max(360, chart.clientWidth);
  const tagOf = (r) => (r.x_ceiling != null ? `${r.d_ref.toFixed(2)}  ${r.x_ceiling}×` : r.d_ref.toFixed(2));
  const ml = Math.min(260, Math.max(64, labelWidth(chart, "series-lab", rows.map(srLabel)) + 16));
  const mr = Math.max(60, labelWidth(chart, "axis-txt", rows.map(tagOf)) + 16), mt = 16, mb = 46, rowH = 24;
  const plotW = W - ml - mr;
  const H = mt + rows.length * rowH + mb;

  const span = (r) => r.ci || [r.d_ref, r.d_ref];   // "all" rows have no interval
  const xmax = Math.max(p.ceiling, ...rows.map((r) => span(r)[1])) * 1.04;
  const ticks = srTicks(0, xmax, 5);
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
  const tickMax = ticks[ticks.length - 1];
  // the ticks can stop short of xmax: map the full data range so CI ends stay in the plot
  const x = linMap(0, Math.max(tickMax, xmax), ml, ml + plotW);
  const gap = 0.006 * tickMax;

  const svg = newSvg(W, H);
  const y0 = mt, y1 = mt + rows.length * rowH;

  // faint vertical gridlines at ticks (owid: no frame; 0 is the baseline)
  ticks.forEach((t) => {
    svg.append(svgEl("line", { class: t === 0 ? "zero-line" : "gridline", x1: x(t), y1: y0, x2: x(t), y2: y1 }));
    svg.append(stext("axis-txt", x(t), y1 + 16, srFmt(t, step), { "text-anchor": "middle" }));
  });

  // ceiling reference rule (green, dashed) — perfect system at finite n
  const xc = x(p.ceiling);
  svg.append(svgEl("line", { x1: xc, y1: y0 - 4, x2: xc, y2: y1, stroke: SR_GREEN, "stroke-width": 1.4, "stroke-dasharray": "4 3" }));
  svg.append(stext("axis-txt", xc, y0 - 7, "ceiling", { "text-anchor": "middle", fill: SR_GREEN }));

  svg.append(stext("axis-lab", ml + plotW / 2, H - 6,
    "d ref — 2-Wasserstein distance in char/s from the reference pace (fewer = closer to natural)",
    { "text-anchor": "middle" }));

  rows.forEach((r, idx) => {
    const yi = mt + idx * rowH + rowH / 2;
    const small = r.n < SMALL_N;
    const c = small ? R_MUT : R_PRIMARY;
    const [lo, hi] = span(r);
    if (lo < r.d_ref - gap)
      mark(svg.appendChild(svgEl("line", { x1: x(lo), y1: yi, x2: x(r.d_ref - gap), y2: yi, stroke: R_RULE, "stroke-width": 1.4 })), srKey(r));
    if (hi > r.d_ref + gap)
      mark(svg.appendChild(svgEl("line", { x1: x(r.d_ref + gap), y1: yi, x2: x(hi), y2: yi, stroke: R_RULE, "stroke-width": 1.4 })), srKey(r));
    mark(svg.appendChild(svgEl("circle", { class: "dot", cx: x(r.d_ref), cy: yi, r: 3.6, fill: c })), srKey(r));
    mark(svg.appendChild(stext("series-lab", ml - 8, yi, srLabel(r), { "text-anchor": "end", "dominant-baseline": "middle", fill: R_INK })), srKey(r));
    mark(svg.appendChild(stext("axis-txt", x(hi) + 8, yi, tagOf(r),
      { "text-anchor": "start", "dominant-baseline": "middle", fill: R_MUT })), srKey(r));
    const band = svgEl("rect", { x: ml, y: yi - rowH / 2, width: plotW, height: rowH, fill: "transparent" });
    srBind(band, srKey(r), r);
    svg.append(band);
  });

  chart.append(svg);
  srReapply();
}

// ---------------------------------------------------------------------------
// distribution  (pace strips: one row per system, shared char/s axis)
// ---------------------------------------------------------------------------
// Each payload series carries its quantile function `q` on the shared probability
// grid P (p2..p98). From it a row draws: a density ridge (the shape), a p10–p90
// whisker + p25–p75 bar + median tick (the summary), and the reference voice's
// ridge as a faint green outline (the comparison). One name per row, so nothing
// collides however many systems a panel has.

// quantile at probability p, interpolated on the grid
function srQuantile(q, P, p) {
  if (p <= P[0]) return q[0];
  if (p >= P[P.length - 1]) return q[q.length - 1];
  let j = 1;
  while (P[j] < p) j++;
  const t = (p - P[j - 1]) / (P[j] - P[j - 1]);
  return q[j - 1] + t * (q[j] - q[j - 1]);
}

// Density on the grid `xs` from the quantile function: each quantile step holds
// probability mass ΔP spread uniformly over its char/s span (at least one cell, so
// tied quantiles don't spike), then a Gaussian smooth. The p<2% / p>98% tails are
// not in the payload and are left out. Scaled to a peak of 1.
function srDensity(q, P, xs) {
  const n = xs.length, dx = xs[1] - xs[0], raw = new Float64Array(n);
  for (let j = 1; j < q.length; j++) {
    const m = P[j] - P[j - 1];
    const w = Math.max(q[j] - q[j - 1], dx);
    const lo = (q[j - 1] + q[j]) / 2 - w / 2, hi = lo + w;
    const i0 = Math.max(0, Math.floor((lo - xs[0]) / dx)), i1 = Math.min(n - 1, Math.ceil((hi - xs[0]) / dx));
    for (let i = i0; i <= i1; i++) {
      const c0 = xs[i] - dx / 2, ov = Math.min(hi, c0 + dx) - Math.max(lo, c0);
      if (ov > 0) raw[i] += m * ov / w;
    }
  }
  const sig = 2, rad = 3 * sig, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0, ws = 0;
    for (let k = -rad; k <= rad; k++) {
      const g = Math.exp(-(k * k) / (2 * sig * sig));
      if (i + k >= 0 && i + k < n) s += raw[i + k] * g;
      ws += g;
    }
    out[i] = s / ws;
  }
  const mx = Math.max(...out) || 1;
  return Array.from(out, (v) => v / mx);
}

function srDistSummary(r, ref) {
  const P = SR.data.P, Q = (p) => srQuantile(r.q, P, p).toFixed(1);
  const dv = (r.median / ref.median - 1) * 100;
  const pace = Math.abs(dv) < 0.5 ? "same median pace as natural speech"
    : `${Math.abs(dv).toFixed(0)}% ${dv > 0 ? "faster" : "slower"} than natural speech`;
  return srSummary(r, srRowColor(r)) +
    `<br><span class="meta">p10 ${Q(0.1)} · p25 ${Q(0.25)} · <b>p50 ${r.median}</b> · p75 ${Q(0.75)} · p90 ${Q(0.9)} char/s<br>${pace}</span>`;
}

// real-speech baselines (gt_*) are drawn grey: they are the floor, not a competitor
const srRowColor = (r) => (r.name.startsWith("gt_") || r.n < SMALL_N ? R_MID : R_PRIMARY);

function drawSrDist(p) {
  const chart = clearSrChart();
  const P = SR.data.P, ref = p.ref;
  const rows = p.series.slice().sort(SR.sort === "dref"
    ? (a, b) => a.d_ref - b.d_ref
    : (a, b) => b.median - a.median);

  const W = Math.max(520, chart.clientWidth);
  const ml = Math.min(230, Math.max(120,
    labelWidth(chart, "series-lab", [...rows.map(srLabel), "natural speech (reference)"]) + 20));
  const cols = [["median", "char/s", 56], ["vs natural", "", 74], ["d", "×ceil", 58]];
  const mr = cols.reduce((s, c) => s + c[2], 0) + 12;
  const rowH = 30, mt = 58, mb = 46;
  const plotW = W - ml - mr;
  const nRows = rows.length + 1;                       // +1: the reference itself, on top
  const H = mt + nRows * rowH + mb;
  const y0 = mt, y1 = mt + nRows * rowH;

  // shared char/s axis over every row's p10..p90 (whiskers always on scale), padded.
  // The p2..p98 tails would let one extreme system squash every other row into the
  // middle third of the plot; ridges beyond the axis are clipped instead.
  const all = [ref.q].concat(rows.map((r) => r.q));
  const lo = Math.min(...all.map((q) => srQuantile(q, P, 0.1)));
  const hi = Math.max(...all.map((q) => srQuantile(q, P, 0.9)));
  const pad = (hi - lo) * 0.06 || 1;
  const xt = srTicks(lo - pad, hi + pad, 8).filter((t) => t >= lo - pad && t <= hi + pad);
  const x0 = lo - pad, x1 = hi + pad;
  const xstep = xt.length > 1 ? xt[1] - xt[0] : 1;
  const X = linMap(x0, x1, ml, ml + plotW);
  const NX = 220, xs = Array.from({ length: NX }, (_, i) => x0 + (x1 - x0) * i / (NX - 1));

  const svg = newSvg(W, H);
  const defs = svgEl("defs", {}), clip = svgEl("clipPath", { id: "sr-plot-clip" });
  clip.append(svgEl("rect", { x: ml, y: 0, width: plotW, height: H }));
  defs.append(clip); svg.append(defs);

  // reference middle half + median: the vertical yardstick through every row
  const r25 = srQuantile(ref.q, P, 0.25), r75 = srQuantile(ref.q, P, 0.75);
  svg.append(svgEl("rect", { x: X(r25), y: y0, width: X(r75) - X(r25), height: y1 - y0, fill: SR_GREEN, opacity: 0.09 }));
  xt.forEach((t) => {
    svg.append(svgEl("line", { class: "gridline", x1: X(t), y1: y0, x2: X(t), y2: y1 }));
    svg.append(stext("axis-txt", X(t), y0 - 8, srFmt(t, xstep), { "text-anchor": "middle" }));
    svg.append(stext("axis-txt", X(t), y1 + 16, srFmt(t, xstep), { "text-anchor": "middle" }));
  });
  svg.append(svgEl("line", { x1: X(ref.median), y1: y0 - 2, x2: X(ref.median), y2: y1, stroke: SR_GREEN, "stroke-width": 1.4, "stroke-dasharray": "4 3" }));
  svg.append(stext("axis-txt", X(ref.median), y0 - 24, `natural median ${ref.median}`, { "text-anchor": "middle", fill: SR_GREEN }));
  svg.append(stext("axis-lab", ml + plotW / 2, H - 6,
    "speaking rate (characters per second of trimmed speech) — left = slower, right = faster", { "text-anchor": "middle" }));

  // right-hand column headers
  let cx = ml + plotW + 12;
  const colX = cols.map(([name, unit, w]) => {
    const xr = cx + w - 6;
    svg.append(stext("axis-txt", xr, y0 - 24, name, { "text-anchor": "end", "font-weight": 700 }));
    if (unit) svg.append(stext("axis-txt", xr, y0 - 10, unit, { "text-anchor": "end" }));
    cx += w;
    return xr;
  });

  const refDen = srDensity(ref.q, P, xs);
  const ridge = (den, yb, h) => {
    let d = `M${X(xs[0]).toFixed(1)} ${yb}`;
    den.forEach((v, i) => { d += ` L${X(xs[i]).toFixed(1)} ${(yb - v * h).toFixed(1)}`; });
    return d + ` L${X(xs[NX - 1]).toFixed(1)} ${yb}`;
  };

  const drawRow = (r, idx, isRef) => {
    const top = mt + idx * rowH, yb = top + rowH - 5, ym = top + rowH - 9;
    const col = isRef ? SR_GREEN : srRowColor(r);
    const g = svgEl("g", {});
    if (!isRef) mark(g, srKey(r));
    if (idx % 2 === 1)
      g.append(svgEl("rect", { x: 0, y: top, width: W, height: rowH, fill: "#000", opacity: 0.018 }));
    // the reference's shape, faint, in every system row
    if (!isRef) g.append(svgEl("path", { d: ridge(refDen, yb, rowH - 9), fill: "none", stroke: SR_GREEN, "stroke-width": 1, opacity: 0.45, "clip-path": "url(#sr-plot-clip)" }));
    g.append(svgEl("path", { d: ridge(isRef ? refDen : srDensity(r.q, P, xs), yb, rowH - 9),
      fill: col, "fill-opacity": 0.16, stroke: col, "stroke-opacity": 0.55, "stroke-width": 1, "clip-path": "url(#sr-plot-clip)" }));
    const q10 = srQuantile(r.q, P, 0.1), q25 = srQuantile(r.q, P, 0.25);
    const q75 = srQuantile(r.q, P, 0.75), q90 = srQuantile(r.q, P, 0.9);
    g.append(svgEl("line", { x1: X(q10), y1: ym, x2: X(q90), y2: ym, stroke: col, "stroke-width": 1.4 }));
    g.append(svgEl("rect", { x: X(q25), y: ym - 3.5, width: Math.max(1, X(q75) - X(q25)), height: 7, rx: 2, fill: col }));
    g.append(svgEl("line", { x1: X(r.median), y1: ym - 5, x2: X(r.median), y2: ym + 5, stroke: "#fafaf9", "stroke-width": 2 }));
    g.append(stext("series-lab", ml - 10, top + rowH / 2, isRef ? "natural speech (reference)" : srLabel(r),
      { "text-anchor": "end", "dominant-baseline": "middle", fill: isRef ? SR_GREEN : R_INK }));
    // numbers: median · vs natural · d ×ceil
    const dv = (r.median / ref.median - 1) * 100;
    const vals = [r.median.toFixed(1),
      isRef ? "—" : `${dv > 0 ? "+" : dv < 0 ? "−" : "±"}${Math.abs(dv).toFixed(0)}%`,
      isRef || r.x_ceiling == null ? "—" : `${r.x_ceiling}×`];
    vals.forEach((v, k) => g.append(stext("axis-txt", colX[k], top + rowH / 2, v,
      { "text-anchor": "end", "dominant-baseline": "middle",
        fill: k === 1 && !isRef && Math.abs(dv) >= 5 ? R_INK : R_MUT, "font-weight": k === 1 && Math.abs(dv) >= 5 ? 700 : 400 })));
    if (!isRef) {
      const band = svgEl("rect", { x: 0, y: top, width: W, height: rowH, fill: "transparent" });
      srBind(band, srKey(r), null);
      band.addEventListener("mouseenter", (ev) => srTipKit.show(srDistSummary(r, ref), ev));
      band.addEventListener("mousemove", (ev) => srTipKit.show(srDistSummary(r, ref), ev));
      g.append(band);
    }
    svg.append(g);
  };
  drawRow(ref, 0, true);
  rows.forEach((r, i) => drawRow(r, i + 1, false));

  chart.append(svg);
  renderSrKey();
  srReapply();
}

// static key for the strip glyphs (the rows carry their own names)
function renderSrKey() {
  const box = $("#sr-legend");
  box.innerHTML = "";
  const item = (svgInner, text) => {
    const it = el("div", "leg-key");
    it.innerHTML = `<svg width="34" height="14" viewBox="0 0 34 14">${svgInner}</svg>`;
    it.append(el("span", "", text));
    box.append(it);
  };
  item(`<path d="M1 13 C9 13 10 2 17 2 C24 2 25 13 33 13" fill="${R_PRIMARY}" fill-opacity=".16" stroke="${R_PRIMARY}" stroke-opacity=".55"/>`, "distribution shape");
  item(`<line x1="2" y1="7" x2="32" y2="7" stroke="${R_PRIMARY}" stroke-width="1.4"/><rect x="10" y="3.5" width="14" height="7" rx="2" fill="${R_PRIMARY}"/><line x1="17" y1="2" x2="17" y2="12" stroke="#fafaf9" stroke-width="2"/>`, "p10 · p25–p75 · median · p90");
  item(`<rect x="8" y="0" width="18" height="14" fill="${SR_GREEN}" opacity=".18"/><line x1="17" y1="0" x2="17" y2="14" stroke="${SR_GREEN}" stroke-width="1.4" stroke-dasharray="4 3"/>`, "natural speech: middle half · median");
  item(`<path d="M1 13 C9 13 10 2 17 2 C24 2 25 13 33 13" fill="none" stroke="${SR_GREEN}" stroke-opacity=".6"/>`, "natural speech shape");
}

// redraw on resize while the speaking-rate view is active
bindChartResize("#sr-chart", "speaking", () => !!SR.data, () => window.renderSpeaking());
