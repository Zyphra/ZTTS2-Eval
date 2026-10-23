/* ============================================================
   ZTTS2 · Eval — speaking-rate views (scripts/speaking_rate_dist.py, live)
   Interactive SVG twins of the char/s distributional metric, in the
   "Our World in Data" theme. Reads /api/speaking_rate (built by
   scripts/results_doc/build_speaking_rate.py).
   Depends on globals from app.js ($, $$, el, svgEl, apiGet,
   BIN_PALETTE) and the chart kit + R_* palette + SMALL_N from
   chart_common.js (loaded first).
   ============================================================ */
"use strict";

const SR_GREEN = "#417a4f";   // ceiling / reference "natural speech" accent

const SR = { data: null, panel: null, type: "dref", hidden: new Set(),
  pin: null };

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

function srSummary(r, col) {
  const xc = r.x_ceiling != null ? ` · ${r.x_ceiling}× ceiling` : "";
  return `<span class="sw" style="background:${col || R_NAVY}"></span><b>${r.name}</b><br>` +
    `d<sub>ref</sub> <b>${r.d_ref.toFixed(2)}</b> char/s · 95% CI [${r.ci[0].toFixed(2)}, ${r.ci[1].toFixed(2)}]<br>` +
    `<span class="meta">median ${r.median} char/s · trunc ${r.truncation_pct}%${xc} · n ${r.n.toLocaleString()}</span>`;
}

// focus emphasises the primary paths via stroke width (colour already encodes
// the system here, unlike the reliability plates' navy recolour)
const setSrFocus = makeFocus("#sr-chart", {
  dim: "0.12",
  emph: (e, on) => {
    if (on) { e.style.strokeWidth = "3"; if (e.parentNode) e.parentNode.appendChild(e); }
    else { e.style.strokeWidth = ""; }
  },
});

const srBind = makeBind(setSrFocus, srTipKit, SR, srSummary);
const srHit = makeHit(srBind);
function srReapply() { if (SR.pin) setSrFocus(SR.pin); }

// ---------------------------------------------------------------------------
function ensureSr() {
  return ensureCache(SR, "/api/speaking_rate",
    (d) => !!(d.panels && d.panels.length),
    (d) => {
      if (!SR.panel || !d.panels.some((x) => x.key === SR.panel))
        SR.panel = (d.panels.find((x) => x.lang === "en") || d.panels[0] || {}).key;
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
  bindSrTypeSeg();
  updateSrHead(p);
  drawSr(p);
  const src = $("#sr-source");
  if (src) src.innerHTML =
    `<span><b>Data:</b> ZTTS2-Eval · ${p.set_label} · ${p.lang} · characters per second over trimmed speech, ` +
    `2-Wasserstein vs the reference-voice distribution</span><span class="lic">CC BY</span>`;
};

// Selecting a panel resets the per-chart series state, which is scoped to one panel.
function pickSrPanel(key) {
  SR.panel = key; SR.hidden = new Set(); SR.pin = null;
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
function bindSrTypeSeg() {
  $$("#sr-type-seg button").forEach((b) => b.classList.toggle("on", b.dataset.sr === SR.type));
  if (_srSegBound) return;
  _srSegBound = true;
  $$("#sr-type-seg button").forEach((b) => {
    b.onclick = () => {
      SR.type = b.dataset.sr;
      $$("#sr-type-seg button").forEach((x) => x.classList.toggle("on", x === b));
      window.renderSpeaking();
    };
  });
}

function updateSrHead(p) {
  const stat = `${p.series.length} system${p.series.length !== 1 ? "s" : ""} · ${p.lang} · ` +
    `reference n ${p.ref.n.toLocaleString()} · ceiling ${p.ceiling.toFixed(2)} char/s`;
  const H = {
    dref: ["Speaking rate — distance from natural pacing",
      `Dot = d ref (2-Wasserstein between the system's and the reference voice's char/s), ` +
      `line = bootstrap 95% CI. Lower = closer to natural pace; the green rule is the split-half ceiling, ` +
      `what a perfect system scores at finite n. ${stat}.`],
    dist: ["The pacing distributions themselves",
      `Cumulative share of utterances at or below each speaking rate (char/s). The reference voice is the ` +
      `thick green line; a system sitting to its left reads faster, to its right slower. The horizontal gap ` +
      `is exactly what d ref measures. Click a legend chip to drop or restore a system. ${stat}.`],
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
  if (SR.type === "dist") return drawSrDist(p);
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
  const maxLabel = Math.max(...rows.map((r) => r.name.length), 3);
  const ml = Math.min(200, Math.max(64, maxLabel * 7.4 + 20));
  const mr = 92, mt = 16, mb = 46, rowH = 24;
  const plotW = W - ml - mr;
  const H = mt + rows.length * rowH + mb;

  const xmax = Math.max(p.ceiling, ...rows.map((r) => r.ci[1])) * 1.04;
  const ticks = srTicks(0, xmax, 5);
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
  const tickMax = ticks[ticks.length - 1];
  const x = linMap(0, tickMax, ml, ml + plotW);
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
    const c = small ? R_MUT : R_NAVY;
    const [lo, hi] = r.ci;
    if (lo < r.d_ref - gap)
      mark(svg.appendChild(svgEl("line", { x1: x(lo), y1: yi, x2: x(r.d_ref - gap), y2: yi, stroke: R_RULE, "stroke-width": 1.4 })), r.name);
    if (hi > r.d_ref + gap)
      mark(svg.appendChild(svgEl("line", { x1: x(r.d_ref + gap), y1: yi, x2: x(hi), y2: yi, stroke: R_RULE, "stroke-width": 1.4 })), r.name);
    mark(svg.appendChild(svgEl("circle", { class: "dot", cx: x(r.d_ref), cy: yi, r: 3.6, fill: c })), r.name);
    mark(svg.appendChild(stext("series-lab", ml - 8, yi, r.name, { "text-anchor": "end", "dominant-baseline": "middle", fill: R_INK })), r.name);
    const tag = r.x_ceiling != null ? `${r.d_ref.toFixed(2)}  ${r.x_ceiling}×` : r.d_ref.toFixed(2);
    mark(svg.appendChild(stext("axis-txt", x(hi) + 8, yi, tag,
      { "text-anchor": "start", "dominant-baseline": "middle", fill: R_MUT })), r.name);
    const band = svgEl("rect", { x: ml, y: yi - rowH / 2, width: plotW, height: rowH, fill: "transparent" });
    srBind(band, r.name, r);
    svg.append(band);
  });

  chart.append(svg);
  srReapply();
}

// ---------------------------------------------------------------------------
// distribution  (ECDF of char/s: reference + each system, linear axes)
// ---------------------------------------------------------------------------
// Colour is keyed on the system's index in the FULL p.series list, not its
// position among the currently visible systems — otherwise hiding one system
// from the legend shifts every subsequent system onto a different colour (the
// instability binColor in app.js documents avoiding). Hidden systems leave gaps.
function srColorMap(p) {
  return new Map(p.series.map((r, i) => [r.name, BIN_PALETTE[i % BIN_PALETTE.length]]));
}

function drawSrDist(p) {
  const chart = clearSrChart();
  const P = SR.data.P;
  const visible = p.series.filter((r) => !SR.hidden.has(r.name));

  const W = Math.max(360, chart.clientWidth);
  const H = 460, ml = 46, mr = 128, mt = 16, mb = 48;
  const plotW = W - ml - mr, plotH = H - mt - mb;
  const svg = newSvg(W, H);

  // x domain across the reference + every visible system
  const qs = [p.ref.q].concat(visible.map((r) => r.q)).filter((q) => q && q.length);
  const lo = Math.min(...qs.map((q) => q[0]));
  const hi = Math.max(...qs.map((q) => q[q.length - 1]));
  const pad = (hi - lo) * 0.04 || 1;
  const xt = srTicks(lo - pad, hi + pad, 6);
  const xstep = xt.length > 1 ? xt[1] - xt[0] : 1;
  const X = linMap(xt[0], xt[xt.length - 1], ml, ml + plotW);
  const Y = linMap(0, 1, mt + plotH, mt);         // cumulative share, zero-baselined

  // horizontal gridlines (owid) at 0/25/50/75/100%; top tick carries the unit
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    svg.append(svgEl("line", { class: t === 0 ? "zero-line" : "gridline", x1: ml, y1: Y(t), x2: ml + plotW, y2: Y(t) }));
    svg.append(stext("axis-txt", ml - 8, Y(t) + 3, t === 1 ? "100%" : String(t * 100), { "text-anchor": "end" }));
  });
  xt.forEach((t) => svg.append(stext("axis-txt", X(t), mt + plotH + 16, srFmt(t, xstep), { "text-anchor": "middle" })));

  // ECDF path from the quantile function; extend flat to the baseline / top so it
  // reads as a full step curve (the 0.02/0.98 tails are the only fudge)
  const pathFor = (q) => {
    let d = `M${X(q[0]).toFixed(1)} ${Y(0).toFixed(1)} `;
    for (let j = 0; j < q.length; j++) d += `L${X(q[j]).toFixed(1)} ${Y(P[j]).toFixed(1)} `;
    d += `L${X(q[q.length - 1]).toFixed(1)} ${Y(1).toFixed(1)} `;
    return d;
  };

  // systems first (thin), then the reference on top (thick green)
  const colOf = srColorMap(p);
  visible.forEach((r) => {
    const col = colOf.get(r.name) || R_MID;
    const d = pathFor(r.q);
    mark(svg.appendChild(svgEl("path", { class: "trace", d, stroke: col, "stroke-width": 1.6 })), r.name, true);
    srHit(svg, d, r.name, r, col);
    // direct end-of-line label at the median (P≈0.5) crossing
    const mj = Math.round(P.length / 2);
    mark(svg.appendChild(stext("series-lab", X(r.q[mj]) + 5, Y(P[mj]) - 4, r.name, { fill: col })), r.name);
  });
  const dref = pathFor(p.ref.q);
  svg.append(svgEl("path", { class: "trace", d: dref, stroke: SR_GREEN, "stroke-width": 3 }));
  const rj = Math.round(P.length * 0.72);
  svg.append(stext("series-lab", X(p.ref.q[rj]) + 6, Y(P[rj]) + 12, "reference", { fill: SR_GREEN }));

  svg.append(stext("axis-lab", ml + plotW / 2, H - 6,
    "speaking rate (characters per second of trimmed speech)", { "text-anchor": "middle" }));

  chart.append(svg);
  renderSrLegend(p);
  srReapply();
}

// legend: every system as a drop/restore chip (colour = its draw colour)
function renderSrLegend(p) {
  const box = $("#sr-legend");
  box.innerHTML = "";
  // same full-series colour key as drawSrDist, so chips match the traces
  const colOf = srColorMap(p);
  p.series.forEach((r) => {
    const hidden = SR.hidden.has(r.name);
    const item = el("div", "leg-item" + (hidden ? " off" : ""));
    const sw = el("span", "swatch");
    const col = colOf.get(r.name) || R_MID;
    sw.style.background = col; sw.style.color = col;
    item.append(sw, el("span", "", r.name));
    item.onclick = () => {
      if (hidden) SR.hidden.delete(r.name); else SR.hidden.add(r.name);
      window.renderSpeaking();
    };
    if (!hidden) {
      item.addEventListener("mouseenter", () => setSrFocus(r.name));
      item.addEventListener("mouseleave", () => setSrFocus(SR.pin));
    }
    box.append(item);
  });
}

// redraw on resize while the speaking-rate view is active
bindChartResize("#sr-chart", "speaking", () => !!SR.data, () => window.renderSpeaking());
