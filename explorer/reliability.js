/* ============================================================
   ZTTS2 · Eval — reliability views (the tufte_* plates, live)
   Interactive SVG twins of scripts/tufte_p99 / _survival_overlay /
   _survival_smallmultiples / _slopegraph, in the Zyphra theme.
   Reads /api/reliability (build_reliability.py).
   Depends on app.js globals ($, $$, el, svgEl, apiGet)
   and the chart kit + R_* palette from
   chart_common.js.  Loaded after both.
   ============================================================ */
"use strict";

const BLOWUP = 40;        // catastrophic worst-case flag (tufte default)
const SLOPE_LIMIT = 12;   // slopegraph shows the N least-reliable (tufte --limit)

const REL = { data: null, ds: null, lang: null, type: "p99", hidden: new Set(),
  pin: null, focusNow: null };

// ---- interactivity (chart_common kit, parametrized for this view) -----------
const relTipKit = makeTip("#rel-chart", "#rel-tip");

function recSummary(r, col) {
  return `<span class="sw" style="background:${col || R_PRIMARY}"></span><b>${r.name}</b><br>` +
    `p99 <b>${r.p99}</b> edits · 95% CI [${r.ci[0]}, ${r.ci[1]}]<br>` +
    `<span class="meta">worst ${r.worst} · n ${r.n.toLocaleString()}` +
    `${r.langs && r.langs.length > 1 ? " · " + r.langs.length + " langs" : ""}</span>`;
}

// focus recolours + raises the primary paths (the owid line-chart idiom)
const _relFocus = makeFocus("#rel-chart", {
  dim: "0.1",
  emph: (e, on) => {
    if (on) {
      e.style.stroke = R_PRIMARY; e.style.strokeWidth = "2.8";
      if (e.parentNode) e.parentNode.appendChild(e);   // raise to front
    } else { e.style.stroke = ""; e.style.strokeWidth = ""; }
  },
});
function setRelFocus(name) { REL.focusNow = _relFocus(name); }

const relBind = makeBind(setRelFocus, relTipKit, REL, recSummary);
const hitPath = makeHit(relBind);

// re-assert focus after a fresh draw (keeps a pinned series highlighted)
function reapplyFocus() { if (REL.pin) setRelFocus(REL.pin); }

// ---------------------------------------------------------------------------
function ensureRel() {
  return ensureCache(REL, "/api/reliability",
    (d) => !!(d.datasets && d.datasets.length),
    (d) => {
      // open on the first dataset: the payload lists the TTS systems first
      if (!REL.ds || !d.datasets.some((x) => x.key === REL.ds))
        REL.ds = (d.datasets[0] || {}).key;
    });
}

function curDataset() {
  return REL.data.datasets.find((d) => d.key === REL.ds) || REL.data.datasets[0];
}

// The series to draw: a model dataset restricted to REL.lang (per-language
// breakdown), else the dataset's aggregate. Returns a shallow "view" of the
// dataset with series + counts recomputed, so every plate redraws unchanged.
function viewFor(ds) {
  let series = ds.series;
  if (ds.kind === "model" && REL.lang && ds.bylang && ds.bylang[REL.lang])
    series = ds.bylang[REL.lang];
  const langs = new Set();
  let utts = 0;
  series.forEach((r) => { (r.langs || []).forEach((l) => langs.add(l)); utts += r.n; });
  return Object.assign({}, ds, {
    series, n_series: series.length, n_langs: langs.size, n_utts: utts,
  });
}

window.renderReliability = async function () {
  const chart = $("#rel-chart");
  const ok = await ensureRel();
  if (!ok) {
    chart.querySelectorAll("svg, .rel-grid").forEach((n) => n.remove());
    chart.style.height = "auto";
    if (!chart.querySelector(".loading"))
      chart.append(el("div", "loading", "no reliability cache on disk — run build_reliability.py"));
    $("#rel-legend").innerHTML = ""; $("#rel-datasets").innerHTML = "";
    $("#rel-source").innerHTML = "";
    $("#ro-channel").textContent = "reliability / —";
    return;
  }
  chart.querySelectorAll(".loading").forEach((n) => n.remove());
  const ds = curDataset();
  if (REL.lang && !(ds.bylang && ds.bylang[REL.lang])) REL.lang = null;   // stale pick
  const view = viewFor(ds);
  $("#ro-channel").textContent = `reliability / ${ds.key}${REL.lang ? " / " + REL.lang : ""}`;
  renderRelDatasets();
  renderRelLangs(ds);
  bindRelTypeSeg();
  updateRelHead(view);
  drawRel(view);
  const src = $("#rel-source");
  const basket = REL.lang ? `${REL.lang} only` : "a fixed language basket";
  const basis = ds.kind === "model"
    ? `per-system re-transcribed WER on ${basket} (ground-truth floor shown), exact edit distance reconstructed per utterance`
    : "ground-truth ASR baseline, exact edit distance reconstructed per utterance";
  if (src) src.innerHTML =
    `<span><b>Data:</b> ZTTS2-Eval · ${ds.label} · ${basis}</span>` +
    `<span class="lic">CC BY</span>`;
};

function renderRelDatasets() {
  const box = $("#rel-datasets");
  box.innerHTML = "";
  REL.data.datasets.forEach((d) => {
    const c = el("button", "lang-chip" + (d.key === REL.ds ? " on" : ""), d.label);
    c.onclick = () => {
      REL.ds = d.key; REL.lang = null; REL.hidden = new Set(); REL.pin = null;
      window.renderReliability();
    };
    box.append(c);
  });
}

// Per-language picker — only for model datasets carrying a breakdown. "All
// languages" is the aggregate; each chip restricts every plate to one language.
function renderRelLangs(ds) {
  const box = $("#rel-langs");
  if (!box) return;
  const langs = (ds.kind === "model" && ds.langs_list) ? ds.langs_list : [];
  if (langs.length < 2) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  box.innerHTML = "";
  const chip = (key, label) => {
    const on = (REL.lang || "all") === key;
    const c = el("button", "lang-chip" + (on ? " on" : ""), label);
    c.onclick = () => {
      REL.lang = key === "all" ? null : key; REL.pin = null;
      window.renderReliability();
    };
    box.append(c);
  };
  chip("all", "All languages");
  langs.forEach((l) => chip(l, l));
}

let _relSegBound = false;
function bindRelTypeSeg() {
  $$("#rel-type-seg button").forEach((b) => b.classList.toggle("on", b.dataset.rel === REL.type));
  if (_relSegBound) return;
  _relSegBound = true;
  $$("#rel-type-seg button").forEach((b) => {
    b.onclick = () => {
      REL.type = b.dataset.rel;
      $$("#rel-type-seg button").forEach((x) => x.classList.toggle("on", x === b));
      window.renderReliability();
    };
  });
}

function updateRelHead(ds) {
  const unit = ds.kind === "model" ? "system" : "series";
  const stat = `${ds.n_series} ${unit}${ds.n_series !== 1 ? "s" : ""} · ${ds.n_langs} language${ds.n_langs !== 1 ? "s" : ""} · ${ds.n_utts.toLocaleString()} utterances · exact edit distance`;
  const H = {
    p99: ["Reliability at the 99th percentile",
      `Dot = p99 edit distance, line = bootstrap 95% CI. ${stat}. Overlapping intervals aren't distinguishable; grey rows rest on fewer than ${SMALL_N} utterances.`],
    survival: [`How reliable is each ${unit}? — the tail is the whole story`,
      `P(edits ≥ k) on log–log axes; lower curve = more reliable. Best / least-reliable / most-catastrophic highlighted — click a legend chip to drop or restore a ${unit}. ${stat}.`],
    multiples: [`How reliable is each ${unit}?`,
      `Each panel: one ${unit} (black) against all others (faint), ordered most → least reliable. Orange dot = a catastrophic worst case (≥ ${BLOWUP} edits). ${stat}.`],
    slope: ["The ranking, column by column",
      `% of utterances with at least k edits, at ${REL.data.ks.length} thresholds. Vertical order = ranking; crossings = rank changes. The ${Math.min(SLOPE_LIMIT, ds.n_series)} least-reliable ${unit}s shown.`],
  };
  const [t, n] = H[REL.type] || H.p99;
  $("#rel-title").textContent = t;
  $("#rel-note").textContent = n;
}

// pick the emphasised series (tufte auto_highlight): best p99, worst p99,
// most-catastrophic worst — among the given pool. Returns Map name->color.
function autoHighlight(series) {
  if (!series.length) return new Map();
  const byP99lo = series.slice().sort((a, b) => a.p99 - b.p99);
  const byP99hi = series.slice().sort((a, b) => b.p99 - a.p99);
  const byWorst = series.slice().sort((a, b) => b.worst - a.worst);
  const m = new Map();
  const add = (rec, col) => { if (rec && !m.has(rec.name)) m.set(rec.name, col); };
  add(byP99lo[0], R_BLUE);   // most reliable
  add(byP99hi[0], R_RED);    // least reliable
  add(byWorst[0], R_GOLD);   // worst blow-up
  return m;
}

function clearChart() {
  const chart = $("#rel-chart");
  chart.querySelectorAll("svg, .rel-grid").forEach((n) => n.remove());
  $("#rel-tip").hidden = true;
  chart.style.overflow = "visible";
  return chart;
}

function drawRel(ds) {
  if (REL.type === "p99") return drawP99(ds);
  if (REL.type === "survival") return drawSurvival(ds);
  if (REL.type === "multiples") return drawMultiples(ds);
  if (REL.type === "slope") return drawSlope(ds);
}

// ---------------------------------------------------------------------------
// p99 · CI  (horizontal dot + bootstrap CI, sorted by p99)
// ---------------------------------------------------------------------------
function drawP99(ds) {
  const chart = clearChart();
  $("#rel-legend").innerHTML = "";
  const rows = ds.series.slice().sort((a, b) => a.p99 - b.p99);

  const W = Math.max(360, chart.clientWidth);
  const ml = Math.min(190, Math.max(56, labelWidth(chart, "series-lab", rows.map((r) => r.name)) + 16));
  const mr = 66, mt = 14, mb = 44, rowH = 22;
  const plotW = W - ml - mr;
  const H = mt + rows.length * rowH + mb;

  const xmax = Math.max(1, ...rows.map((r) => r.ci[1]));
  const step = niceStep(xmax);
  const tickMax = Math.ceil(xmax / step) * step;
  const ticks = [];
  for (let t = 0; t <= tickMax; t += step) ticks.push(t);
  const x = linMap(0, tickMax, ml, ml + plotW);
  const gap = 0.01 * tickMax;                      // white gap either side of the dot

  const svg = newSvg(W, H);

  // faint vertical gridlines at ticks (owid: no frame; value=0 is the baseline)
  const y0 = mt, y1 = mt + rows.length * rowH;
  ticks.forEach((t) => {
    svg.append(svgEl("line", { class: t === 0 ? "zero-line" : "gridline", x1: x(t), y1: y0, x2: x(t), y2: y1 }));
    svg.append(stext("axis-txt", x(t), y1 + 15, String(t), { "text-anchor": "middle" }));
  });
  svg.append(stext("axis-lab", ml + plotW / 2, H - 6,
    "p99 edit distance — 99% of utterances are within this many edits (fewer = more reliable)",
    { "text-anchor": "middle" }));

  rows.forEach((r, idx) => {
    const yi = mt + idx * rowH + rowH / 2;
    const small = r.n < SMALL_N;
    const c = small ? R_MUT : R_PRIMARY;
    const [lo, hi] = r.ci;
    if (lo < r.p99 - gap)
      mark(svg.appendChild(svgEl("line", { x1: x(lo), y1: yi, x2: x(r.p99 - gap), y2: yi, stroke: R_RULE, "stroke-width": 1.4 })), r.name);
    if (hi > r.p99 + gap)
      mark(svg.appendChild(svgEl("line", { x1: x(r.p99 + gap), y1: yi, x2: x(hi), y2: yi, stroke: R_RULE, "stroke-width": 1.4 })), r.name);
    mark(svg.appendChild(svgEl("circle", { class: "dot", cx: x(r.p99), cy: yi, r: 3.6, fill: c })), r.name);
    mark(svg.appendChild(stext("series-lab", ml - 8, yi, r.name, { "text-anchor": "end", "dominant-baseline": "middle", fill: R_INK })), r.name);
    mark(svg.appendChild(stext("axis-txt", x(hi) + 8, yi, small ? `${r.p99}  n=${r.n}` : `${r.p99}`,
      { "text-anchor": "start", "dominant-baseline": "middle", fill: R_MUT })), r.name);
    // full-width transparent hit band so the whole row responds
    const band = svgEl("rect", { x: ml, y: yi - rowH / 2, width: plotW, height: rowH, fill: "transparent" });
    relBind(band, r.name, r);
    svg.append(band);
  });

  chart.append(svg);
  reapplyFocus();
}

// ---------------------------------------------------------------------------
// survival overlay  (log-log P(edits ≥ k), bundle + highlighted)
// ---------------------------------------------------------------------------
function drawSurvival(ds) {
  const chart = clearChart();
  const visible = ds.series.filter((r) => !REL.hidden.has(r.name));
  const emph = autoHighlight(visible);

  const W = Math.max(360, chart.clientWidth);
  const H = 470, ml = 56, mr = 130, mt = 16, mb = 48;
  const plotW = W - ml - mr, plotH = H - mt - mb;
  const svg = newSvg(W, H);

  if (!visible.length) {
    svg.append(stext("axis-lab", W / 2, H / 2, "all series hidden — restore one from the legend", { "text-anchor": "middle" }));
    chart.append(svg); renderRelLegendToggle(ds, emph); return;
  }

  const xmax = Math.max(2, ...visible.map((r) => r.k[r.k.length - 1] || 1));
  let ymin = Math.min(...visible.map((r) => Math.min(...r.s)));
  if (!isFinite(ymin) || ymin <= 0) ymin = 0.01;
  const x0 = 0.9, x1 = xmax * 1.15, y0 = ymin * 0.7, y1 = 120;
  const X = logMap(x0, x1, ml, ml + plotW);
  const Y = logMap(y0, y1, mt + plotH, mt);
  const xt = [1, 3, 10, 30, 100, 300, 1000].filter((t) => t >= x0 && t <= x1);
  const yt = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100].filter((t) => t >= y0 && t <= y1);

  // faint horizontal gridlines (owid) + y labels; the top tick carries the unit
  const ytop = yt[yt.length - 1];
  yt.forEach((t) => {
    svg.append(svgEl("line", { class: "gridline", x1: ml, y1: Y(t), x2: ml + plotW, y2: Y(t) }));
    svg.append(stext("axis-txt", ml - 8, Y(t) + 3, t === ytop ? t + "%" : "" + t, { "text-anchor": "end" }));
  });
  xt.forEach((t) => svg.append(stext("axis-txt", X(t), mt + plotH + 16, "" + t, { "text-anchor": "middle" })));

  const pathFor = (r) => {
    let d = "";
    for (let j = 0; j < r.k.length; j++) {
      const v = r.s[j]; if (!(v > 0)) continue;
      d += `${d ? "L" : "M"}${X(r.k[j]).toFixed(1)} ${Y(v).toFixed(1)} `;
    }
    return d;
  };
  visible.forEach((r) => {
    if (emph.has(r.name)) return;
    const d = pathFor(r); if (!d) return;
    mark(svg.appendChild(svgEl("path", { class: "trace", d, stroke: R_BUNDLE, "stroke-width": 1.1 })), r.name, true);
    hitPath(svg, d, r.name, r, R_MID);
  });

  let ei = 0;
  emph.forEach((col, name) => {
    const r = visible.find((s) => s.name === name); if (!r) return;
    const d = pathFor(r);
    if (d) { mark(svg.appendChild(svgEl("path", { class: "trace", d, stroke: col })), r.name, true); hitPath(svg, d, r.name, r, col); }
    const lastJ = r.k.length - 1;
    mark(svg.appendChild(svgEl("circle", { class: "dot", cx: X(r.k[lastJ]), cy: Y(r.s[lastJ]), r: 3.4, fill: col })), r.name);
    const j = Math.round((0.3 + 0.2 * ei) * (r.k.length - 1));
    mark(svg.appendChild(stext("series-lab", X(r.k[j]) + 5, Y(r.s[j]) - 6, r.name, { fill: col })), r.name);
    ei++;
  });

  // owid: no frame, no rotated y-title — the y unit rides the top tick and the
  // metric (P(edits ≥ k)) is named in the subtitle above the chart
  svg.append(stext("axis-lab", ml + plotW / 2, H - 6, "edit distance  k   (edits in one utterance)", { "text-anchor": "middle" }));

  chart.append(svg);
  renderRelLegendToggle(ds, emph);
  reapplyFocus();
}

// legend: every series as a toggle chip; swatch shows its current draw colour
function renderRelLegendToggle(ds, emph) {
  const box = $("#rel-legend");
  box.innerHTML = "";
  ds.series.forEach((r) => {
    const hidden = REL.hidden.has(r.name);
    const col = emph.get(r.name) || R_MID;
    const item = el("div", "leg-item" + (hidden ? " off" : ""));
    const sw = el("span", "swatch");
    sw.style.background = col; sw.style.color = col;
    item.append(sw, el("span", "", r.name));
    item.onclick = () => {
      if (hidden) REL.hidden.delete(r.name); else REL.hidden.add(r.name);
      window.renderReliability();
    };
    if (!hidden) {                                   // hover a chip → focus its curve
      item.addEventListener("mouseenter", () => setRelFocus(r.name));
      item.addEventListener("mouseleave", () => setRelFocus(REL.pin));
    }
    box.append(item);
  });
}

// ---------------------------------------------------------------------------
// small multiples  (one log-log panel per series, shared scale)
// ---------------------------------------------------------------------------
function drawMultiples(ds) {
  const chart = clearChart();
  $("#rel-legend").innerHTML = "";
  chart.style.height = "auto";
  // order most → least reliable by P(edits ≥ 3); slope index of k=3
  const k3 = REL.data.ks.indexOf(3) >= 0 ? REL.data.ks.indexOf(3) : 1;
  const models = ds.series.slice().sort((a, b) => a.slope[k3] - b.slope[k3]);

  const VW = 240, VH = 196, ml = 30, mr = 8, mt = 30, mb = 24;
  const plotW = VW - ml - mr, plotH = VH - mt - mb;
  const xmax = Math.max(2, ...ds.series.map((r) => r.worst || 1));
  let ymin = Math.min(...ds.series.map((r) => Math.min(...r.s)));
  if (!isFinite(ymin) || ymin <= 0) ymin = 0.01;
  const x0 = 0.9, x1 = xmax * 1.1, y0 = ymin * 0.7, y1 = 120;
  const X = logMap(x0, x1, ml, ml + plotW);
  const Y = logMap(y0, y1, mt + plotH, mt);
  const xt = [1, 10, 100, 1000].filter((t) => t >= x0 && t <= x1);
  const yt = [0.01, 0.1, 1, 10, 100].filter((t) => t >= y0 && t <= y1);

  const pathFor = (r) => {
    let d = "";
    for (let j = 0; j < r.k.length; j++) { const v = r.s[j]; if (!(v > 0)) continue; d += `${d ? "L" : "M"}${X(r.k[j]).toFixed(1)} ${Y(v).toFixed(1)} `; }
    return d;
  };
  const bundle = ds.series.map(pathFor).filter(Boolean);

  const grid = el("div", "rel-grid");
  models.forEach((r) => {
    const svg = svgEl("svg", { viewBox: `0 0 ${VW} ${VH}`, preserveAspectRatio: "xMidYMid meet" });
    svg.style.width = "100%"; svg.style.height = "auto";
    yt.forEach((t) => {
      svg.append(svgEl("line", { class: "gridline", x1: ml, y1: Y(t), x2: ml + plotW, y2: Y(t) }));
      svg.append(stext("axis-txt", ml - 4, Y(t) + 2.5, "" + t, { "text-anchor": "end", "font-size": "7" }));
    });
    xt.forEach((t) => svg.append(stext("axis-txt", X(t), mt + plotH + 11, "" + t, { "text-anchor": "middle", "font-size": "7" })));
    bundle.forEach((d) => svg.append(svgEl("path", { class: "trace", d, stroke: R_FAINT, "stroke-width": 1 })));
    const dm = pathFor(r);
    if (dm) mark(svg.appendChild(svgEl("path", { class: "trace", d: dm, stroke: R_PRIMARY, "stroke-width": 1.9 })), r.name);
    if (r.worst >= BLOWUP) {
      const lastJ = r.k.length - 1;
      mark(svg.appendChild(svgEl("circle", { cx: X(r.k[lastJ]), cy: Y(r.s[lastJ]), r: 2.6, fill: R_RED })), r.name);
    }
    // owid: gridlines carry the panel, no enclosing frame
    svg.append(stext("series-lab", ml - 2, 13, r.name, { fill: R_INK, "font-size": "11" }));
    svg.append(stext("axis-txt", ml - 2, 24, `p99 ${r.p99} · worst ${r.worst} · n ${r.n}`,
      { "font-size": "8", fill: r.worst >= BLOWUP ? R_RED : R_MUT }));
    // transparent hit rect: hover focuses this panel's series across the grid +
    // lifts the panel; click drills into the survival overlay pinned to it
    const hit = svgEl("rect", { x: 0, y: 0, width: VW, height: VH, fill: "transparent" });
    hit.style.cursor = "pointer";
    hit.addEventListener("mouseenter", () => { setRelFocus(r.name); svg.style.outline = `1px solid ${R_RULE}`; });
    hit.addEventListener("mouseleave", () => { setRelFocus(REL.pin); svg.style.outline = ""; });
    hit.addEventListener("click", () => {
      REL.pin = r.name; REL.hidden.delete(r.name); REL.type = "survival";
      window.renderReliability();
    });
    svg.append(hit);
    grid.append(svg);
  });
  chart.append(grid);
  reapplyFocus();
}

// ---------------------------------------------------------------------------
// slopegraph  (P(edits ≥ k) at thresholds; ranking + crossings)
// ---------------------------------------------------------------------------
function nudge(vals, gap) {
  const idx = vals.map((_, i) => i).sort((a, b) => vals[a] - vals[b]);
  const pos = vals.slice();
  let prev = -Infinity;
  for (const i of idx) { pos[i] = Math.max(pos[i], prev + gap); prev = pos[i]; }
  const mp = pos.reduce((a, b) => a + b, 0) / pos.length;
  const mv = vals.reduce((a, b) => a + b, 0) / vals.length;
  return pos.map((p) => p - (mp - mv));
}

function drawSlope(ds) {
  const chart = clearChart();
  $("#rel-legend").innerHTML = "";
  const KS = REL.data.ks;
  // the N least-reliable by p99 (tufte --limit), then layout order by 2nd column
  let pool = ds.series.slice().sort((a, b) => b.p99 - a.p99).slice(0, SLOPE_LIMIT);
  const emph = autoHighlight(pool);
  const colFor = (r) => emph.get(r.name) || R_MID;
  const orderCol = Math.min(1, KS.length - 1);
  const models = pool.slice().sort((a, b) => a.slope[orderCol] - b.slope[orderCol]);

  const W = Math.max(360, chart.clientWidth);
  const H = Math.max(460, models.length * 24 + 130);
  const fmt = (v) => (v >= 1 ? v.toFixed(1) : v.toFixed(2));
  // values sit centred on the end columns, so names go outside their half-width
  const vh = labelWidth(chart, "axis-txt", models.flatMap((r) => r.slope.map(fmt))) / 2;
  const lw = labelWidth(chart, "series-lab", models.map((r) => r.name)) + vh + 16;   // names both ends
  const ml = Math.max(80, lw), mr = Math.max(80, lw), mt = 60, mb = 24;
  const plotW = W - ml - mr, plotH = H - mt - mb;
  const svg = newSvg(W, H);

  const floor = Math.min(...models.flatMap((r) => r.slope.filter((v) => v > 0))) || 0.01;
  const Lval = (v) => log10(Math.max(v, floor * 0.5));
  // nudge each column independently, keep exact order
  const cols = KS.map((_, j) => nudge(models.map((r) => Lval(r.slope[j])), 0.085));
  let tmin = Infinity, tmax = -Infinity;
  cols.forEach((c) => c.forEach((v) => { if (v < tmin) tmin = v; if (v > tmax) tmax = v; }));
  const cx = (j) => ml + (KS.length === 1 ? plotW / 2 : (j / (KS.length - 1)) * plotW);
  const cy = linMap(tmin, tmax, mt + plotH, mt);

  // column headers
  KS.forEach((k, j) => svg.append(stext("axis-lab", cx(j), mt - 18, `≥ ${k} edit${k > 1 ? "s" : ""}`, { "text-anchor": "middle", fill: R_INK })));

  models.forEach((r, i) => {
    const col = colFor(r);
    const acc = emph.has(r.name);
    let d = "";
    KS.forEach((_, j) => { d += `${j ? "L" : "M"}${cx(j).toFixed(1)} ${cy(cols[j][i]).toFixed(1)} `; });
    mark(svg.appendChild(svgEl("path", { class: "trace", d, stroke: col, "stroke-width": acc ? 2 : 1, opacity: acc ? 1 : 0.85 })), r.name, true);
    // value nodes with a white halo for legibility
    KS.forEach((_, j) => {
      const s = fmt(r.slope[j]);
      const gx = cx(j), gy = cy(cols[j][i]);
      mark(svg.appendChild(svgEl("rect", { x: gx - vh - 2, y: gy - 6.5, width: 2 * vh + 4, height: 13, fill: "#f4f4f2", opacity: 0.85 })), r.name);
      mark(svg.appendChild(stext("axis-txt", gx, gy + 3, s, { "text-anchor": "middle", fill: col })), r.name);
    });
    // series names both ends
    mark(svg.appendChild(stext("series-lab", ml - vh - 8, cy(cols[0][i]), r.name, { "text-anchor": "end", "dominant-baseline": "middle", fill: col })), r.name);
    mark(svg.appendChild(stext("series-lab", ml + plotW + vh + 8, cy(cols[KS.length - 1][i]), r.name, { "text-anchor": "start", "dominant-baseline": "middle", fill: col })), r.name);
    hitPath(svg, d, r.name, r, col);
  });

  chart.append(svg);
  reapplyFocus();
}

// redraw on resize while the reliability view is active
bindChartResize("#rel-chart", "reliability", () => !!REL.data, () => window.renderReliability());
