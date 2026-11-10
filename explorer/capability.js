/* ============================================================
   ZTTS2 · Eval — index view (ECI-style composite scores)
   One row per system, ranked by the index, with its 90% bootstrap
   interval; the capability index or the prosody index (the same fit
   over prosody items only). Reads /api/capability_index (built by
   scripts/results_doc/build_capability_index.py; the prosody index
   is its `prosody` member, same shape).
   Depends on app.js globals ($, el, svgEl, apiGet, STATE, cleanName,
   labelWidth) and the chart kit (stext, newSvg, linMap, mark,
   makeFocus, makeTip, makeBind, ensureCache, bindChartResize, R_*)
   from chart_common.js (loaded first).
   ============================================================ */
"use strict";

// index: "capability" | "prosody" · mode: "rank" | "time" | "params"
const CI_VIEW = { data: null, pin: null, index: "capability", mode: "rank" };

function ensureIndex() {
  return ensureCache(CI_VIEW, "/api/capability_index", (d) => !!(d.systems && d.systems.length));
}

const ciHasProsody = () => !!(CI_VIEW.data.prosody && CI_VIEW.data.prosody.systems && CI_VIEW.data.prosody.systems.length);
// the payload of the selected index: {method, systems}
const ciIndex = () => (CI_VIEW.index === "prosody" ? CI_VIEW.data.prosody : CI_VIEW.data);

const ciName = (s) => (s.name === "real speech" ? s.name : cleanName(s.name));
const ciLangs = (s, m) => `${s.langs["clean"]}/${m.n_langs["clean"]} Clean · ${s.langs["in-the-wild"]}/${m.n_langs["in-the-wild"]} ITW`;
const ciPartial = (s, m) => s.langs["clean"] < m.n_langs["clean"] || s.langs["in-the-wild"] < m.n_langs["in-the-wild"];

const ciZyphra = (s) => s.name.startsWith("zonos");
const ciDate = (s) => Date.parse(s.release + "T00:00:00Z");
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const ciDay = (t) => { const d = new Date(t); return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`; };

function ciSummary(s) {
  const m = ciIndex().method;
  const band = s.anchor ? "anchor — fixed by definition" : `90% CI [${s.ci[0]}, ${s.ci[1]}]`;
  const rel = s.release
    ? `<br><span class="meta">released ${ciDay(ciDate(s))} — ${s.release_source}</span>` : "";
  const p = s.params;
  const size = p
    ? `<br><span class="meta">${fmtParams(p.total)} parameters${p.active ? ` (MoE, ${fmtParams(p.active)} active)` : ""} — ${p.source}</span>` : "";
  return `<b>${ciName(s)}</b><br>index <b>${s.index}</b> · ${band}<br>` +
    `<span class="meta">${ciLangs(s, m)} languages · ${s.n_items} scores</span>${rel}${size}` +
    (s.breakdown && CI_VIEW.pin !== s.name ? `<br><span class="meta">click for what decides the index</span>` : "");
}

const ciFocus = makeFocus("#ci-chart", { dim: "0.15", after: () => drawBreakdown() });
const ciTip = makeTip("#ci-chart", "#ci-tip");
const ciBind = makeBind(ciFocus, ciTip, CI_VIEW, ciSummary);

// ---- breakdown of the pinned system's index -------------------------------------
// Per group (dimension → metric, then test set), from build_capability_index.breakdown:
// `level` is where the group's scores alone would put the system, `weight` the group's
// share of the information the fit has about it, `points` = weight × level; the
// points add up to the index.
const CI_SET_LABEL = { clean: "Clean", "in-the-wild": "In-The-Wild" };
const ciMetricLabel = (k) => (k === "charsec_x" ? "char/s ×ceil"
  : (METRICS.find((m) => m.key === k) || { label: k }).label);
const ciLevel = (v) => (v == null ? "—" : Math.abs(v) >= 1000 ? (v > 0 ? "> 999" : "< −999") : v.toFixed(1));
const ciPct = (w) => (w > 0 && w < 0.01 ? "<1%" : `${Math.round(w * 100)}%`);
let _ciBreakdownKey = null;

function drawBreakdown() {
  const box = $("#ci-breakdown");
  const s = CI_VIEW.pin && CI_VIEW.data && ciIndex().systems.find((x) => x.name === CI_VIEW.pin);
  const key = s && s.breakdown ? `${CI_VIEW.index}|${s.name}` : null;
  if (key === _ciBreakdownKey) return;
  _ciBreakdownKey = key;
  if (!key) { box.hidden = true; box.innerHTML = ""; return; }

  const row = (cls, label, p) =>
    `<tr class="${cls}"><td>${label}</td><td>${p.n}</td>` +
    `<td class="wt">${ciPct(p.weight)}<span class="wbar"><i style="width:${(p.weight * 100).toFixed(1)}%"></i></span></td>` +
    `<td>${ciLevel(p.level)}</td><td>${p.points.toFixed(1)}</td></tr>`;
  const b = s.breakdown;
  const body = b.dimensions.map((g) =>
    row("grp", g.key.charAt(0).toUpperCase() + g.key.slice(1), g) +
    (g.metrics.length > 1 ? g.metrics.map((mm) => row("sub", ciMetricLabel(mm.key), mm)).join("") : "")).join("") +
    `<tr class="sep"><td colspan="5"></td></tr>` +
    b.sets.map((g) => row("grp", CI_SET_LABEL[g.key] || g.key, g)).join("");
  box.innerHTML =
    `<div class="bd-head"><h3>${ciName(s)} <span class="dim">· ${CI_VIEW.index} index ${s.index.toFixed(1)}</span></h3>` +
    `<button class="bd-close" title="unpin">×</button></div>` +
    `<p><b>Level</b>: where that group's scores alone would put the system on this scale. ` +
    `<b>Weight</b>: how much of the placement the group decides — its share of the information the ` +
    `fit has about this system; items near the system's level and with steep slopes count most, so ` +
    `it is not the equal per-dimension weighting of the residuals. <b>Points</b> = weight × level; ` +
    `the dimensions add up to the index, and so do the two test sets.</p>` +
    `<table><thead><tr><th>group</th><th>scores</th><th>weight</th><th>level</th><th>points</th></tr></thead>` +
    `<tbody>${body}<tr class="total"><td>index</td><td>${s.n_items}</td><td class="wt">100%</td>` +
    `<td></td><td>${s.index.toFixed(1)}</td></tr></tbody></table>`;
  box.hidden = false;
  box.querySelector(".bd-close").onclick = () => { CI_VIEW.pin = null; ciTip.hide(); ciFocus(null); };
}

let _ciSegBound = false;
function bindIndexSegs() {
  $$("#ci-index-seg button").forEach((b) => b.classList.toggle("on", b.dataset.index === CI_VIEW.index));
  $$("#ci-mode-seg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === CI_VIEW.mode));
  if (_ciSegBound) return;
  _ciSegBound = true;
  $$("#ci-index-seg button").forEach((b) => {
    b.onclick = () => { CI_VIEW.index = b.dataset.index; window.renderIndex(); };
  });
  $$("#ci-mode-seg button").forEach((b) => {
    b.onclick = () => { CI_VIEW.mode = b.dataset.mode; window.renderIndex(); };
  });
}

window.renderIndex = async function () {
  bindIndexSegs();
  const chart = $("#ci-chart");
  const ok = await ensureIndex();
  if (!ok) {
    chart.querySelectorAll("svg").forEach((n) => n.remove());
    if (!chart.querySelector(".loading"))
      chart.append(el("div", "loading", "no index cache — run scripts/results_doc/build_capability_index.py"));
    $("#ci-source").innerHTML = ""; $("#ro-channel").textContent = `${CI_VIEW.index} index / —`;
    return;
  }
  chart.querySelectorAll(".loading").forEach((n) => n.remove());
  // a payload built before the prosody index existed: keep the button, say why it is off
  const pb = $('#ci-index-seg button[data-index="prosody"]');
  pb.disabled = !ciHasProsody();
  pb.title = pb.disabled ? "this payload has no prosody index — rerun build_capability_index.py" : "";
  if (pb.disabled && CI_VIEW.index === "prosody") { CI_VIEW.index = "capability"; bindIndexSegs(); }

  const d = ciIndex(), m = d.method, prosody = CI_VIEW.index === "prosody";
  $("#ci-title").textContent = prosody ? "Prosody index" : "Capability index";
  $("#ro-channel").textContent = `${CI_VIEW.index} index / ${d.systems.length} systems`;
  const [hiA, loA] = Object.entries(m.anchors);
  const anchors = `Anchors: ${hiA[0]} = ${hiA[1]}, ${cleanName(loA[0])} = ${loA[1]}. `;
  const head = prosody
    ? `One prosody score per system, fit like the capability index but over ${m.n_items} prosody ` +
      `items only. Pace (rhythm pace, char/s, paired duration), pauses (rhythm pause, paired pause), ` +
      `pitch (pitch range, paired pitch range) and contour (MPM span) weigh equally; the paired ` +
      `metrics exist on In-The-Wild only, MPM on Clean only. ` + anchors
    : `One score per system, fit like Epoch's Capabilities Index: item response theory over ` +
      `${m.n_items} items (metric × test set × language), each with its own difficulty and slope, ` +
      `so systems scored on different language sets share one scale. Intelligibility, voice, ` +
      `quality and prosody weigh equally. ` + anchors;
  const tail = {
    time: `Each system sits at its release date (source in the tooltip); the line is the frontier, ` +
      `the best system scored on every language released by then. Vertical line = 90% bootstrap interval.`,
    params: `Each system sits at its published model size (the LM / backbone the developer names; ` +
      `codecs excluded; source in the tooltip), log scale. The line is the size frontier: the best ` +
      `system scored on every language at or below that size. Mixture-of-experts models (ZONOS2, ` +
      `ZONOS3 Mini) sit at their total parameters; the active count is in the tooltip. ` +
      `Vertical line = 90% bootstrap interval.`,
    rank: `Line = 90% bootstrap interval (an arrowhead = it runs past the axis; hover for the full range); ` +
      `grey = not scored on every language.`,
  };
  $("#ci-note").textContent = head + tail[CI_VIEW.mode];
  $("#ci-source").innerHTML =
    `<span><b>Data:</b> ZTTS2-Eval · Clean (FLEURS-R, ${m.n_langs["clean"]} languages) + ` +
    `In-The-Wild (VoxBlink2, ${m.n_langs["in-the-wild"]} languages) · ${m.n_obs.toLocaleString()} scores · ` +
    `method: <a href="https://epoch.ai/data/eci-documentation/methodology" target="_blank" rel="noopener">Epoch Capabilities Index</a> · ` +
    `${m.bootstrap} bootstrap refits</span><span class="lic">CC BY</span>`;
  drawIndexView(d);
};

function drawIndexView(d) {
  if (CI_VIEW.mode === "rank") drawIndex(d); else drawIndexScatter(d, CI_AXES[CI_VIEW.mode]);
}

function drawIndex(d) {
  const chart = $("#ci-chart"), m = d.method;
  chart.querySelectorAll("svg").forEach((n) => n.remove());
  $("#ci-legend").innerHTML = "";
  $("#ci-tip").hidden = true;
  const rows = d.systems;
  const tag = (s) => `${s.index.toFixed(1)}${s.anchor ? "  anchor" : ""}`;

  const ml = Math.min(260, Math.max(80, labelWidth(chart, "series-lab", rows.map(ciName)) + 16));
  const mr = Math.max(60, labelWidth(chart, "axis-txt", rows.map(tag)) + 16);
  const W = chartWidth(chart, 420, ml + mr);
  const mt = 30, mb = 46, rowH = 24;
  const plotW = W - ml - mr;
  const H = mt + rows.length * rowH + mb;

  // the axis spans the point estimates, not the intervals: a system scored on one or
  // two languages has an interval wide enough to squeeze everyone else into a sliver,
  // so intervals past the axis are cut at the edge and marked with an arrowhead
  const step = 25;
  const t0 = Math.floor(Math.min(0, ...rows.map((s) => s.index)) / step) * step;
  const t1 = Math.ceil(Math.max(100, ...rows.map((s) => s.index)) / step) * step;
  const clamp = (v) => Math.min(t1, Math.max(t0, v));
  const lo = (s) => clamp(s.ci ? s.ci[0] : s.index), hi = (s) => clamp(s.ci ? s.ci[1] : s.index);
  const x = linMap(t0, t1, ml, ml + plotW);
  const gap = 0.006 * (t1 - t0);                  // white gap either side of the dot

  chart.style.overflow = "visible";               // the chart is as tall as it has rows
  const svg = newSvg(W, H);
  const y0 = mt, y1 = mt + rows.length * rowH;
  for (let t = t0; t <= t1; t += step) {
    svg.append(svgEl("line", { class: t === 0 ? "zero-line" : "gridline", x1: x(t), y1: y0, x2: x(t), y2: y1 }));
    svg.append(stext("axis-txt", x(t), y1 + 15, String(t), { "text-anchor": "middle" }));
  }
  // the two anchors: the scale is defined by them, so they get labelled rules
  Object.entries(m.anchors).forEach(([name, v]) => {
    svg.append(svgEl("line", { x1: x(v), y1: y0, x2: x(v), y2: y1, stroke: R_RED, "stroke-width": 1, "stroke-dasharray": "3 3" }));
    svg.append(stext("axis-txt", x(v), mt - 10, `${name === "real speech" ? name : cleanName(name)} = ${v}`, { "text-anchor": "middle", fill: R_RED }));
  });
  svg.append(stext("axis-lab", ml + plotW / 2, H - 6, `${CI_VIEW.index} index (higher = better)`, { "text-anchor": "middle" }));

  rows.forEach((s, i) => {
    const yi = mt + i * rowH + rowH / 2;
    const c = ciPartial(s, m) ? R_MID : R_PRIMARY;
    if (s.ci) {
      if (lo(s) < s.index - gap)
        mark(svg.appendChild(svgEl("line", { x1: x(lo(s)), y1: yi, x2: x(s.index - gap), y2: yi, stroke: R_RULE, "stroke-width": 1.4 })), s.name);
      if (hi(s) > s.index + gap)
        mark(svg.appendChild(svgEl("line", { x1: x(s.index + gap), y1: yi, x2: x(hi(s)), y2: yi, stroke: R_RULE, "stroke-width": 1.4 })), s.name);
      if (s.ci[0] < t0)
        mark(svg.appendChild(svgEl("path", { d: `M${x(t0)} ${yi} l6 -3.5 v7 Z`, fill: R_MID })), s.name);
      if (s.ci[1] > t1)
        mark(svg.appendChild(svgEl("path", { d: `M${x(t1)} ${yi} l-6 -3.5 v7 Z`, fill: R_MID })), s.name);
    }
    const cx = x(s.index);
    mark(svg.appendChild(s.anchor
      ? svgEl("path", { d: `M${cx} ${yi - 5} L${cx + 5} ${yi} L${cx} ${yi + 5} L${cx - 5} ${yi} Z`, fill: R_RED })
      : svgEl("circle", { class: "dot", cx, cy: yi, r: 3.6, fill: c })), s.name);
    mark(svg.appendChild(stext("series-lab", ml - 8, yi, ciName(s), { "text-anchor": "end", "dominant-baseline": "middle", fill: ciPartial(s, m) ? R_MUT : R_INK })), s.name);
    mark(svg.appendChild(stext("axis-txt", x(hi(s)) + 8, yi, tag(s), { "text-anchor": "start", "dominant-baseline": "middle", fill: s.anchor ? R_RED : R_MUT })), s.name);
    const band = svgEl("rect", { x: ml, y: yi - rowH / 2, width: plotW, height: rowH, fill: "transparent" });
    ciBind(band, s.name, s);
    svg.append(band);
  });

  chart.append(svg);
  ciFocus(CI_VIEW.pin);
}

// x-axis specs of the scatter modes. `frontier` says who may set the frontier, which is
// the running best index in order of x (best released by then / best at or below that
// size); `scale` returns the x map plus [value, label] ticks for the plot's pixel span.
const fmtParams = (b) => (b >= 1 ? `${+b.toFixed(2)}B` : `${Math.round(b * 1000)}M`);
const CI_AXES = {
  time: {
    has: (s) => !!s.release,
    x: ciDate,
    frontier: (s, m) => !ciPartial(s, m),
    label: "public release date",
    scale(xs, r0, r1) {   // whole quarters around the release dates
      const a = new Date(Math.min(...xs)), b = new Date(Math.max(...xs));
      const q0 = Date.UTC(a.getUTCFullYear(), Math.floor(a.getUTCMonth() / 3) * 3, 1);
      const q1 = Date.UTC(b.getUTCFullYear(), Math.floor(b.getUTCMonth() / 3) * 3 + 3, 1);
      const ticks = [];
      for (let yy = a.getUTCFullYear(), mo = Math.floor(a.getUTCMonth() / 3) * 3; Date.UTC(yy, mo, 1) <= q1; mo += 3) {
        if (mo >= 12) { mo -= 12; yy += 1; }
        if (Date.UTC(yy, mo, 1) <= q1) ticks.push([Date.UTC(yy, mo, 1), `${MONTHS[mo]} ${yy}`]);
      }
      return { x: linMap(q0, q1, r0, r1), ticks };
    },
  },
  params: {
    has: (s) => !!s.params,
    x: (s) => s.params.total,
    frontier: (s, m) => !ciPartial(s, m),
    label: "parameters — published model size (log scale)",
    scale(xs, r0, r1) {   // log axis, ticks at 1 and 3 per decade
      const lo = Math.min(...xs) / 1.6, hi = Math.max(...xs) * 1.6;
      const ticks = [];
      for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
        [1, 3].forEach((k) => { const v = k * 10 ** e; if (v >= lo && v <= hi) ticks.push([v, fmtParams(v)]); });
      return { x: logMap(lo, hi, r0, r1), ticks };
    },
  },
};

// index against an x quantity (ECI's headline chart when x is the release date): one dot
// per system with its interval, and the frontier among systems scored on every language
function drawIndexScatter(d, ax) {
  const chart = $("#ci-chart"), m = d.method;
  chart.querySelectorAll("svg").forEach((n) => n.remove());
  $("#ci-tip").hidden = true;
  const pts = d.systems.filter(ax.has).sort((a, b) => ax.x(a) - ax.x(b) || b.index - a.index);
  const real = d.systems.find((s) => s.name === "real speech");

  const frontier = [];
  pts.filter((s) => ax.frontier(s, m)).forEach((s) => {
    if (!frontier.length || s.index > frontier[frontier.length - 1].index) frontier.push(s);
  });

  const ml = 48, mt = 24, mb = 46, H = 460;
  const mr = Math.max(60, labelWidth(chart, "series-lab", frontier.map(ciName)) + 18);
  const W = chartWidth(chart, 520, ml + mr);
  const plotW = W - ml - mr, plotH = H - mt - mb;
  const { x, ticks } = ax.scale(pts.map(ax.x), ml, ml + plotW);
  // label every k-th tick when neighbouring labels would collide (narrow screens);
  // gridlines stay on every tick
  const tickPx = Math.min(...ticks.slice(1).map(([t], i) => x(t) - x(ticks[i][0])));
  const every = tickStride(tickPx, labelWidth(chart, "axis-txt", ticks.map(([, lab]) => lab)));
  const step = 25;
  const v0 = Math.floor(Math.min(0, ...pts.map((s) => s.index)) / step) * step;
  const v1 = Math.ceil(Math.max(100, ...pts.map((s) => s.index)) / step) * step;
  const y = linMap(v0, v1, mt + plotH, mt);
  const clamp = (v) => Math.min(v1, Math.max(v0, v));

  chart.style.overflow = "visible";
  const svg = newSvg(W, H);
  for (let v = v0; v <= v1; v += step) {
    svg.append(svgEl("line", { class: v === 0 ? "zero-line" : "gridline", x1: ml, y1: y(v), x2: ml + plotW, y2: y(v) }));
    svg.append(stext("axis-txt", ml - 8, y(v), String(v), { "text-anchor": "end", "dominant-baseline": "middle" }));
  }
  ticks.forEach(([t, lab], i) => {
    svg.append(svgEl("line", { class: "gridline", x1: x(t), y1: mt, x2: x(t), y2: mt + plotH }));
    if (i % every === 0) svg.append(stext("axis-txt", x(t), mt + plotH + 15, lab, { "text-anchor": "middle" }));
  });
  svg.append(stext("axis-lab", ml + plotW / 2, H - 6, ax.label, { "text-anchor": "middle" }));
  if (real) {   // no date or size: the anchor is a level, not a point
    svg.append(svgEl("line", { x1: ml, y1: y(real.index), x2: ml + plotW, y2: y(real.index), stroke: R_RED, "stroke-width": 1, "stroke-dasharray": "3 3" }));
    svg.append(stext("axis-txt", ml + 4, y(real.index) - 6, `real speech = ${real.index}`, { fill: R_RED }));
  }

  // frontier as a step line, carried to the right edge
  let fd = "";
  frontier.forEach((s, i) => {
    fd += i ? `H${x(ax.x(s))}V${y(s.index)}` : `M${x(ax.x(s))} ${y(s.index)}`;
  });
  if (frontier.length) fd += `H${ml + plotW}`;
  svg.append(svgEl("path", { d: fd, fill: "none", stroke: R_PRIMARY, "stroke-width": 1.6, "stroke-opacity": 0.55 }));

  // systems sharing an x (variants of one checkpoint) fan out sideways
  const same = {};
  pts.forEach((s) => (same[ax.x(s)] ||= []).push(s));
  const pos = new Map(pts.map((s) => {
    const grp = same[ax.x(s)], k = grp.indexOf(s);
    return [s, { cx: x(ax.x(s)) + (k - (grp.length - 1) / 2) * 6, cy: y(s.index), half: ((grp.length - 1) / 2) * 6 }];
  }));
  pts.forEach((s) => {
    const { cx, cy } = pos.get(s);
    const col = ciZyphra(s) ? R_RED : ciPartial(s, m) ? R_MID : R_PRIMARY;
    if (s.ci)
      mark(svg.appendChild(svgEl("line", { x1: cx, y1: y(clamp(s.ci[0])), x2: cx, y2: y(clamp(s.ci[1])), stroke: col, "stroke-opacity": 0.3, "stroke-width": 1.4 })), s.name);
    mark(svg.appendChild(svgEl("circle", { cx, cy, r: 4.2, fill: col })), s.name);
  });

  // frontier labels: the candidate spot (right/left of the dot's group, above/below, then
  // further out) with the fewest collisions with the plot edge, dots and earlier labels;
  // the first collision-free one wins
  const boxes = [];
  const hits = (b) =>
    (b.x0 < ml || b.x1 > W - 2 ? 1 : 0) +
    pts.filter((o) => { const p = pos.get(o); return !(p.cx < b.x0 - 5 || p.cx > b.x1 + 5 || p.cy < b.y0 - 5 || p.cy > b.y1 + 5); }).length +
    boxes.filter((o) => !(b.x1 < o.x0 || b.x0 > o.x1 || b.y1 < o.y0 || b.y0 > o.y1)).length;
  frontier.forEach((s) => {
    const { cy, half } = pos.get(s), gx = x(ax.x(s)), w = labelWidth(chart, "series-lab", [ciName(s)]);
    const spots = [[gx + half + 8, cy - 9, "start"], [gx - half - 8, cy - 9, "end"],
                   [gx + half + 8, cy + 15, "start"], [gx - half - 8, cy + 15, "end"],
                   [gx - w / 2, cy - 14, "start"], [gx - w / 2, cy + 22, "start"],
                   [gx - w / 2, cy - 26, "start"], [gx - w / 2, cy + 34, "start"]];
    const boxOf = ([tx, ty, a]) => ({ x0: a === "end" ? tx - w : tx, x1: a === "end" ? tx : tx + w, y0: ty - 10, y1: ty + 3 });
    const scored = spots.map((c) => [hits(boxOf(c)), c]);
    const spot = (scored.find(([n]) => n === 0) || scored.reduce((a, b) => (b[0] < a[0] ? b : a)))[1];
    boxes.push(boxOf(spot));
    mark(svg.appendChild(stext("series-lab", spot[0], spot[1], ciName(s), { fill: R_INK, "text-anchor": spot[2] })), s.name);
  });
  pts.forEach((s) => {
    const { cx, cy } = pos.get(s);
    const hit = svgEl("circle", { cx, cy, r: 9, fill: "transparent" });
    ciBind(hit, s.name, s);
    svg.append(hit);
  });

  chart.append(svg);
  ciFocus(CI_VIEW.pin);

  const key = (svgInner, text) => {
    const it = el("div", "leg-key");
    it.innerHTML = `<svg width="22" height="12" viewBox="0 0 22 12">${svgInner}</svg>`;
    it.append(el("span", "", text));
    $("#ci-legend").append(it);
  };
  $("#ci-legend").innerHTML = "";
  key(`<path d="M1 10H9V3H21" fill="none" stroke="${R_PRIMARY}" stroke-opacity=".55" stroke-width="1.6"/>`, "frontier");
  key(`<circle cx="11" cy="6" r="4.2" fill="${R_PRIMARY}"/>`, "scored on every language");
  key(`<circle cx="11" cy="6" r="4.2" fill="${R_MID}"/>`, "some languages only");
  key(`<circle cx="11" cy="6" r="4.2" fill="${R_RED}"/>`, "Zyphra");
}

bindChartResize("#ci-chart", "index", () => !!CI_VIEW.data, () => drawIndexView(ciIndex()));
