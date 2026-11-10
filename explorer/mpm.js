/* ============================================================
   ZTTS2 · Eval — prosody realism (MPM span dot plot)
   One row per system, SORTED best→worst by mean span, so the
   ranking is vertical position. Each row: the system's per-language
   spans (faint dots) with a mean marker + range, on the shared span
   axis (0 = ground truth … 1 = contour-scrambled; past 1 = further
   from real than scrambled). Reads /api/mpm_dist (build_mpm_dist.py).
   Depends on app.js globals ($, el, svgEl, apiGet, STATE) and the
   chart kit (stext, newSvg, makeFocus, makeTip, ensureCache,
   bindChartResize) from chart_common.js (loaded first).
   ============================================================ */
"use strict";

// zyphra palette: ground truth in ink, the scrambled side in the orange accent
const M_INK = "#111111", M_GT = "#161616", M_MUT = "#9a9a95",
      M_RULE = "#d4d4d0", M_SCR = "#e7620f", M_FAINT = "#efefec", M_DOT = "#c9c9c4";

const MPM = { data: null, pin: null };

function ensureMpm() {
  return ensureCache(MPM, "/api/mpm_dist", (d) => !!(d.langs && d.langs.length));
}

window.renderMPM = async function () {
  const chart = $("#mpm-chart");
  const ok = await ensureMpm();
  if (!ok) {
    chart.querySelectorAll("svg").forEach((n) => n.remove());
    if (!chart.querySelector(".loading"))
      chart.append(el("div", "loading", "no MPM cache — run scripts/prosody/score_mpm_shuffle_floor.py + scripts/results_doc/build_mpm_dist.py"));
    $("#mpm-legend").innerHTML = ""; $("#mpm-langs").innerHTML = "";
    $("#mpm-source").innerHTML = ""; $("#ro-channel").textContent = "prosody realism / —";
    return;
  }
  chart.querySelectorAll(".loading").forEach((n) => n.remove());
  $("#ro-channel").textContent = `prosody realism / ${MPM.data.systems.length} systems`;
  $("#mpm-langs").innerHTML = "";
  updateMpmHead();
  drawMpm(MPM.data);
  const src = $("#mpm-source");
  if (src) src.innerHTML =
    `<span><b>Data:</b> ZTTS2-Eval · MPM (TTSDS masked prosody model) · span = distance from real speech, ` +
    `normalized per language (0 = ground truth, 1 = contour-scrambled), on audio cut to its Silero VAD speech span. Dot = one language; ◆ = mean. Point estimates.</span>` +
    `<span class="lic">CC BY</span>`;
};

function updateMpmHead() {
  $("#mpm-title").textContent = "Prosody realism — distance from ground truth";
  $("#mpm-note").textContent =
    `Systems ranked (top = most realistic prosody). Each ◆ is the mean across languages; ` +
    `faint dots are the individual languages, so a wide spread = inconsistent across languages. ` +
    `0 = indistinguishable from ground truth, 1 = fully contour-scrambled — a mean past the ` +
    `scrambled line is further from real than scrambling the prosody would be.`;
}

// focus = the system currently shown (hovered, or the pinned one). Dims other
// rows and (via the `after` hook) reveals the focused system's per-language labels.
const _mpmFocus = makeFocus("#mpm-chart", {
  dim: "0.18",
  after: (chart, shown) => {
    chart.querySelectorAll("[data-lbl]").forEach((e) => {
      e.style.opacity = (shown && e.getAttribute("data-lbl") === shown) ? "1" : "0";
    });
  },
});
function mpmFocus(name) { _mpmFocus(name || MPM.pin); }
function mpmTogglePin(name) {
  MPM.pin = MPM.pin === name ? null : name;
  mpmFocus(null);
}
const mpmTipKit = makeTip("#mpm-chart", "#mpm-tip");
function mpmTip(html, ev) { mpmTipKit.show(html, ev); }
function mpmHideTip() { mpmTipKit.hide(); }

function drawMpm(d) {
  const chart = $("#mpm-chart");
  chart.querySelectorAll("svg").forEach((n) => n.remove());
  $("#mpm-tip").hidden = true;
  $("#mpm-legend").innerHTML = "";

  // pivot: system -> its per-language spans; rows sorted best (lowest mean) first
  const bySys = {};
  d.systems.forEach((s) => (bySys[s.name] = { name: s.name, mean: s.mean, pts: [] }));
  d.langs.forEach((L) => L.systems.forEach((s) => {
    if (bySys[s.name]) bySys[s.name].pts.push({ lang: L.lang, span: s.span });
  }));
  const rows = Object.values(bySys).sort((a, b) => a.mean - b.mean);

  const rh = 40, mt = 34, mb = 40, mr = 46;
  const ml = Math.max(80, labelWidth(chart, "series-lab", rows.map((r) => r.name)) + 24);
  const W = chartWidth(chart, 440, ml + mr);
  const plotW = W - ml - mr;
  const H = mt + rows.length * rh + mb;
  const xmax = Math.max(1.05, d.xmax);
  const x = (v) => ml + (v / xmax) * plotW;

  // The chart is as tall as it has systems, so it must set its own height: the
  // .scope-chart rule pins any bare <svg> to 440px, which a viewBox then honours by
  // shrinking the whole drawing (text included) to fit. newSvg sets width/height
  // inline, which wins, and the container has to stop clipping the overflow.
  chart.style.overflow = "visible";
  const svg = newSvg(W, H);
  const yTop = mt, yBot = mt + rows.length * rh;

  // "worse than scrambled" zone
  if (xmax > 1) svg.append(svgEl("rect", { x: x(1), y: yTop, width: x(xmax) - x(1), height: rows.length * rh, fill: M_SCR, opacity: 0.06 }));

  // x grid + anchor lines (0 = GT, 1 = scrambled)
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  if (xmax > 1.25) ticks.push(Math.round(xmax * 10) / 10);
  // narrow screens draw this chart at its minimum width, where the quarter ticks sit
  // ~10px apart: label every 2nd quarter, or only the 0 / 1 anchors (the stride rounded
  // up to a divisor of 4); the far tick past 1 is always labelled
  const k = tickStride(x(0.25) - x(0), labelWidth(chart, "axis-txt", ["0.75"]));
  const quarter = k === 1 ? 1 : k === 2 ? 2 : 4;
  ticks.forEach((t) => {
    const anchor = t === 0 || t === 1;
    svg.append(svgEl("line", { x1: x(t), y1: yTop, x2: x(t), y2: yBot, stroke: anchor ? (t === 0 ? M_GT : M_SCR) : M_RULE, "stroke-width": anchor ? 1.4 : 1, "stroke-dasharray": anchor ? "" : "2 3" }));
    if (t > 1 || Math.round(t * 4) % quarter === 0)
      svg.append(stext("axis-txt", x(t), yBot + 15, String(t), { "text-anchor": "middle", fill: anchor ? (t === 0 ? M_GT : M_SCR) : M_MUT }));
  });
  // "= ground truth" / "= scrambled" sit over their lines; when the lines are too close
  // for both centred labels (narrow screens) they part outward from their lines
  const apart = x(1) - x(0) < (labelWidth(chart, "axis-txt", ["= ground truth"]) + labelWidth(chart, "axis-txt", ["= scrambled"])) / 2 + 8;
  svg.append(stext("axis-txt", apart ? x(0) + 4 : x(0), mt - 10, "= ground truth", { "text-anchor": apart ? "end" : "middle", fill: M_GT }));
  svg.append(stext("axis-txt", apart ? x(1) - 4 : x(1), mt - 10, "= scrambled", { "text-anchor": apart ? "start" : "middle", fill: M_SCR }));
  svg.append(stext("axis-lab", ml + plotW / 2, H - 6, "MPM span — distance from real speech  (← more real)", { "text-anchor": "middle" }));

  const dotTip = (row, p) => `<b>${row.name}</b> · ${p.lang}<br>span <b>${p.span.toFixed(2)}</b>${p.span > 1 ? " · past scrambled" : ""}`;
  // mean value labels sit right of each row's range; one that would straddle the
  // scrambled line (narrow screens, where 0 → 1 is ~40px) moves just past it
  const valW = labelWidth(chart, "axis-txt", rows.map((r) => r.mean.toFixed(2)));
  rows.forEach((row, i) => {
    const yc = mt + i * rh + rh / 2;
    if (i % 2 === 1) svg.append(svgEl("rect", { x: ml, y: mt + i * rh, width: plotW, height: rh, fill: M_FAINT }));
    const g = svgEl("g", {});
    g.setAttribute("data-s", row.name);

    // full-row hit band FIRST (behind the marks) so dots stay hoverable; whole row
    // toggles the pin on click and focuses on hover
    const band = svgEl("rect", { x: ml, y: mt + i * rh, width: plotW, height: rh, fill: "transparent" });
    band.style.cursor = "pointer";
    band.addEventListener("mouseenter", () => mpmFocus(row.name));
    band.addEventListener("mouseleave", () => mpmFocus(null));
    band.addEventListener("click", () => mpmTogglePin(row.name));
    g.append(band);

    // range line (min→max across languages) — shows spread
    const spans = row.pts.map((p) => p.span);
    const lo = Math.min(...spans), hi = Math.max(...spans);
    g.append(svgEl("line", { x1: x(lo), y1: yc, x2: x(hi), y2: yc, stroke: M_DOT, "stroke-width": 2 }));

    // individual language dots + their (hidden) language labels
    row.pts.forEach((p) => {
      const dot = svgEl("circle", { cx: x(p.span), cy: yc, r: 3.4, fill: p.span > 1 ? M_SCR : M_MUT, opacity: 0.6 });
      dot.style.cursor = "pointer";
      dot.addEventListener("mouseenter", (ev) => { mpmFocus(row.name); mpmTip(dotTip(row, p), ev); });
      dot.addEventListener("mousemove", (ev) => mpmTip(dotTip(row, p), ev));
      dot.addEventListener("mouseleave", () => { mpmFocus(null); mpmHideTip(); });
      dot.addEventListener("click", (ev) => { ev.stopPropagation(); mpmTogglePin(row.name); });
      g.append(dot);
      const lbl = stext("axis-txt", x(p.span), yc - 8, p.lang,
        { "text-anchor": "middle", fill: p.span > 1 ? M_SCR : M_MUT });
      lbl.setAttribute("data-lbl", row.name);
      lbl.style.opacity = "0"; lbl.style.pointerEvents = "none";
      g.append(lbl);
    });

    // mean marker (diamond) + value
    const mx = x(row.mean), past = row.mean > 1;
    const dia = svgEl("path", { d: `M${mx} ${yc - 6} L${mx + 6} ${yc} L${mx} ${yc + 6} L${mx - 6} ${yc} Z`, fill: past ? M_SCR : M_GT, stroke: "#fafaf9", "stroke-width": 1 });
    dia.style.pointerEvents = "none";
    g.append(dia);
    let vx = x(hi) + 10;
    if (vx < x(1) + 4 && vx + valW > x(1) - 4) vx = x(1) + 6;
    g.append(stext("axis-txt", vx, yc, row.mean.toFixed(2), { "text-anchor": "start", "dominant-baseline": "middle", fill: past ? M_SCR : M_MUT }));

    // system name (rank = row position)
    g.append(stext("series-lab", ml - 12, yc, row.name, { "text-anchor": "end", "dominant-baseline": "middle", fill: M_INK }));
    svg.append(g);
  });

  chart.append(svg);
  if (MPM.pin) mpmFocus(MPM.pin);
}

// redraw on resize while the prosody-overlap view is active (the row layout is
// computed against the container width, so a stale draw is a stretched draw)
bindChartResize("#mpm-chart", "mpm", () => !!MPM.data, () => drawMpm(MPM.data));
