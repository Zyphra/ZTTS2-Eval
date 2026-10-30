/* ============================================================
   ZTTS2 · Eval — shared chart infrastructure ("chart kit")
   One copy of the interactivity/layout helpers the reliability,
   speaking-rate and MPM views all need: SVG text/scale helpers,
   the hover-to-focus (dim others) + pin + mouse-tooltip idiom,
   wide transparent hit overlays, a non-latching cache loader and
   the resize-redraw binding. Each view parametrizes its look
   (dim opacity, focus emphasis) instead of cloning the kit.
   Depends on app.js globals: $, svgEl, apiGet, STATE.
   Loaded after app.js, before the view scripts.
   ============================================================ */
"use strict";

// palette (owid theme — mirrors app.css :root); shared by all chart views
const R_INK = "#333333", R_NAVY = "#1d3d63", R_MUT = "#858585",
      R_FAINT = "#e7e7e7", R_RULE = "#dadada", R_BUNDLE = "#e0e0e0",
      R_MID = "#9a9a9a", R_RED = "#b13507", R_GOLD = "#C0870B", R_BLUE = "#4C6A9C";
const SMALL_N = 300;      // series below this are greyed (tufte default)

// ---- scales -----------------------------------------------------------------
const log10 = (v) => Math.log10(v);
function logMap(d0, d1, r0, r1) {
  const a = log10(d0), b = log10(d1);
  return (v) => r0 + (log10(v) - a) / (b - a) * (r1 - r0);
}
function linMap(d0, d1, r0, r1) {
  return (v) => (d1 === d0 ? r0 : r0 + (v - d0) / (d1 - d0) * (r1 - r0));
}
const niceStep = (span) => Math.max(1, Math.round(span / 5));

// ---- SVG helpers ------------------------------------------------------------
function stext(cls, x, y, str, extra) {
  const a = { class: cls, x, y };
  if (extra) Object.assign(a, extra);
  const t = svgEl("text", a);
  t.textContent = str;
  return t;
}

function newSvg(W, H) {
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, preserveAspectRatio: "none" });
  svg.style.width = W + "px"; svg.style.height = H + "px";
  return svg;
}

function mark(elm, name, emph) {
  elm.setAttribute("data-s", name);
  if (emph) elm.setAttribute("data-emph", "");
  return elm;
}

// ---- interactivity: hover-to-focus (dim others) + pin + mouse tooltip -------
// Every element carrying series data gets data-s=<name>; the primary line paths
// also get data-emph so focus can emphasise + raise them (the owid line-chart
// idiom). Hit targets are wide transparent overlays so thin lines are easy to
// grab. Focus is applied by mutating the live SVG, not by re-rendering.

// tooltip at the mouse, inside chartSel's coordinate frame. It is centred above the
// pointer (CSS translate(-50%, -120%)) and the chart clips overflow, so keep it inside:
// clamp it sideways, and flip it below the pointer when there is no room above.
function makeTip(chartSel, tipSel) {
  return {
    show(html, ev) {
      const tip = $(tipSel), r = $(chartSel).getBoundingClientRect();
      tip.innerHTML = html;
      tip.hidden = false;
      const w = tip.offsetWidth, h = tip.offsetHeight;
      const x = ev.clientX - r.left, y = ev.clientY - r.top - 8;
      tip.style.left = Math.min(Math.max(x, w / 2 + 4), r.width - w / 2 - 4) + "px";
      tip.style.top = (y - 1.2 * h < 0 ? y + 1.2 * h + 24 : y) + "px";
    },
    hide() { $(tipSel).hidden = true; },
  };
}

// setFocus(name): dim every [data-s] except `name`, apply opts.emph to the
// [data-s][data-emph] primaries, run opts.after. A focus target absent from the
// current chart (e.g. a pin not among a plate's top-N) must not dim everything —
// it is treated as no focus. Returns the effective name.
//   opts: { dim: "0.1", emph(elm, focused), after(chart, name) }
function makeFocus(chartSel, opts) {
  opts = opts || {};
  return function setFocus(name) {
    const chart = $(chartSel);
    if (name) {
      let found = false;
      chart.querySelectorAll("[data-s]").forEach((e) => { if (e.getAttribute("data-s") === name) found = true; });
      if (!found) name = null;
    }
    chart.querySelectorAll("[data-s]").forEach((e) => {
      const s = e.getAttribute("data-s");
      e.style.opacity = (!name || s === name) ? "" : (opts.dim || "0.1");
    });
    if (opts.emph) {
      chart.querySelectorAll("[data-s][data-emph]").forEach((e) => {
        opts.emph(e, !!(name && e.getAttribute("data-s") === name));
      });
    }
    if (opts.after) opts.after(chart, name);
    return name;
  };
}

// wire enter/move/leave/click on an element to focus `name` and show `rec`'s
// tip; click toggles the pin held in state.pin
function makeBind(focus, tip, state, summary) {
  return function bind(elm, name, rec, col) {
    elm.style.cursor = "pointer";
    elm.addEventListener("mouseenter", (ev) => { focus(name); if (rec) tip.show(summary(rec, col), ev); });
    elm.addEventListener("mousemove", (ev) => { if (rec) tip.show(summary(rec, col), ev); });
    elm.addEventListener("mouseleave", () => { focus(state.pin); if (!state.pin) tip.hide(); });
    elm.addEventListener("click", (ev) => {
      ev.stopPropagation();
      state.pin = state.pin === name ? null : name;
      focus(state.pin);
      if (!state.pin) tip.hide();
    });
  };
}

// a wide transparent overlay so a thin path/line is easy to hover
function makeHit(bind) {
  return function hit(svg, d, name, rec, col) {
    const h = svgEl("path", { d, fill: "none", stroke: "transparent", "stroke-width": "12", "pointer-events": "stroke" });
    bind(h, name, rec, col);
    svg.append(h);
  };
}

// ---- data loading -----------------------------------------------------------
// Fetch-and-cache a view's payload on state.data. Failure does NOT latch: a
// transient apiGet error (server briefly down, cache mid-rebuild) is retried on
// the next render instead of permanently showing "no cache on disk".
//   ok(d)   -> truthy when the payload is usable (also gates the cached value)
//   init(d) -> optional one-time state setup after a fresh load
async function ensureCache(state, url, ok, init) {
  if (state.data) return !!ok(state.data);
  try {
    const d = await apiGet(url);
    if (!d || d.error || !ok(d)) return false;
    state.data = d;
    if (init) init(d);
    return true;
  } catch (e) { return false; }
}

// redraw on resize while `viewName` is the active view
function bindChartResize(chartSel, viewName, hasData, redraw) {
  let raf = 0;
  new ResizeObserver(() => {
    if (typeof STATE === "undefined" || STATE.view !== viewName || !hasData()) return;
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(redraw);
  }).observe($(chartSel));
}
