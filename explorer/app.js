/* ============================================================
   ZTTS2 · Eval — explorer (vanilla js, no deps)
   Zyphra theme (see app.css)
   ============================================================ */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const el = (tag, cls, txt) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (txt != null) n.textContent = txt;
  return n;
};

const arr = (x) => (Array.isArray(x) ? x[0] : x);
// snapshot wer arrays are [mean, l50, clip] — use the clipped value (WER_CLIP)
const clip = (x) => (Array.isArray(x) ? (x.length >= 3 ? x[2] : x[0]) : x);
const fmt2 = (v) => (v == null || isNaN(v) ? null : (+v).toFixed(2));
const fmt1 = (v) => (v == null || isNaN(v) ? null : (+v).toFixed(1));

// `ci` is the 95% interval where one is on record — shown on panel boards, where the
// sampling error of the subsample is the whole reason to be careful with the number.
const METRICS = [
  { key: "wer",       label: "WER",   get: (r) => clip(r.wer),      fmt: fmt2, low: true,
    ci: (r) => (r.wer_ci && r.wer_ci.length >= 4 ? [r.wer_ci[2], r.wer_ci[3]] : null) },
  { key: "tail50",    label: "Tail>50%", get: (r) => r.tail50,     fmt: fmt1, low: true  },
  { key: "spk_xutt",  label: "SPK held-out", get: (r) => r.spk_xutt, fmt: fmt1, low: false },
  { key: "spk",       label: "SPK prompt", get: (r) => r.spk,     fmt: fmt1, low: false },
  { key: "msr_utmos", label: "MSR",   get: (r) => r.msr_utmos,     fmt: fmt2, low: false,
    ci: (r) => (r.msr_utmos_ci && r.msr_utmos_ci[0] != null ? r.msr_utmos_ci : null) },
  { key: "mpm_span",  label: "MPM (Clean only)", get: (r) => r.mpm_span, fmt: fmt2, low: true  },
  { key: "rhythm_rate_x", label: "Rhythm pace ×ceil", get: (r) => r.rhythm_rate_x, fmt: fmt2, low: true },
  { key: "rhythm_x",  label: "Rhythm pause ×ceil", get: (r) => r.rhythm_x, fmt: fmt2, low: true },
  { key: "pitch_range_x", label: "Pitch range ×ceil", get: (r) => r.pitch_range_x, fmt: fmt2, low: true },
  { key: "paired_dur", label: "Dur |log2|", get: (r) => r.paired_dur, fmt: fmt2, low: true },
  { key: "paired_pause", label: "Pause Δ", get: (r) => r.paired_pause, fmt: fmt2, low: true },
  { key: "paired_range", label: "Range |log2|", get: (r) => r.paired_range, fmt: fmt2, low: true },
  { key: "n",         label: "N",     get: (r) => r.n,             fmt: (v) => (v == null ? null : String(v)), low: false, neutral: true },
];
const METRIC = Object.fromEntries(METRICS.map((m) => [m.key, m]));

const STATE = {
  snap: null,
  task: null,
  lang: null,
  metric: "wer",
  view: "board",
  sortKey: "wer",
  sortDir: null, // 'asc' | 'desc'
  filter: "",
  // grid colour range: `gridClip` trims that percent of the cells off EACH end of the
  // automatic range; `gridRange[metric]` = manual {lo, hi} (either null = automatic)
  gridClip: 0,
  gridRange: {},
};

// ---------------------------------------------------------------------------
// JSONP data loader
// Some content-inspection proxies pass <script> (application/javascript) intact
// but truncate large application/json responses. So we fetch data as JSONP via
// dynamic <script>, in small paginated chunks — the exact delivery path app.js
// itself uses successfully.
// ---------------------------------------------------------------------------
let _cbN = 0;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function jsonp(path, timeoutMs = 20000) {
  return new Promise((resolve, reject) => {
    const cb = "__ztts2cb" + (++_cbN);
    const sep = path.includes("?") ? "&" : "?";
    const s = document.createElement("script");
    let done = false;
    const finish = (fn, arg) => { if (done) return; done = true; clearTimeout(to); try { delete window[cb]; } catch (e) { window[cb] = undefined; } s.remove(); fn(arg); };
    const to = setTimeout(() => finish(reject, new Error("timeout loading " + path)), timeoutMs);
    window[cb] = (data) => finish(resolve, data);
    s.onerror = () => finish(reject, new Error("network error loading " + path));
    // if the script loaded but never called our callback, it was truncated/blocked
    s.onload = () => setTimeout(() => finish(reject, new Error("response truncated or blocked: " + path)), 0);
    // the serve-time token (minted into index.html) unlocks the JSONP path —
    // without it the server answers plain JSON, which a <script> can't read
    s.src = path + sep + "cb=" + cb + "&t=" + encodeURIComponent(window.ZTTS2_TOKEN || "") + "&_=" + _cbN;
    document.head.appendChild(s);
  });
}

async function jsonpRetry(path, tries = 4) {
  let last;
  for (let i = 0; i <= tries; i++) {
    try { return await jsonp(path); }
    catch (e) { last = e; if (i < tries) await sleep(650); }
  }
  throw last;
}

// Primary loader: normal fetch + JSON. Pages are small (~4KB gzipped) so they
// sail through. If a body still arrives truncated (JSON.parse throws) we retry,
// then fall back to JSONP for that request — so a hostile proxy can't blank us.
let _bust = 0;
async function apiGet(path, tries = 3) {
  for (let i = 0; i <= tries; i++) {
    try {
      const u = path + (path.includes("?") ? "&" : "?") + "_=" + (++_bust);
      const resp = await fetch(u, { cache: "no-store" });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      return JSON.parse(await resp.text()); // throws on a truncated body
    } catch (e) {
      if (i < tries) { await sleep(350); continue; }
      console.warn("plain JSON failed for", path, "→ JSONP fallback:", e.message);
      return jsonpRetry(path);
    }
  }
}

// fetch a large list endpoint as small pages, in parallel, and reassemble
async function pagedGet(baseUrl, total, page = 120) {
  const sep = baseUrl.includes("?") ? "&" : "?";
  const offs = [];
  for (let o = 0; o < total; o += page) offs.push(o);
  const parts = await Promise.all(offs.map((o) => apiGet(`${baseUrl}${sep}offset=${o}&limit=${page}`)));
  parts.sort((a, b) => (a.offset || 0) - (b.offset || 0));
  return parts.flatMap((p) => p.rows || []);
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
async function boot() {
  buildMetricSelect();
  bindConsole();
  bindDrawer();
  loadSnapshot();
}

function applySnapshot(snap) {
  try {
    STATE.snap = snap;
    $("#ro-models").textContent = snap.models.length;
    initTaskLang();
    render();
  } catch (e) {
    console.error("render failed after snapshot loaded:", e);
    $("#board-body").innerHTML =
      `<tr><td class="empty-state">snapshot loaded OK, but rendering threw:<br>` +
      `<small>${e.message}</small><br><br>open the browser console for the full stack trace.</td></tr>`;
  }
}

async function loadSnapshot() {
  $("#board-body").innerHTML = `<tr><td class="empty-state loading">acquiring snapshot</td></tr>`;
  try {
    const meta = await apiGet("/api/meta");                 // tiny: total + option lists
    const rows = await pagedGet("/api/rows", meta.total, 120); // small paginated chunks
    applySnapshot({ rows, models: meta.models, tasks: meta.tasks, sets: meta.sets,
                    panels: meta.panels || {} });
  } catch (e) {
    console.error("snapshot load failed:", e);
    $("#board-body").innerHTML =
      `<tr><td class="empty-state">couldn't load snapshot from <b>${location.origin}</b><br>` +
      `<small>${e.message}</small><br><br>` +
      `<button id="retry-snap" style="font:inherit;color:var(--bg);background:var(--primary);border:0;border-radius:4px;padding:5px 12px;cursor:pointer">retry</button></td></tr>`;
    const rb = document.getElementById("retry-snap");
    if (rb) rb.onclick = () => loadSnapshot();
  }
}

// GT and Whisper are reference baselines: flat results/<model>/<set>/ dirs, so
// their leaf carries an empty task and is joined into the board of the full task
// their model name encodes (gt_zero_shot -> zero_shot).
function isBaseline(model) { return /^(?:gt|whisper)_/.test(model); }
function gtTaskOf(model) { return model.replace(/^(?:gt|whisper)_/, ""); }
// The join is exact on both sides, and both halves matter:
//   - task === "" — a baseline also carries real per-panel rows (build_results.py
//     restricts its full leaf to each panel's items and weights). Matching on the
//     model name alone would replay every one of them onto the full-task board.
//   - gtTaskOf === task, not a prefix — `gt_zero_shot_voxblink` is a different data
//     family, not a `zero_shot` variant, and prefix-matching renders its VoxBlink
//     numbers as the FLEURS-R floor.
// A panel board therefore never sees the flat row: the full-set floor is a different
// item set and would misstate every "vs GT" read (see explorer/README.md).
function isBaselineFor(r, task) {
  return isBaseline(r.model) && r.task === "" && gtTaskOf(r.model) === task;
}

// ---------------------------------------------------------------------------
// coreset panels (scripts/coreset/, docs/coreset.md)
// A panel task is a weighted subsample of a full task. Its board cells are the
// weighted estimates built by build_results.py, not plain leaf means, and the two
// kinds mean different things: `mini` estimates the full-set level, `delta` only
// resolves differences between systems. Both need saying on screen — a panel score
// read as if it were a full-set score is the one way to misuse these boards.
// ---------------------------------------------------------------------------
function panelOf(task) { return (STATE.snap.panels || {})[task] || null; }

function panelLabel(task) {
  const p = panelOf(task);
  if (!p) return task;
  return `${task} — ${p.kind} panel · k=${p.k_actual}`;
}

function panelNote(task) {
  const p = panelOf(task);
  if (!p) return null;
  const src = p.source_task || "the full set";
  const prov = `${p.k_actual} items over ${p.langs} langs · selector ${p.selector} · seed ${p.seed}`;
  const docs = `minimum detectable difference and the leave-family-out validation:
                <code>RESULTS_CORESET.md</code>`;
  // Almost every row here is derived, not run. Say so on screen: a panel board that
  // looks like a cheap benchmark but is actually the expensive one restricted would
  // misrepresent both the cost and the independence of what's on it.
  const derived = `Rows are the system's <b>full ${src} leaf restricted</b> to the
           panel's items and weights — the same generations, scored on the panel's
           population. Only rows tagged <span class="pill">run</span> were generated
           on the panel task itself, so the rest are what the panel <em>would</em>
           have reported, not an independent trial of it.`;
  // The board is showing either the panel's own estimate (macro over its languages,
  // what score_panel.py reports) or one language's slice of it. Those two mean very
  // different things, so the note says which one is on screen.
  const agg = STATE.lang === ALL_SET;
  const scope = agg
    ? `Showing the <b>panel estimate</b>: each system's weighted mean per language,
       macro-averaged over all ${p.langs} — the number
       <code>scripts/coreset/score_panel.py</code> reports, with its bootstrap CI.
       Pick a language to see the slice behind it.`
    : `Showing <b>one language's slice</b> of the panel — roughly
       ${Math.round(p.k_actual / p.langs)} items, so read it as an indication and
       switch to <b>ALL langs</b> for the estimate the panel was built to deliver.`;
  if (p.kind === "delta") {
    return [`<b>Delta panel.</b> Discriminative subsample of <b>${src}</b>: read
             <em>differences between systems</em> only. The level shown is not an estimate
             of the ${src} score — the items are chosen for their power to separate
             checkpoints, which biases the level by construction. Its power is in the
             <em>paired aggregate over all ${p.k_actual} items</em>${agg ? "" : `, not in a
             single language's cell: spread over ${p.langs} languages, one language holds
             only a handful of items, few enough that systems routinely tie`}. Use
             <code>scripts/coreset/score_panel.py</code> for the paired read.
             ${docs}<br>${scope}<br>${derived}`, prov];
  }
  return [`<b>Mini panel.</b> Weighted estimate of the <b>${src}</b> score, using the
           panel estimator (Σ w·y over the selected items), not a plain mean — so it
           carries sampling error, hence the interval under each cell. Resolves
           tiers, not the frontier. ${docs}<br>${scope}<br>${derived}`, prov];
}

function tasksAvailable() {
  const t = STATE.snap.tasks.slice();
  if (t.length) return t;
  // Baseline-only snapshot (no TTS-model generations scored yet): fall back to
  // the tasks implied by the baseline model names so GT/Whisper are still
  // browsable instead of the board coming up empty.
  const s = new Set();
  STATE.snap.rows.forEach((r) => { if (isBaseline(r.model)) s.add(gtTaskOf(r.model)); });
  return Array.from(s).sort();
}
// The reserved set name build_results.py gives a macro-averaged row: on a panel it is the
// panel estimate over all its languages, which IS the panel's score; on a full set it is
// the unweighted macro over the set's language basket (RESULTS_CURRENT.md's headline
// number). Sorts first so it heads the picker, and it is the default scope: one number
// per system first, the per-language cells one pick away.
const ALL_SET = "ALL";
function langsForTask(task) {
  const s = new Set();
  STATE.snap.rows.forEach((r) => { if (r.task === task) s.add(r.set); });
  if (!s.size) STATE.snap.sets.forEach((x) => s.add(x));
  const all = s.delete(ALL_SET);
  return (all ? [ALL_SET] : []).concat(Array.from(s).sort());
}

function defaultLang(task) {
  const langs = langsForTask(task);
  if (langs.includes(ALL_SET)) return ALL_SET;
  return langs.includes("en") ? "en" : langs[0];
}

function initTaskLang() {
  const tasks = tasksAvailable();
  STATE.task = tasks.includes("zero_shot") ? "zero_shot" : tasks[0];
  const taskSel = $("#sel-task");
  taskSel.innerHTML = "";
  tasks.forEach((t) => taskSel.append(new Option(panelLabel(t), t)));
  taskSel.value = STATE.task;
  refreshLangSelect();
}

// Switching INTO a panel snaps to its aggregate: carrying a language into a panel would
// open on its noisiest view. Otherwise the scope carries over when the task has it.
function refreshLangSelect() {
  const langs = langsForTask(STATE.task);
  if (!langs.includes(STATE.lang) || (panelOf(STATE.task) && STATE.lang !== ALL_SET)) {
    STATE.lang = defaultLang(STATE.task);
  }
  const sel = $("#sel-lang");
  sel.innerHTML = "";
  langs.forEach((l) => sel.append(new Option(l === ALL_SET ? "ALL langs" : l, l)));
  sel.value = STATE.lang;
}

// ---------------------------------------------------------------------------
// console (filter bar) wiring
// ---------------------------------------------------------------------------
// the grid's heatmap metric (the board shows every metric as a column); picking one
// also makes it the board's sort, as the old pill row did
function buildMetricSelect() {
  const sel = $("#sel-metric");
  METRICS.forEach((m) => sel.append(new Option(m.label, m.key)));
  sel.value = STATE.metric;
  sel.onchange = () => {
    STATE.metric = sel.value;
    STATE.sortKey = sel.value;
    STATE.sortDir = null;
    render();
  };
}

function bindConsole() {
  $("#sel-task").onchange = (e) => { STATE.task = e.target.value; refreshLangSelect(); render(); };
  $("#sel-lang").onchange = (e) => { STATE.lang = e.target.value; render(); };
  $("#inp-filter").oninput = (e) => { STATE.filter = e.target.value.toLowerCase(); render(); };
  $$("#view-seg button").forEach((b) => {
    b.onclick = () => {
      STATE.view = b.dataset.view;
      $$("#view-seg button").forEach((x) => x.classList.toggle("on", x === b));
      render();
    };
  });
  bindGridTools();
}

function bindGridTools() {
  const manual = () => (STATE.gridRange[STATE.metric] ||= { lo: null, hi: null });
  const num = (s) => (s.trim() === "" || isNaN(Number(s)) ? null : Number(s));
  $("#grid-clip").oninput = (e) => { STATE.gridClip = Number(e.target.value); renderGrid(); };
  // `change`, not `input`: a half-typed number ("1.", "-") must not rescale the grid
  $("#grid-lo").onchange = (e) => { manual().lo = num(e.target.value); renderGrid(); };
  $("#grid-hi").onchange = (e) => { manual().hi = num(e.target.value); renderGrid(); };
  $("#grid-auto").onclick = () => {
    delete STATE.gridRange[STATE.metric];
    STATE.gridClip = 0;
    renderGrid();
  };
}

// Linear-interpolated quantile of a pre-sorted array, q in [0, 1].
function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q, i = Math.floor(pos), f = pos - i;
  return i + 1 < sorted.length ? sorted[i] + (sorted[i + 1] - sorted[i]) * f : sorted[i];
}

// ---------------------------------------------------------------------------
// data selection
// ---------------------------------------------------------------------------
function boardRows() {
  const { task, lang } = STATE;
  return STATE.snap.rows.filter((r) => {
    if (r.set !== lang) return false;
    return r.task === task || isBaselineFor(r, task);
  }).filter((r) => !STATE.filter || r.model.toLowerCase().includes(STATE.filter));
}

function modelTag(model) {
  if (model.startsWith("gt_")) return ["ref", "GT"];
  if (model.startsWith("whisper_")) return ["ref", "ASR"];
  if (model.startsWith("zonos2")) return ["zonos", "Z2"];
  if (model.startsWith("zonos3")) return ["zonos", "Z3"];
  if (model.startsWith("omni_")) return ["", "OMNI"];
  if (model.startsWith("fal")) return ["", "FAL"];
  if (/^(elevenlabs|cartesia|inworld|gemini)_/.test(model)) return ["", "API"];
  return ["", ""];
}
// Pretty display names for results dirs whose on-disk name is not the public
// model name (kept in sync with scripts/results_doc/gen_md.py MODEL_LABEL).
const MODEL_NAMES = {
  "zonos2_stability": "zonos2-stability",
};
function cleanName(model) {
  if (MODEL_NAMES[model]) return MODEL_NAMES[model];
  return model.replace(/^omni_/, "").replace(/^gt_/, "gt:");
}

// ---------------------------------------------------------------------------
// render dispatch
// ---------------------------------------------------------------------------
function render() {
  const v = STATE.view;
  $("#board-view").classList.toggle("hidden", v !== "board");
  $("#grid-view").classList.toggle("hidden", v !== "grid");
  $("#rel-view").classList.toggle("hidden", v !== "reliability");
  $("#sr-view").classList.toggle("hidden", v !== "speaking");
  $("#mpm-view").classList.toggle("hidden", v !== "mpm");
  $("#ci-view").classList.toggle("hidden", v !== "index");
  // board/grid: app layout — the page stops scrolling and only the table box does
  // (app.css body.app-layout); the chart views keep ordinary page scrolling
  document.body.classList.toggle("app-layout", v === "board" || v === "grid");
  // task/lang/metric/filter controls don't apply to the reliability / speaking-rate /
  // prosody-overlap / index views (each carries its own selectors), and the board shows
  // every metric as a sortable column, so the metric picker is the grid's alone; the
  // grid shows every language as a column, so it has no language picker
  const selfSel = v === "reliability" || v === "speaking" || v === "mpm" || v === "index";
  const hide = { "sel-metric": v === "board", "sel-lang": v === "grid" };
  [$("#sel-task"), $("#sel-lang"), $("#inp-filter"), $("#sel-metric")].forEach((elm) => {
    const c = elm.closest(".ctl");
    if (c) c.style.display = selfSel || hide[elm.id] ? "none" : "";
  });
  // before the self-selecting views return early — those ignore STATE.task, so a
  // panel note left over from the board would caption data it has nothing to do with
  renderPanelNote();
  if (v === "reliability") { if (window.renderReliability) window.renderReliability(); return; }
  if (v === "speaking") { if (window.renderSpeaking) window.renderSpeaking(); return; }
  if (v === "mpm") { if (window.renderMPM) window.renderMPM(); return; }
  if (v === "index") { if (window.renderIndex) window.renderIndex(); return; }
  $("#ro-channel").textContent = v === "grid" ? STATE.task : `${STATE.task} / ${STATE.lang}`;
  if (v === "board") renderBoard(); else renderGrid();
}

function renderPanelNote() {
  const box = $("#panel-note");
  if (!box) return;
  const note = panelNote(STATE.task);
  const show = note && (STATE.view === "board" || STATE.view === "grid");
  box.classList.toggle("hidden", !show);
  if (!show) return;
  const [body, prov] = note;
  box.className = `panel-note ${panelOf(STATE.task).kind}`;
  box.innerHTML = `<p>${body}</p><p class="prov">${prov}</p>`;
}

// ---------------------------------------------------------------------------
// leaderboard
// ---------------------------------------------------------------------------
function renderBoard() {
  const rows = boardRows();
  $("#ro-rows").textContent = rows.length;

  // only show metric columns that have data for this view (e.g. WERn/MSR are
  // only present on some tasks) — keeps WER always, drops dead "—" columns
  const cols = METRICS.filter((m) =>
    m.key === "wer" || rows.some((r) => { const v = m.get(r); return v != null && !isNaN(v); }));
  // if the active sort column got hidden, fall back to WER
  if (STATE.sortKey !== "model" && !cols.some((m) => m.key === STATE.sortKey)) {
    STATE.sortKey = "wer"; STATE.sortDir = null;
  }

  // header
  const head = $("#board-head");
  head.innerHTML = "";
  const tr = el("tr");
  const th0 = el("th", "lead", "model");
  th0.onclick = () => setSort("model");
  tr.append(th0);
  cols.forEach((m) => {
    const th = el("th");
    th.append(document.createTextNode(m.label));
    if (STATE.sortKey === m.key) {
      const a = el("span", "arr", STATE.sortDir === "asc" ? "▲" : "▼");
      th.append(a);
    }
    th.onclick = () => setSort(m.key);
    tr.append(th);
  });
  head.append(tr);

  // per-column min/max for gauges (exclude baselines + nulls so gauges reflect real models)
  const ranges = {};
  cols.forEach((m) => {
    const vals = rows.filter((r) => !isBaseline(r.model))
      .map((r) => m.get(r)).filter((v) => v != null && !isNaN(v));
    ranges[m.key] = vals.length ? [Math.min(...vals), Math.max(...vals)] : [0, 1];
  });

  // sort
  const sm = METRIC[STATE.sortKey];
  const dir = STATE.sortDir || (sm && sm.low ? "asc" : "desc");
  const sorted = rows.slice().sort((a, b) => {
    if (STATE.sortKey === "model") return a.model.localeCompare(b.model) * (dir === "asc" ? 1 : -1);
    let va = sm.get(a), vb = sm.get(b);
    if (va == null) return 1; if (vb == null) return -1;
    return (va - vb) * (dir === "asc" ? 1 : -1);
  });

  const body = $("#board-body");
  body.innerHTML = "";
  if (!sorted.length) {
    body.innerHTML = `<tr><td colspan="${cols.length + 1}" class="empty-state">no rows for ${STATE.task} / ${STATE.lang}</td></tr>`;
    return;
  }

  sorted.forEach((r, i) => {
    const row = el("tr");
    const isGt = isBaseline(r.model);
    if (isGt) row.classList.add("gt");
    row.style.animationDelay = `${Math.min(i * 22, 600)}ms`;

    const lead = el("td", "lead");
    const wrap = el("div", "model-name");
    const [tcls, tlab] = modelTag(r.model);
    if (tlab) wrap.append(el("span", `tag ${tcls}`, tlab));
    wrap.append(el("span", "nm", cleanName(r.model)));
    // On a panel board almost every row is the full-set leaf restricted to the
    // panel, so the rare thing worth marking is a row that was actually generated
    // on the panel task; and a leaf that misses part of the panel is estimating a
    // slightly different population, so it says how much it covers.
    if (r.panel) {
      if (!r.panel.backfilled) {
        const t = el("span", "tag run", "run");
        t.title = "generated and scored on the panel task itself";
        wrap.append(t);
      }
      if (r.panel.coverage != null && r.panel.coverage < 0.999) {
        const t = el("span", "tag thin", `${Math.round(r.panel.coverage * 100)}%`);
        t.title = `leaf covers ${(r.panel.coverage * 100).toFixed(1)}% of this language's panel weight; weights renormalised over what it has`;
        wrap.append(t);
      }
    }
    // a full-set macro needs every language: a system scored on fewer says how many
    if (r.macro && r.macro.langs < r.macro.of) {
      const t = el("span", "tag thin", `${r.macro.langs}/${r.macro.of} langs`);
      t.title = `scored on ${r.macro.langs} of the set's ${r.macro.of} languages: no macro, ` +
        `a mean over a different language subset is a different quantity`;
      wrap.append(t);
    }
    lead.append(wrap);
    row.append(lead);

    cols.forEach((m) => {
      const td = el("td");
      const v = m.get(r);
      const cell = el("div", "cell");
      const vs = m.fmt(v);
      const vSpan = el("span", "v" + (vs == null ? " na" : ""), vs == null ? "—" : vs);
      cell.append(vSpan);
      // On a panel board the cell is an estimate from a subsample, so it carries its
      // interval: without it the extra digits read as precision the panel doesn't have.
      const ci = r.panel && m.ci ? m.ci(r) : null;
      if (ci && vs != null) {
        cell.append(el("span", "ci", `${m.fmt(ci[0])} – ${m.fmt(ci[1])}`));
      }
      if (vs != null && !m.neutral) {
        const [lo, hi] = ranges[m.key];
        let t = hi > lo ? (v - lo) / (hi - lo) : 1;       // 0..1 raw
        if (m.low) t = 1 - t;                             // 1 = best
        const g = el("div", "gauge");
        const bar = el("i");
        bar.style.width = `${Math.max(4, t * 100)}%`;
        if (t < 0.34) bar.classList.add("bad");
        else if (t < 0.67) bar.classList.add("warn");
        g.append(bar);
        cell.append(g);
      } else if (m.neutral && vs != null) {
        const g = el("div", "gauge");
        const bar = el("i");
        const [lo, hi] = ranges[m.key];
        bar.style.width = `${hi > lo ? Math.max(8, ((v - lo) / (hi - lo)) * 100) : 100}%`;
        g.append(bar); cell.append(g);
      }
      td.append(cell);
      row.append(td);
    });

    row.onclick = () => openDrawer(r.model, STATE.task, STATE.lang);
    body.append(row);
  });
}

function setSort(key) {
  if (STATE.sortKey === key) {
    STATE.sortDir = STATE.sortDir === "asc" ? "desc" : "asc";
  } else {
    STATE.sortKey = key;
    const m = METRIC[key];
    STATE.sortDir = key === "model" ? "asc" : (m && m.low ? "asc" : "desc");
  }
  renderBoard();
}

// ---------------------------------------------------------------------------
// heatmap grid
// ---------------------------------------------------------------------------
function lerpColor(t) {
  // t: 0=worst .. .5=neutral .. 1=best — a zyphra orange↔blue diverging scale,
  // pale enough for dark text to read on top (worse=orange, best=blue). The neutral
  // stop is the surface white, so it stays distinct from empty (--bg) cells.
  const stops = [[243, 190, 156], [250, 250, 249], [176, 208, 250]];
  const seg = t < 0.5 ? 0 : 1;
  const f = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
  const a = stops[seg], b = stops[seg + 1];
  const c = a.map((x, i) => Math.round(x + (b[i] - x) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function renderGrid() {
  const m = METRIC[STATE.metric];
  const scroll = $("#grid-scroll");
  scroll.innerHTML = "";

  const task = STATE.task;
  // models that have any row for this task (+ matching gt), filtered
  const langs = langsForTask(task);
  const modelSet = new Set();
  STATE.snap.rows.forEach((r) => {
    if (r.task === task) modelSet.add(r.model);
    if (isBaselineFor(r, task) && langs.includes(r.set)) modelSet.add(r.model);
  });
  let models = Array.from(modelSet).sort();
  if (STATE.filter) models = models.filter((x) => x.toLowerCase().includes(STATE.filter));

  // index value by model|set
  const idx = {};
  STATE.snap.rows.forEach((r) => { idx[`${r.model}|${r.task}|${r.set}`] = r; });
  const lookup = (model, set) => {
    const r = idx[`${model}|${task}|${set}`];
    // Fall back to the flat leaf only for the baseline's OWN full task. On a panel
    // board a missing panel row means build_results.py dropped it for coverage —
    // substituting the full-set floor there would score a different item set.
    if (r || !isBaseline(model) || gtTaskOf(model) !== task) return r;
    return idx[`${model}||${set}`];
  };

  // value range across grid: exclude baselines so the colour scale reflects the
  // spread among real models (GT/Whisper shouldn't compress it), and exclude the
  // ALL column — it is a macro-average of the very cells beside it, so letting it
  // set the scale would compress the per-language spread the heatmap exists to show.
  // But if only baselines are present, calibrate over them — otherwise the range
  // collapses and every cell clamps to a colour extreme.
  const collect = (skipBaseline) => {
    const out = [];
    models.forEach((mo) => langs.forEach((l) => {
      if (l === ALL_SET) return;
      const r = lookup(mo, l); if (!r || (skipBaseline && isBaseline(mo))) return;
      const v = m.get(r); if (v != null && !isNaN(v)) out.push(v);
    }));
    return out;
  };
  let vals = collect(true);
  if (!vals.length) vals = collect(false);
  vals.sort((a, b) => a - b);
  const clip = STATE.gridClip / 100;
  const autoLo = vals.length ? quantile(vals, clip) : 0;
  const autoHi = vals.length ? quantile(vals, 1 - clip) : 1;
  const man = STATE.gridRange[STATE.metric] || {};
  let lo = man.lo != null ? man.lo : autoLo;
  let hi = man.hi != null ? man.hi : autoHi;
  const bad = !(hi > lo);   // inverted / empty manual range: fall back to automatic
  if (bad) { lo = autoLo; hi = autoHi; }
  // values outside [lo, hi] clamp to the colour extremes (see the cell loop)
  const norm = (v) => { let t = hi > lo ? (v - lo) / (hi - lo) : 1; return m.low ? 1 - t : t; };
  syncGridTools(m, autoLo, autoHi, man, bad);

  const tbl = el("table", "heat");
  const thead = el("thead"); const htr = el("tr");
  htr.append(el("th", "", `${m.label} ▾`));
  langs.forEach((l) => htr.append(el("th", l === ALL_SET ? "all" : "", l === ALL_SET ? "ALL" : l)));
  thead.append(htr); tbl.append(thead);

  const tbody = el("tbody");
  models.forEach((mo) => {
    const tr = el("tr");
    const th = el("th", "", cleanName(mo));
    th.onclick = () => openDrawer(mo, task, ALL_SET);   // the system's per-language breakdown
    tr.append(th);
    langs.forEach((l) => {
      const r = lookup(mo, l);
      const v = r ? m.get(r) : null;
      const td = el("td", l === ALL_SET ? "all" : "");
      if (v == null || isNaN(v)) {
        td.classList.add("empty"); td.textContent = "·";
      } else {
        const t = norm(v);
        td.style.background = lerpColor(Math.max(0, Math.min(1, t)));
        td.textContent = m.fmt(v);
        td.title = l === ALL_SET
          ? `${cleanName(mo)} · panel estimate over ${r.panel ? r.panel.langs : "all"} languages\n${m.label} = ${m.fmt(v)}  (n=${r.n})`
          : `${cleanName(mo)} · ${l}\n${m.label} = ${m.fmt(v)}  (n=${r.n})`;
        td.onclick = () => openDrawer(mo, task, l);
      }
      tr.append(td);
    });
    tbody.append(tr);
  });
  tbl.append(tbody);
  scroll.append(tbl);
  $("#ro-rows").textContent = models.length;

  const legend = $("#grid-legend");
  legend.innerHTML = "";
  // the bar runs worse → better, so for lower-is-better metrics the left end is `hi`
  const [worseV, betterV] = m.low ? [hi, lo] : [lo, hi];
  legend.append(el("span", "", `worse`));
  legend.append(el("span", "legend-end", `${m.low ? "≥" : "≤"} ${m.fmt(worseV)}`));
  const bar = el("div", "legend-bar");
  bar.style.background = `linear-gradient(90deg, ${lerpColor(0)}, ${lerpColor(0.5)}, ${lerpColor(1)})`;
  legend.append(bar);
  legend.append(el("span", "legend-end", `${m.low ? "≤" : "≥"} ${m.fmt(betterV)}`));
  legend.append(el("span", "", "better"));
}

// Reflect the effective range in the (static) knob: the clip slider, and the min/max
// fields — a manual bound shows as its value, an automatic one as the placeholder.
function syncGridTools(m, autoLo, autoHi, man, bad) {
  const r4 = (v) => String(Number(v.toPrecision(4)));
  $("#grid-clip").value = STATE.gridClip;
  $("#grid-clip-val").textContent = STATE.gridClip ? `p${STATE.gridClip}–p${100 - STATE.gridClip}` : "min–max";
  [["#grid-lo", man.lo, autoLo], ["#grid-hi", man.hi, autoHi]].forEach(([id, mv, av]) => {
    const inp = $(id);
    if (document.activeElement !== inp) inp.value = mv != null ? r4(mv) : "";
    inp.placeholder = r4(av);
    inp.classList.toggle("bad", bad);
  });
  $("#grid-auto").disabled = !STATE.gridClip && man.lo == null && man.hi == null;
}

// ---------------------------------------------------------------------------
// drawer / drill-down
// ---------------------------------------------------------------------------
let DD_ROWS = [];

function bindDrawer() {
  $("#dd-close").onclick = closeDrawer;
  $("#scrim").onclick = closeDrawer;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  $("#dd-filter").oninput = (e) => filterUtt(e.target.value.toLowerCase());
}

function openDrawer(model, task, lang) {
  resetAudio();   // the drawer body is about to be rebuilt — drop retained players
  $("#drawer").classList.remove("hidden");
  $("#scrim").classList.remove("hidden");
  $("#drawer").setAttribute("aria-hidden", "false");
  $("#dd-title").textContent = cleanName(model);
  const pmeta = panelOf(task);
  $("#dd-meta").innerHTML = `<b>${task}</b> · <b>${lang === ALL_SET ? "all languages" : lang}</b>`
    + (pmeta ? ` · <span class="pill">${pmeta.kind} panel</span>` : "");
  $("#dd-filter").value = "";
  $("#dd-stats").innerHTML = "";
  $("#dd-body").innerHTML = `<div class="loading">loading utterances</div>`;

  // The ALL row is a macro-average over languages, not a leaf: there is no single
  // dir on disk to list, and its 3500 items would be an unreadable wall anyway. Its
  // real drill-down is the breakdown it averages — which languages carry the number
  // — from the snapshot rows already loaded. Clicking one opens that language.
  if (lang === ALL_SET) return openAggregateDrawer(model, task);

  // Baselines live in flat results/<model>/<lang>/ dirs (empty task on disk); the
  // server recovers their data family from the model name. Passing the synthesized
  // task would look for a non-existent results/<model>/<task>/<lang>/ leaf.
  const diskTask = isBaseline(model) ? "" : task;
  // On a panel board the server restricts the leaf to the panel's items and attaches
  // each item's weight, so the drawer shows the same population the board scored —
  // for the baselines that means filtering their full-set leaf down to the panel.
  const pn = panelOf(task) ? `&panel=${encodeURIComponent(task)}` : "";
  const base = `/api/utterances?model=${encodeURIComponent(model)}&task=${encodeURIComponent(diskTask)}&lang=${encodeURIComponent(lang)}${pn}`;
  apiGet(`${base}&offset=0&limit=120`).then(async (first) => {
    if (first.error || !first.total) {
      $("#dd-body").innerHTML = `<div class="empty-state">no per-utterance data on disk for this cell.<br><small>${first.error || ""}</small></div>`;
      return;
    }
    let rows = first.rows || [];
    const offs = [];
    for (let o = rows.length; o < first.total; o += 120) offs.push(o);
    if (offs.length) {
      const more = await Promise.all(offs.map((o) => apiGet(`${base}&offset=${o}&limit=120`)));
      more.sort((a, b) => (a.offset || 0) - (b.offset || 0));
      more.forEach((p) => { rows = rows.concat(p.rows || []); });
    }
    DD_ROWS = rows;
    renderStats(rows);
    renderUtt(rows);
  }).catch((e) => {
    $("#dd-body").innerHTML = `<div class="empty-state">failed to load utterances<br><small>${e.message}</small></div>`;
  });
}

// Drill-down for an ALL row: the per-language pieces the macro-average is built from,
// worst first. This is the honest view of an aggregate — it shows how much of the
// number one bad language is carrying, which a single figure hides.
function openAggregateDrawer(model, task) {
  const m = METRIC[STATE.metric === "n" ? "wer" : STATE.metric];
  // a baseline's full-set rows are flat (task ""); on a panel it has real panel rows
  const t = isBaseline(model) && !panelOf(task) ? "" : task;
  const agg = STATE.snap.rows.find(
    (r) => r.model === model && r.task === t && r.set === ALL_SET);
  const parts = STATE.snap.rows
    .filter((r) => r.model === model && r.task === t && r.set !== ALL_SET)
    .map((r) => ({ row: r, v: m.get(r) }))
    .filter((p) => p.v != null && !isNaN(p.v))
    .sort((a, b) => (m.low ? b.v - a.v : a.v - b.v));   // worst first

  const stats = [
    ["langs", parts.length],
    ["items", agg && agg.n != null ? agg.n : "—"],
    [m.label, agg ? (m.fmt(m.get(agg)) ?? "—") : "—"],
  ];
  const ci = agg && m.ci ? m.ci(agg) : null;
  if (ci) stats.push(["95% CI", `${m.fmt(ci[0])} – ${m.fmt(ci[1])}`]);
  const box = $("#dd-stats");
  box.innerHTML = "";
  stats.forEach(([k, v]) => {
    const s = el("div", "s");
    s.innerHTML = `<b>${v == null ? "—" : v}</b><span>${k}</span>`;
    box.append(s);
  });

  DD_ROWS = [];
  const body = $("#dd-body");
  body.innerHTML = "";
  if (!parts.length) {
    body.innerHTML = `<div class="empty-state">no per-language rows behind this estimate.</div>`;
    return;
  }
  const spread = [parts[parts.length - 1].v, parts[0].v];   // best, worst
  parts.forEach(({ row, v }) => {
    const wrap = el("div", "utt");
    wrap.dataset.search = row.set;
    const line = el("div", "utt-row");
    const id = el("div", "utt-id");
    id.append(el("span", "", row.set));
    id.append(el("span", "txt", `${row.n} items` +
      (row.panel && row.panel.coverage < 0.999
        ? ` · covers ${Math.round(row.panel.coverage * 100)}% of the panel weight` : "")));
    line.append(id);
    // bar is positioned within this system's own best..worst spread, so it reads as
    // "which languages drag this estimate", not as a cross-system comparison
    const t = spread[1] === spread[0] ? 1 : (v - spread[0]) / (spread[1] - spread[0]);
    const g = el("div", "gauge");
    const bar = el("i");
    bar.style.width = `${Math.max(4, (1 - t) * 100)}%`;
    if (t > 0.66) bar.classList.add("bad"); else if (t > 0.33) bar.classList.add("warn");
    g.append(bar);
    g.style.width = "120px";
    line.append(g);
    line.append(chip(m.label, m.fmt(v) ?? "—", ""));
    line.append(el("span", "caret", "▸"));
    line.onclick = () => openDrawer(model, task, row.set);
    wrap.append(line);
    body.append(wrap);
  });
}

function closeDrawer() {
  resetAudio();
  $("#drawer").classList.add("hidden");
  $("#scrim").classList.add("hidden");
  $("#drawer").setAttribute("aria-hidden", "true");
}

// Weighted when the rows carry panel weights (`w`), plain otherwise — so the drawer
// summary reproduces the board cell instead of quietly disagreeing with it.
function mean(rows, f) {
  let sy = 0, sw = 0;
  rows.forEach((r) => {
    const v = f(r);
    if (v == null || isNaN(v)) return;
    const w = r.w == null ? 1 : r.w;
    sy += w * v; sw += w;
  });
  return sw ? sy / sw : null;
}

function renderStats(rows) {
  const weighted = rows.some((r) => r.w != null);
  // per-utt .wer is a raw 0..1+ fraction (runaway ASR loops exceed 1.0); the board
  // cell is WER_CLIP, so clip at 1.0 here too (utils/per_uid.py's clip) — otherwise
  // the drawer header silently disagrees with the cell the user just clicked. The
  // per-utterance chips below keep the raw values that explain any gap.
  const stats = [
    ["utts", rows.length],
    [weighted ? "WER·w" : "WER", fmt2(mean(rows, (r) => (r.wer == null ? null : Math.min(r.wer, 1) * 100)))],
    [weighted ? "MSR·w" : "MSR", fmt2(mean(rows, (r) => r.msr_utmos))],
    [weighted ? "SPK·w" : "SPK", fmt1(mean(rows, (r) => r.spk))],
  ];
  const box = $("#dd-stats");
  box.innerHTML = "";
  stats.forEach(([k, v]) => {
    const s = el("div", "s");
    s.innerHTML = `<b>${v == null ? "—" : v}</b><span>${k}</span>`;
    box.append(s);
  });
}

function werClass(v) {
  if (v == null) return "";
  if (v <= 5) return "good";
  if (v <= 15) return "warn";
  return "bad";
}
function msrClass(v) { if (v == null) return ""; if (v >= 4) return "good"; if (v >= 3) return "warn"; return "bad"; }
function spkClass(v) { if (v == null) return ""; if (v >= 75) return "good"; if (v >= 60) return "warn"; return "bad"; }

function renderUtt(rows) {
  const body = $("#dd-body");
  body.innerHTML = "";
  rows.forEach((r) => {
    const wrap = el("div", "utt");
    wrap.dataset.search = `${r.uid} ${(r.target_text || r.ref_text || "").toLowerCase()}`;

    const row = el("div", "utt-row");
    const id = el("div", "utt-id");
    id.append(el("span", "", r.uid));
    id.append(el("span", "txt", r.target_text || r.ref_text || ""));
    row.append(id);

    const werPct = r.wer == null ? null : r.wer * 100; // per-utt .wer is a 0..1 fraction
    row.append(chip("WER", werPct == null ? "—" : werPct.toFixed(1), werClass(werPct)));
    row.append(chip("MSR", fmt2(r.msr_utmos) || "—", msrClass(r.msr_utmos)));
    row.append(chip("SPK", fmt1(r.spk) || "—", spkClass(r.spk)));
    row.append(el("span", "caret", "▸"));

    const detail = el("div", "utt-detail");
    let built = false;
    row.onclick = () => {
      const open = wrap.classList.toggle("open");
      if (open && !built) { buildDetail(detail, r); built = true; }
    };

    wrap.append(row, detail);
    body.append(wrap);
  });
}

function chip(lab, val, cls) {
  const c = el("div", "chip " + cls);
  c.append(el("span", "lab", lab));
  c.append(document.createTextNode(val));
  return c;
}

function buildDetail(detail, r) {
  // WER diff
  if (r.ref_text != null && r.hyp_text != null) {
    const d = el("div", "txt-block");
    d.append(el("div", "lab", "asr diff · ref → hyp"));
    d.append(renderDiff(r.ref_text, r.hyp_text));
    detail.append(d);
  } else if (r.target_text) {
    const d = el("div", "txt-block");
    d.append(el("div", "lab", "target text"));
    d.append(el("p", "", r.target_text));
    detail.append(d);
  }

  // prompt text
  if (r.prompt_text) {
    const p = el("div", "txt-block");
    p.append(el("div", "lab", "prompt transcript (reference voice)"));
    p.append(el("p", "", r.prompt_text));
    detail.append(p);
  }

  const extras = [];
  // panel weight: this item's share of its language's estimate (weights sum to 1
  // within a language, so 1/k is "average pull")
  if (r.w != null) extras.push(`<span class="emo-tag">panel w: ${(+r.w).toFixed(4)}</span>`);
  if (r.pause_alpha != null) extras.push(`<span class="emo-tag">pause α: ${(+r.pause_alpha).toFixed(3)}</span>`);
  if (r.ins != null) extras.push(`<span class="emo-tag">ins ${(r.ins * 100).toFixed(1)} · del ${(r.del * 100).toFixed(1)} · sub ${(r.sub * 100).toFixed(1)} %</span>`);
  if (extras.length) {
    const e = el("div", "txt-block");
    e.innerHTML = `<div class="lab">measures</div><div style="display:flex;gap:8px;flex-wrap:wrap">${extras.join("")}</div>`;
    detail.append(e);
  }

  // players
  const players = el("div", "players");
  if (r.gen_url) players.append(makePlayer(r.gen_url, "gen", "◣ generated"));
  if (r.prompt_url) players.append(makePlayer(r.prompt_url, "prompt", "◢ reference prompt"));
  detail.append(players);
}

function filterUtt(q) {
  $$(".utt").forEach((u) => {
    u.style.display = !q || u.dataset.search.includes(q) ? "" : "none";
  });
}

// token-level diff (LCS) ------------------------------------------------------
function diffTokens(ref, hyp) {
  const a = (ref || "").trim().split(/\s+/).filter(Boolean);
  const b = (hyp || "").trim().split(/\s+/).filter(Boolean);
  const n = a.length, m = b.length;
  const eq = (x, y) => x.toLowerCase().replace(/[.,!?;:"']/g, "") === y.toLowerCase().replace(/[.,!?;:"']/g, "");
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = eq(a[i], b[j]) ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const ops = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (eq(a[i], b[j])) { ops.push({ t: "ok", w: b[j] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { ops.push({ t: "del", w: a[i] }); i++; }
    else { ops.push({ t: "ins", w: b[j] }); j++; }
  }
  while (i < n) ops.push({ t: "del", w: a[i++] });
  while (j < m) ops.push({ t: "ins", w: b[j++] });
  // collapse del+ins -> substitution
  const out = [];
  for (let k = 0; k < ops.length; k++) {
    if (ops[k].t === "del" && ops[k + 1] && ops[k + 1].t === "ins") {
      out.push({ t: "sub", w: ops[k + 1].w, d: ops[k].w }); k++;
    } else out.push(ops[k]);
  }
  return out;
}

function renderDiff(ref, hyp) {
  const wrap = el("div", "diff");
  diffTokens(ref, hyp).forEach((op) => {
    let s;
    if (op.t === "ok") s = el("span", "ok", op.w);
    else if (op.t === "sub") { s = el("span", "sub", op.w); s.title = `was: ${op.d}`; }
    else if (op.t === "ins") s = el("span", "ins", op.w);
    else { s = el("span", "del", op.w); }
    wrap.append(s, document.createTextNode(" "));
  });
  return wrap;
}

// ---------------------------------------------------------------------------
// custom audio players (waveform + range-streamed playback)
// ---------------------------------------------------------------------------
// decoded-peak LRU: bounded so browsing many large leaves in one session doesn't
// retain a peaks entry (and its full wav fetch+decode promise) per wav forever
const peakCache = new Map();
const PEAK_CACHE_MAX = 100;
let CURRENT_AUDIO = null;
const ALL_AUDIO = new Set();

function fmtTime(s) {
  if (!isFinite(s)) return "0:00";
  const m = Math.floor(s / 60), ss = Math.floor(s % 60);
  return `${m}:${String(ss).padStart(2, "0")}`;
}

function loadPeaks(url, n) {
  if (peakCache.has(url)) {
    const hit = peakCache.get(url);
    peakCache.delete(url); peakCache.set(url, hit);   // LRU refresh
    return hit;
  }
  const p = (async () => {
    const buf = await fetch(url).then((r) => r.arrayBuffer());
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC();
    const audio = await ctx.decodeAudioData(buf);
    const data = audio.getChannelData(0);
    const block = Math.max(1, Math.floor(data.length / n));
    const peaks = [];
    for (let k = 0; k < n; k++) {
      let mx = 0; const start = k * block;
      for (let j = 0; j < block; j += 32) { const v = Math.abs(data[start + j] || 0); if (v > mx) mx = v; }
      peaks.push(mx);
    }
    const dur = audio.duration;
    ctx.close();
    return { peaks, dur };
  })().catch(() => null);
  peakCache.set(url, p);
  while (peakCache.size > PEAK_CACHE_MAX) peakCache.delete(peakCache.keys().next().value);
  return p;
}

function makePlayer(url, kind, title) {
  const root = el("div", "player " + kind);
  const head = el("div", "player-head");
  head.append(el("span", "", title));
  const tEl = el("span", "", "0:00");
  head.append(tEl);
  root.append(head);

  const prow = el("div", "player-row");
  const btn = el("button", "play-btn"); btn.innerHTML = "<span>▶</span>";
  const wave = el("div", "wave");
  const canvas = el("canvas");
  wave.append(canvas);
  prow.append(btn, wave);
  root.append(prow);

  const audio = new Audio();
  audio.preload = "none";
  audio.src = url;
  ALL_AUDIO.add(audio);

  const accent = kind === "gen" ? "#e7620f" : "#2684ff";   // gen=orange, prompt=blue
  let peaks = null, dur = 0;

  function draw() {
    const r = wave.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(2, r.width * dpr);
    canvas.height = Math.max(2, r.height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const W = canvas.width, H = canvas.height, mid = H / 2;
    const prog = dur ? audio.currentTime / dur : 0;
    if (!peaks) {
      ctx.strokeStyle = "rgba(99,99,95,.5)";   // muted baseline
      ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();
      return;
    }
    const bw = W / peaks.length;
    for (let k = 0; k < peaks.length; k++) {
      const h = Math.max(dpr, peaks[k] * H * 0.92);
      const x = k * bw;
      const played = k / peaks.length <= prog;
      ctx.fillStyle = played ? accent : "rgba(212,212,208,.95)";   // grey unplayed
      ctx.fillRect(x, mid - h / 2, Math.max(dpr, bw - dpr), h);
    }
  }

  loadPeaks(url, 110).then((res) => {
    if (res) { peaks = res.peaks; dur = res.dur; tEl.textContent = fmtTime(dur); }
    draw();
  });

  btn.onclick = () => {
    if (audio.paused) {
      if (CURRENT_AUDIO && CURRENT_AUDIO !== audio) stopAllAudio();
      audio.play(); CURRENT_AUDIO = audio;
      btn.classList.add("playing"); btn.innerHTML = "<span>❚❚</span>";
    } else {
      audio.pause();
    }
  };
  wave.onclick = (e) => {
    if (!dur) return;
    const r = wave.getBoundingClientRect();
    audio.currentTime = ((e.clientX - r.left) / r.width) * dur;
    draw();
  };
  audio.ontimeupdate = () => { tEl.textContent = `${fmtTime(audio.currentTime)} / ${fmtTime(dur || audio.duration)}`; draw(); };
  audio.onpause = () => { btn.classList.remove("playing"); btn.innerHTML = "<span>▶</span>"; };
  audio.onended = () => { btn.classList.remove("playing"); btn.innerHTML = "<span>▶</span>"; draw(); };
  audio.onloadedmetadata = () => { if (!dur) { dur = audio.duration; tEl.textContent = fmtTime(dur); } };

  // redraw on resize for crispness
  new ResizeObserver(() => draw()).observe(wave);
  return root;
}

function stopAllAudio() {
  ALL_AUDIO.forEach((a) => { try { a.pause(); a.currentTime = 0; } catch (e) {} });
  CURRENT_AUDIO = null;
}

// Drop every retained Audio element. The drawer rebuilds its DOM wholesale
// (innerHTML = ""), which would otherwise leave each expanded utterance's Audio
// (and its buffered media) alive in ALL_AUDIO forever.
function resetAudio() {
  stopAllAudio();
  ALL_AUDIO.clear();
}

// ---------------------------------------------------------------------------
// SVG helpers shared by the chart views (chart_common.js and the view files)
// ---------------------------------------------------------------------------
const NSVG = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs) => {
  const n = document.createElementNS(NSVG, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
};
// Widest rendered width of `texts` set in chart text class `cls` (series-lab, axis-txt …)
// inside `chart`. Label margins are sized from this, not a per-character guess, so they
// fit whatever font the theme sets. Returns 0 while `chart` is not laid out.
function labelWidth(chart, cls, texts) {
  if (!texts.length) return 0;
  const probe = svgEl("svg", {});
  probe.style.cssText = "position:absolute;visibility:hidden;width:0;height:0";
  texts.forEach((s) => { const t = svgEl("text", { class: cls }); t.textContent = s; probe.append(t); });
  chart.append(probe);
  const w = Math.max(...Array.from(probe.children, (t) => t.getComputedTextLength()));
  probe.remove();
  return Math.ceil(w);
}

boot();
