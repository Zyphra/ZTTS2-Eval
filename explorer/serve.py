#!/usr/bin/env python3
"""ZTTS2 · SIGNAL — a zero-dependency explorer for the ZTTS2-Eval TTS benchmark.

Serves a single-page UI plus a small JSON API over the on-disk eval results.
Stdlib only (http.server / json / os / urllib). No metrics are recomputed --
the existing snapshot and per-utterance score files are parsed and joined by uid.

    python explorer/serve.py --port 8000 --root .
    open http://localhost:8000
"""

import argparse
import gzip
import json
import mimetypes
import os
import re
import secrets
import sys
import threading
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))  # repo root, overridable via --root
# Extra realpath roots files may resolve into (--allow-root), e.g. a shared result store
# that results/<system> symlinks point at. Empty = only ROOT.
ALLOWED_ROOTS = []


def _confined(path):
    """True iff realpath `path` is ROOT, under ROOT, or under an --allow-root dir."""
    for root in [os.path.realpath(ROOT)] + ALLOWED_ROOTS:
        if path == root or path.startswith(root + os.sep):
            return True
    return False


# The per-uid score-file parsers are shared with the coreset tooling. utils/per_uid.py
# is stdlib-only, so importing it keeps this server zero-dependency. Resolve against the
# file's own location, not --root, so the import works whatever --root the caller passes.
sys.path.insert(0, ROOT)
from utils.per_uid import (  # noqa: E402
    parse_score_path,
    parse_spk,
    parse_tsv,
    parse_wer,
)
from utils.scp import load_scp, load_text  # noqa: E402

# The explorer's data snapshot resolves live-cache-first, published-fallback:
#   scripts/results_doc/_cache/  — gitignored build scratch; freshest, present on
#                                  the origin host right after a build
#   explorer/snapshot/           — the committed, portable snapshot (publish_snapshot.py);
#                                  what a fresh clone has
# So the same server serves live data on the origin and the shipped snapshot on a
# clone, with no gitignore trickery. Keep the filenames in sync with PUBLISHED in
# scripts/results_doc/publish_snapshot.py.
CACHE_DIR = "scripts/results_doc/_cache"
PUBLISHED_DIR = "explorer/snapshot"

_WARNED = set()
_WARNED_LOCK = threading.Lock()


def _warn_once(msg):
    with _WARNED_LOCK:
        if msg in _WARNED:
            return
        _WARNED.add(msg)
    print(msg, file=sys.stderr)


def _stat_sig(path):
    """(mtime_ns, size) of a file, or None if it can't be stat'ed."""
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def cache_info(name):
    """(abs_path, source, mtime_ns) for a snapshot file, choosing the NEWER of the
    live scratch copy and the committed published copy when both exist — an old or
    aborted build's scratch file must not shadow a freshly pulled published snapshot.
    source is "live" | "published"; mtime_ns is None when neither copy exists."""
    live = _abs(os.path.join(CACHE_DIR, name))
    pub = _abs(os.path.join(PUBLISHED_DIR, name))
    ls, ps = _stat_sig(live), _stat_sig(pub)
    if ls and ps:
        if ps[0] > ls[0]:
            _warn_once(f"[serve] {name}: published snapshot is newer than the _cache "
                       f"scratch copy — serving published (delete {CACHE_DIR}/{name} "
                       f"or rebuild to get back on scratch)")
            return pub, "published", ps[0]
        return live, "live", ls[0]
    if ls:
        return live, "live", ls[0]
    if ps:
        return pub, "published", ps[0]
    return pub, "published", None  # missing everywhere; callers handle the open failure


def cache_path(name):
    """Absolute path to a snapshot file (see cache_info for the resolution rule)."""
    return cache_info(name)[0]


# ---------------------------------------------------------------------------
# tiny mtime-keyed in-process cache: the board fan-out fires ~24 parallel
# /api/rows requests and the drawer ~13 /api/utterances pages, each of which
# would otherwise re-parse the same multi-MB snapshot / leaf files.
# ---------------------------------------------------------------------------
_MEMO = OrderedDict()   # key -> (stat_sig_tuple, value)
_MEMO_LOCK = threading.Lock()
_MEMO_MAX = 32


def _memo_get(key, sig):
    with _MEMO_LOCK:
        hit = _MEMO.get(key)
        if hit is not None and hit[0] == sig:
            _MEMO.move_to_end(key)
            return hit[1]
    return None


def _memo_put(key, sig, value):
    with _MEMO_LOCK:
        _MEMO[key] = (sig, value)
        _MEMO.move_to_end(key)
        while len(_MEMO) > _MEMO_MAX:
            _MEMO.popitem(last=False)


# Per-process random token gating the JSONP path (see Handler._send_api). Minted
# into index.html at serve time; a page from any other origin cannot know it.
JSONP_TOKEN = secrets.token_urlsafe(16)

SNAPSHOT = "results_snapshot.json"
RELIABILITY = "reliability.json"
SPEAKING_RATE = "speaking_rate.json"
MPM_DIST = "mpm_dist.json"
PANELS = "panels.json"
CAPABILITY_INDEX = "capability_index.json"

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/chart_common.js": ("chart_common.js", "application/javascript; charset=utf-8"),
    "/reliability.js": ("reliability.js", "application/javascript; charset=utf-8"),
    "/speaking_rate.js": ("speaking_rate.js", "application/javascript; charset=utf-8"),
    "/mpm.js": ("mpm.js", "application/javascript; charset=utf-8"),
    "/capability.js": ("capability.js", "application/javascript; charset=utf-8"),
}


# ---------------------------------------------------------------------------
# path helper (the per-uid score-file and uid-file parsers live in utils/per_uid.py
# and utils/scp.py and are imported above)
# ---------------------------------------------------------------------------
def _abs(rel):
    return os.path.join(ROOT, rel)


# ---------------------------------------------------------------------------
# API builders
# ---------------------------------------------------------------------------
def _load_cache_json(name):
    """Parse a snapshot cache file, memoized on (path, mtime, size).

    Returns (data, err) where err is None on success, "not found", or a message
    for a corrupt/truncated file (e.g. read mid-rebuild before the builder's
    atomic replace lands). Corrupt reads are NOT memoized — the next request
    re-reads and picks up the finished file."""
    path = cache_path(name)
    sig = _stat_sig(path)
    if sig is not None:
        cached = _memo_get(("json", name), sig)
        if cached is not None:
            return cached
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError:
        return None, "not found"
    except ValueError as e:
        return None, f"corrupt or truncated cache JSON ({name}): {e}"
    if sig is not None:
        _memo_put(("json", name), sig, (data, None))
    return data, None


def build_snapshot():
    data, err = _load_cache_json(SNAPSHOT)
    rows = data if isinstance(data, list) else []
    models = sorted({r["model"] for r in rows})
    tasks = sorted({r["task"] for r in rows if r.get("task")})
    sets = sorted({r["set"] for r in rows})
    out = {"rows": rows, "models": models, "tasks": tasks, "sets": sets}
    if err and err != "not found":
        out["error"] = err
    return out


def build_panels():
    """Coreset panel provenance, `{task: {kind, source_task, selector, k_actual, ...}}`.

    Prefers the published `panels.json` (so a clone with no `data/` tree still labels
    the panel boards), falling back to scanning the shipped panel manifests."""
    try:
        with open(cache_path(PANELS), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        pass
    out = {}
    ddir = _abs("data")
    if not os.path.isdir(ddir):
        return out
    for task in sorted(os.listdir(ddir)):
        man = _panel_manifest(task)
        if man:
            out[task] = {k: v for k, v in man.items() if k != "langs"}
            out[task]["langs"] = len(man.get("langs") or {})
    return out


def _plain_component(x):
    """True iff x is a single plain path component: non-empty, not `.`/`..`, and
    free of separators / absolute prefixes. Everything that reaches the filesystem
    from the query string must pass this before being joined onto a directory."""
    return bool(x) and x not in (os.curdir, os.pardir) and os.path.basename(x) == x


def _panel_manifest(task):
    """The full `data/<task>/panel.json` (weights included), or None.

    `task` reaches here straight from the query string, so it is constrained to a
    single plain path component before it is joined onto `data/`."""
    if not _plain_component(task):
        return None
    try:
        with open(_abs(os.path.join("data", task, "panel.json")), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _snapshot_files_meta():
    """{name: {source, mtime}} for every snapshot file — the data vintage, so a
    client (or curl) can see which copy each view is being served from. Warns once
    when existing files mix live and published copies (e.g. a partial rebuild),
    since the views would then quietly describe different result sets."""
    out = {}
    sources = set()
    for name in (SNAPSHOT, RELIABILITY,
                 SPEAKING_RATE, MPM_DIST, PANELS, CAPABILITY_INDEX):
        _, src, mtime = cache_info(name)
        if mtime is None:
            continue
        out[name] = {"source": src, "mtime": mtime // 10 ** 9}
        sources.add(src)
    if len(sources) > 1:
        _warn_once("[serve] snapshot files mix live (_cache) and published copies — "
                   "views may describe different result vintages; rebuild or publish "
                   "to re-sync (see /api/meta snapshot_files)")
    return out


def build_meta():
    s = build_snapshot()
    out = {"total": len(s["rows"]), "models": s["models"], "tasks": s["tasks"],
           "sets": s["sets"], "panels": build_panels(),
           "snapshot_files": _snapshot_files_meta()}
    if s.get("error"):
        out["error"] = s["error"]
    return out


def build_rows(offset, limit):
    s = build_snapshot()
    rows = s["rows"]
    out = {"offset": offset, "total": len(rows), "rows": rows[offset:offset + limit]}
    if s.get("error"):
        out["error"] = s["error"]
    return out


def uid_sort_key(uid):
    m = re.search(r"(\d+)$", uid)
    return (int(m.group(1)) if m else 1 << 30, uid)


def _data_task(model, task):
    """The data/ task family for this leaf. Model-generation rows carry an explicit
    `task`; the GT/whisper baseline dirs fold it into the model name
    (`gt_clean`, `gt_in-the-wild`), so recover it from the suffix."""
    if task:
        return task
    for t in ("in-the-wild", "clean"):
        if model.endswith(t):
            return t
    return task


def build_utterances(model, task, lang, offset=0, limit=10 ** 9, panel=None):
    """Join every per-utterance file under the leaf result dir by uid, plus the
    matching data/<task>/<lang>/ target+prompt text and prompt audio. The leaf is
    results/<model>/<task>/<lang>/ for model generations, or the flat
    results/<model>/<lang>/ for the GT/whisper baselines (empty task) — mirroring
    scripts/results_doc/build_results.py.

    `panel` names a coreset panel task (`clean_mini`, ...). It restricts the rows
    to that panel's items and attaches each one's weight `w`, which is what the board's
    weighted estimate is built from — needed because the GT/Whisper baselines are only
    ever scored on the full source task, so their panel view is that full leaf filtered
    down. For a leaf that IS a panel leaf the filter is a no-op and only `w` is added.
    Returns a paginated slice with `total` so the client can fetch in small chunks."""
    # model/task/lang come straight from the query string: each must be a single
    # plain path component (panel is checked the same way in _panel_manifest) or
    # the join below would walk out of results/ and data/.
    for name, val in (("model", model), ("task", task), ("lang", lang)):
        if (val or name != "task") and not _plain_component(val):
            return {"error": f"invalid {name}", "rows": [], "total": 0,
                    "offset": offset, "_code": 400}
    rel_parts = ["results", model] + ([task] if task else []) + [lang]
    rdir = os.path.join(*rel_parts)
    rabs = os.path.realpath(_abs(rdir))
    # belt-and-braces: resolve, then require the leaf to still be under ROOT (or an
    # --allow-root dir)
    if not _confined(rabs):
        return {"error": "invalid path", "rows": [], "total": 0,
                "offset": offset, "_code": 400}

    key = ("leaf", model, task, lang, panel)
    sig = tuple(_stat_sig(p) for p in _leaf_paths(model, task, lang, panel, rabs))
    full = _memo_get(key, sig)
    if full is None:
        full = _leaf_join(model, task, lang, panel, rabs, rdir)
        _memo_put(key, sig, full)
    rows = full.get("rows") or []
    out = {"model": model, "task": task, "lang": lang,
           "total": len(rows), "offset": offset, "rows": rows[offset:offset + limit]}
    if full.get("error"):
        out["error"] = full["error"]
    return out


def _leaf_paths(model, task, lang, panel, rabs):
    """Every file the leaf join reads — the memo key's stat signature. Both
    candidate data/ dirs are included when a panel is named, so a panel.json
    appearing/vanishing (which flips the join's ddir choice) changes the sig."""
    paths = [rabs] + [os.path.join(rabs, f) for f in
                      ("wav.scp", "wav_res_ref_text.wer", "msr_utmos_scores.txt",
                       "spk_simi_scores.txt", "pause_metadata.tsv")]
    fams = []
    if panel and _plain_component(panel):
        paths.append(_abs(os.path.join("data", panel, "panel.json")))
        fams.append(panel)
    fams.append(_data_task(model, task) or "")
    for fam in fams:
        d = _abs(os.path.join("data", fam, lang))
        paths += [os.path.join(d, f) for f in ("text", "prompt_text", "prompt_wav.scp")]
    return paths


def _leaf_join(model, task, lang, panel, rabs, rdir):
    """The full (unsliced) per-utterance join for one leaf; see build_utterances."""
    if not os.path.isdir(rabs):
        return {"error": f"no result dir: {rdir}", "rows": []}

    gen = load_scp(os.path.join(rabs, "wav.scp"), resolve=False, missing_ok=True)
    wer = parse_wer(os.path.join(rabs, "wav_res_ref_text.wer"))
    msr = parse_score_path(os.path.join(rabs, "msr_utmos_scores.txt"))
    spk = parse_spk(os.path.join(rabs, "spk_simi_scores.txt"))
    pause = parse_tsv(os.path.join(rabs, "pause_metadata.tsv"))

    pman = _panel_manifest(panel) if panel else None
    weights = dict((pman.get("langs") or {}).get(lang) or []) if pman else None
    ddir = os.path.join("data", panel if pman else _data_task(model, task), lang)
    target = load_text(_abs(os.path.join(ddir, "text")), missing_ok=True)
    prompt_text = load_text(_abs(os.path.join(ddir, "prompt_text")), missing_ok=True)
    prompt_scp = load_scp(_abs(os.path.join(ddir, "prompt_wav.scp")), resolve=False, missing_ok=True)

    # union of every uid we saw anywhere
    uids = set(gen) | set(wer) | set(msr) | set(spk) | set(target)
    if weights:
        uids &= set(weights)
    rows = []
    for uid in sorted(uids, key=uid_sort_key):
        w = wer.get(uid, {})
        row = {
            "uid": uid,
            "wer": w.get("wer"),
            "ins": w.get("ins"),
            "del": w.get("del"),
            "sub": w.get("sub"),
            "ref_text": w.get("ref_text"),
            "hyp_text": w.get("hyp_text"),
            "msr_utmos": msr.get(uid),
            "spk": spk.get(uid),
            "target_text": target.get(uid),
            "prompt_text": prompt_text.get(uid),
        }
        if weights:
            row["w"] = weights.get(uid)
        if uid in pause:
            row["pause_alpha"] = pause[uid].get("effective_alpha", pause[uid].get("auto_alpha"))
        if uid in gen:
            # Generated audio is <leaf>/wavs/<file> (scripts/score/lib_infer_scoring.sh). The
            # wav.scp path records where the leaf lived when it was scored (another
            # checkout, a pre-rename task dir), so only its file name is used.
            wav = os.path.join(rdir, "wavs", os.path.basename(gen[uid]))
            row["gen_url"] = "/api/audio?path=" + urllib.parse.quote(wav)
        if uid in prompt_scp:
            row["prompt_url"] = "/api/audio?path=" + urllib.parse.quote(prompt_scp[uid])
        rows.append(row)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "ZTTS2Signal/1.0"
    protocol_version = "HTTP/1.1"  # clean Content-Length framing through reverse proxies

    def log_message(self, fmt, *args):  # quieter logs
        pass

    # --- helpers ---------------------------------------------------------
    def _accepts_gzip(self):
        return "gzip" in (self.headers.get("Accept-Encoding") or "").lower()

    def _send_body(self, body, ctype, code=200, cache="no-store", allow_gzip=True):
        """Send a complete in-memory body, gzipping when the client accepts it.
        gzip keeps static assets small; JSON is sent identity (allow_gzip=False)
        because some inspection proxies mangle Content-Encoding on application/json."""
        encoding = None
        if allow_gzip and self._accepts_gzip() and len(body) > 860:
            body = gzip.compress(body, 6)
            encoding = "gzip"
        self.close_connection = True  # clean TCP EOF — dumb proxies can't mis-frame the body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("Connection", "close")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _send_json(self, obj, code=200):
        self._send_body(json.dumps(obj).encode("utf-8"),
                        "application/json; charset=utf-8", code=code, cache="no-store")

    def _send_api(self, obj, qs, code=200):
        """Return API data. With ?cb=<name> we wrap it as application/javascript
        (JSONP) so it travels the same path as app.js — which content-inspection
        proxies pass through intact, unlike large application/json bodies.

        JSONP bypasses the same-origin policy, so it is gated on a per-process
        random token (?t=) minted into index.html at serve time: our own page
        always has it, while a hostile page doing <script src=http://localhost:…>
        cannot guess it and falls through to plain JSON, which a script tag
        cannot read. Without a valid token the response is plain JSON."""
        cb = (qs.get("cb") or [None])[0]
        tok = (qs.get("t") or [None])[0]
        if cb and re.match(r"^[A-Za-z_$][\w$]*$", cb) and tok == JSONP_TOKEN:
            body = (cb + "(" + json.dumps(obj) + ");").encode("utf-8")
            self._send_body(body, "application/javascript; charset=utf-8",
                            code=code, cache="no-store", allow_gzip=True)
        else:
            self._send_json(obj, code=code)

    def _send_static(self, fname, ctype):
        path = os.path.join(HERE, fname)
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            return self._send_json({"error": "not found"}, 404)
        if fname == "index.html":
            # mint the JSONP token into the page (see _send_api)
            inject = ("<script>window.ZTTS2_TOKEN=%s;</script>"
                      % json.dumps(JSONP_TOKEN)).encode("utf-8")
            body = body.replace(b"</head>", inject + b"</head>", 1)
        self._send_body(body, ctype, cache="no-cache")

    def _safe_path(self, rel):
        """Resolve a path from a .scp (which may be repo-relative or absolute),
        refusing anything that escapes ROOT and the --allow-root dirs."""
        # some .scp files (e.g. in-the-wild) store absolute paths; others are relative.
        # realpath (not abspath) so symlinks inside ROOT cannot escape it.
        full = os.path.realpath(rel if os.path.isabs(rel) else os.path.join(ROOT, rel))
        return full if _confined(full) else None

    def _serve_audio(self, qs):
        rel = (qs.get("path") or [""])[0]
        if not rel.lower().endswith(".wav"):
            return self._send_json({"error": "only .wav"}, 400)
        full = self._safe_path(rel)
        if not full or not os.path.isfile(full):
            return self._send_json({"error": "not found", "path": rel}, 404)

        size = os.path.getsize(full)
        ctype = mimetypes.guess_type(full)[0] or "audio/wav"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng:
            # the three RFC 7233 forms: bytes=a-b, bytes=a- and the suffix bytes=-n
            m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
            if m and (m.group(1) or m.group(2)):
                partial = True
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:                                  # bytes=-n → the last n bytes
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start >= size or start > end:       # unsatisfiable → 416
                    self.close_connection = True
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    return
        length = end - start + 1

        self.close_connection = True
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Connection", "close")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(full, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    # --- routing ---------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if route in STATIC:
            fname, ctype = STATIC[route]
            return self._send_static(fname, ctype)
        def qint(name, default):
            try:
                return int((qs.get(name) or [default])[0])
            except ValueError:
                return default

        # a corrupt/truncated snapshot (read mid-rebuild) is a transient server-side
        # state → 503, so the client's retry loop re-fetches instead of caching it
        def send_maybe_503(obj):
            return self._send_api(obj, qs, code=503 if obj.get("error") else 200)

        if route == "/api/snapshot":            # full payload (curl / back-compat)
            return send_maybe_503(build_snapshot())
        if route == "/api/meta":                # tiny: total + filter option lists
            return send_maybe_503(build_meta())
        if route == "/api/rows":                # paginated snapshot rows
            return send_maybe_503(build_rows(qint("offset", 0), qint("limit", 120)))
        if route == "/api/utterances":
            model = (qs.get("model") or [""])[0]
            task = (qs.get("task") or [""])[0]
            lang = (qs.get("lang") or [""])[0]
            if not (model and lang):  # task is empty for the flat GT/whisper baselines
                return self._send_api({"error": "model and lang required"}, qs)
            panel = (qs.get("panel") or [""])[0] or None
            res = build_utterances(model, task, lang, qint("offset", 0), qint("limit", 120),
                                   panel=panel)
            return self._send_api(res, qs, code=res.pop("_code", 200))
        if route == "/api/audio":
            return self._serve_audio(qs)
        if route in ("/api/reliability", "/api/speaking_rate",
                     "/api/mpm_dist", "/api/capability_index"):
            fmap = {"/api/reliability": RELIABILITY,
                    "/api/speaking_rate": SPEAKING_RATE, "/api/mpm_dist": MPM_DIST,
                    "/api/capability_index": CAPABILITY_INDEX}
            data, err = _load_cache_json(fmap[route])
            if err == "not found":   # 200 + error object: the ensure* helpers handle it
                return self._send_api({"error": "not found"}, qs)
            if err:                  # corrupt/truncated (mid-rebuild) → transient → 503
                return self._send_api({"error": err}, qs, code=503)
            return self._send_api(data, qs)
        return self._send_json({"error": "not found", "route": route}, 404)


def main():
    global ROOT
    ap = argparse.ArgumentParser(description="ZTTS2 · SIGNAL eval explorer server")
    ap.add_argument("--port", type=int, default=8000)
    # loopback by default -- this serves the whole results/ + data/ tree, so
    # exposing it beyond the local machine should be an explicit choice.
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (use 0.0.0.0 to expose on the network)")
    ap.add_argument("--root", default=ROOT, help="repo root containing results/ and data/")
    ap.add_argument("--allow-root", action="append", default=[], metavar="DIR",
                    help="also serve files that resolve under DIR (repeatable), e.g. the "
                         "shared store results/<system> symlinks point into")
    args = ap.parse_args()
    ROOT = os.path.abspath(args.root)
    ALLOWED_ROOTS.extend(os.path.realpath(d) for d in args.allow_root)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ZTTS2 · SIGNAL  →  http://localhost:{args.port}")
    print(f"   root: {ROOT}")
    for d in ALLOWED_ROOTS:
        print(f"   also serving files resolving under: {d}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        httpd.shutdown()


if __name__ == "__main__":
    main()
