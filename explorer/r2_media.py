#!/usr/bin/env python3
"""Host the explorer's per-utterance drill-down on Cloudflare R2.

Two subcommands:

  export   for every board cell of the snapshot, writes serve.build_utterances' rows
           as utt/<model>/<board task>/<lang>.json and every generated/prompt wav they
           reference as 64 kbps mono Ogg Opus at
           audio/<canonical repo path minus .wav>.<fp>.opus (fp = size+mtime
           fingerprint, so keys are immutable), then uploads both to the (private)
           bucket. Resumable: clips already in the bucket are skipped, and cell JSON
           goes up last so no published cell links audio that is not there yet.
           --local DIR writes the same tree to disk instead of uploading.
  publish  the release step: CORS for the Pages origin, the bucket's Cloudflare-managed
           public URL (https://pub-<bucket id>.r2.dev), the ZTTS2_MEDIA_URL repository
           variable, and a Pages rebuild (which deploys only while the repo is public).

Env: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN (upload and publish), R2_BUCKET
(default ztts2-eval), RCLONE (default rclone; must be >= 1.59). The S3 keys rclone
uses are derived from the token; ~/.config/rclone/rclone.conf is not touched.

    set -a && . ./.env && set +a
    python3 explorer/r2_media.py export --allow-root /data/groups/audio/ZTTS2-Eval/results --jobs 64 --prune
    python3 explorer/r2_media.py publish            # at release; --dry-run to preview
"""

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import serve  # noqa: E402
from utils.benchmark import FLAT_PREFIXES, SOURCE_TASK, is_panel_alias  # noqa: E402

GH_REPO = "Zyphra/ZTTS2-Eval"
DEFAULT_BUCKET = "ztts2-eval"
CF_API = "https://api.cloudflare.com/client/v4"
OPUS_ARGS = ["-ac", "1", "-ar", "48000", "-c:a", "libopus", "-b:a", "64k",
             "-application", "audio", "-f", "ogg"]
AUDIO_HEADERS = ["Content-Type: audio/ogg", "Cache-Control: public, max-age=31536000, immutable"]
UTT_HEADERS = ["Content-Type: application/json", "Cache-Control: public, max-age=300"]

RCLONE = os.environ.get("RCLONE") or "rclone"
BUCKET = os.environ.get("R2_BUCKET") or DEFAULT_BUCKET


def log(msg):
    print(f"[r2_media] {msg}", flush=True)


def require_env():
    for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
        if not os.environ.get(name):
            sys.exit(f"[r2_media] {name} is not set (see .env.example)")
    return os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["CLOUDFLARE_API_TOKEN"]


# ---------------------------------------------------------------------------
# Cloudflare API
# ---------------------------------------------------------------------------
def cf(method, path, body=None):
    """One Cloudflare API call → parsed JSON. HTTP errors return the parsed error
    body with `_status` set, so callers decide what a 404/403 means."""
    req = urllib.request.Request(
        CF_API + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["CLOUDFLARE_API_TOKEN"],
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    try:
        out = json.loads(raw)
    except ValueError:
        sys.exit(f"[r2_media] {method} {path}: HTTP {status}, non-JSON: {raw[:300]!r}")
    if status >= 400:
        out["_status"] = status
    return out


def s3_env():
    """os.environ plus an env-defined rclone remote `r2:` with S3 keys derived from
    the API token (key id = token id, secret = sha256 of the token value)."""
    acct, token = require_env()
    ver = cf("GET", "/user/tokens/verify")
    if not ver.get("success"):
        ver = cf("GET", f"/accounts/{acct}/tokens/verify")
    if not ver.get("success"):
        sys.exit("[r2_media] CLOUDFLARE_API_TOKEN failed verification")
    env = dict(os.environ)
    env.update({
        "RCLONE_CONFIG_R2_TYPE": "s3",
        "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": ver["result"]["id"],
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": hashlib.sha256(token.encode()).hexdigest(),
        "RCLONE_CONFIG_R2_ENDPOINT": f"https://{acct}.r2.cloudflarestorage.com",
        "RCLONE_CONFIG_R2_NO_CHECK_BUCKET": "true",
    })
    return env


def ensure_bucket(bucket):
    acct = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    got = cf("GET", f"/accounts/{acct}/r2/buckets/{bucket}")
    if got.get("success"):
        return
    if got.get("_status") != 404:
        sys.exit(f"[r2_media] GET bucket {bucket}: {got}")
    made = cf("POST", f"/accounts/{acct}/r2/buckets", {"name": bucket})
    if not made.get("success"):
        sys.exit(f"[r2_media] create bucket {bucket}: {made}")
    log(f"created private bucket {bucket}")


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def check_tools(transcode, upload):
    if transcode:
        try:
            enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                 capture_output=True, text=True).stdout
        except OSError:
            enc = ""
        if "libopus" not in enc:
            sys.exit("[r2_media] ffmpeg with libopus required")
    if upload:
        try:
            first = subprocess.run([RCLONE, "version"], capture_output=True,
                                   text=True).stdout.splitlines()[:1]
        except OSError:
            first = []
        m = re.match(r"rclone v(\d+)\.(\d+)", first[0] if first else "")
        if not m or (int(m.group(1)), int(m.group(2))) < (1, 59):
            sys.exit("[r2_media] rclone >= 1.59 required (Cloudflare provider); "
                     "set RCLONE=/path/to/rclone")


def rclone(env, *args):
    subprocess.run([RCLONE, *args], env=env, check=True)


def remote_keys(env, bucket, prefix):
    """Every object key under `prefix` in the bucket (a missing prefix is empty)."""
    res = subprocess.run([RCLONE, "lsf", "-R", "--files-only", "--fast-list",
                          f"r2:{bucket}/{prefix}"], env=env, capture_output=True, text=True)
    if res.returncode == 3:      # directory not found
        return set()
    if res.returncode:
        sys.exit(f"[r2_media] rclone lsf {prefix}: {res.stderr.strip()[-300:]}")
    return {prefix + line for line in res.stdout.splitlines() if line}


def move_up(env, src, dst, headers):
    """Upload a staged tree and delete it locally (disk stays bounded by one chunk)."""
    if not os.path.isdir(src):
        return
    args = ["move", src, dst, "--transfers", "64", "--checkers", "64",
            "--no-check-dest", "--delete-empty-src-dirs"]
    for h in headers:
        args += ["--header-upload", h]
    rclone(env, *args)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def board_cells(models, langs):
    """(model, board task, disk task, panel, lang) for every per-language board cell —
    the same mapping openDrawer applies (isBaseline / gtTaskOf / panelOf)."""
    panels = serve.build_panels()
    cells = set()
    for r in serve.build_snapshot()["rows"]:
        if r["set"] == "ALL":
            continue
        model, lang = r["model"], r["set"]
        if (models and model not in models) or (langs and lang not in langs):
            continue
        flat = model.startswith(FLAT_PREFIXES)
        board_task = r["task"]
        if not board_task:
            board_task = next(model[len(p):] for p in FLAT_PREFIXES if model.startswith(p))
        disk_task = "" if flat else board_task
        panel = board_task if board_task in panels else None
        cells.add((model, board_task, disk_task, panel, lang))
    return sorted(cells, key=lambda c: tuple(x or "" for x in c))


def _under(path, root):
    return path == root or path.startswith(root + os.sep)


# ~4 M wav paths share a few thousand directories: resolve each directory once and
# lstat only the file, following file symlinks hop by hop through the same cache (a
# full realpath per file costs ~20 lstats on the shared FS)
@functools.lru_cache(maxsize=None)
def _realdir(d):
    return os.path.realpath(d)


@functools.lru_cache(maxsize=None)
def _panel_alias(system, task):
    return is_panel_alias(os.path.join(serve.ROOT, "results", system), task)


def _resolve(path):
    """(realpath, stat) of a file, or (None, None) when it is missing."""
    for _ in range(40):                     # symlink hops, as the kernel's limit
        src = os.path.join(_realdir(os.path.dirname(path)), os.path.basename(path))
        try:
            st = os.lstat(src)
            if not stat.S_ISLNK(st.st_mode):
                return (src, st) if stat.S_ISREG(st.st_mode) else (None, None)
            # a relative target is relative to the link's (real) directory
            path = os.path.join(os.path.dirname(src), os.readlink(src))
        except OSError:
            return None, None
    return None, None


def audio_key(rel):
    """(bucket key, source realpath) for a repo-relative or absolute wav path, or
    (None, None) when it is outside the served roots or missing. The key uses the
    canonical repo path (in-tree symlinks resolved, panel aliases collapsed onto the
    full task) plus a size/mtime fingerprint, so a regenerated wav gets a new key."""
    root = os.path.realpath(serve.ROOT)
    p = os.path.normpath(rel)
    if os.path.isabs(p):
        if _under(p, serve.ROOT):
            p = os.path.relpath(p, serve.ROOT)
        elif _under(os.path.realpath(p), root):
            p = os.path.relpath(os.path.realpath(p), root)
        else:
            return None, None
    if p == os.pardir or p.startswith(os.pardir + os.sep):
        return None, None
    src, st = _resolve(os.path.join(serve.ROOT, p))
    if not src or not serve._confined(src):
        return None, None
    if _under(src, root):
        canon = os.path.relpath(src, root)
    else:
        parts = p.split(os.sep)
        if len(parts) > 3 and parts[0] == "results" and _panel_alias(parts[1], parts[2]):
            parts[2] = SOURCE_TASK[parts[2]]
        canon = os.path.join(*parts)
    fp = hashlib.sha1(f"{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:10]
    key = "audio/" + os.path.splitext(canon)[0].replace(os.sep, "/") + "." + fp + ".opus"
    return key, src


def transcode(key, src, sink):
    """wav → Opus at <sink>/<key>; None on success, else the ffmpeg error tail."""
    dst = os.path.join(sink, key)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    res = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                          "-i", src, *OPUS_ARGS, dst + ".part"],
                         capture_output=True, text=True)
    if res.returncode:
        try:
            os.remove(dst + ".part")
        except OSError:
            pass
        return (res.stderr.strip() or f"ffmpeg exit {res.returncode}")[-300:]
    os.replace(dst + ".part", dst)
    return None


def local_keys(local):
    base = os.path.join(local, "audio")
    out = set()
    for d, _, files in os.walk(base):
        for f in files:
            if f.endswith(".opus"):
                out.add("audio/" + os.path.relpath(os.path.join(d, f), base).replace(os.sep, "/"))
    return out


def cmd_export(args):
    if args.prune and (args.model or args.lang or args.local or args.dry_run):
        sys.exit("[r2_media] --prune needs a full upload run "
                 "(not with --model/--lang/--local/--dry-run)")
    serve.ROOT = os.path.abspath(args.root)
    serve.ALLOWED_ROOTS.extend(os.path.realpath(d) for d in args.allow_root)
    stage = os.path.abspath(args.stage or os.path.join(serve.ROOT, "logs", "r2-stage"))
    upload = not args.local and not args.dry_run

    check_tools(transcode=not args.dry_run, upload=upload)
    env = None
    if upload:
        require_env()
        env = s3_env()
        ensure_bucket(BUCKET)
        shutil.rmtree(stage, ignore_errors=True)   # scratch: never upload leftovers
    sink = os.path.abspath(args.local) if args.local else stage

    # pass 1: cell JSON + the clip map
    cells = board_cells(set(args.model), set(args.lang))
    keymap, written = {}, set()   # clip key → source wav; utt keys of the cells
    no_leaf = dropped = 0
    for i, (model, board_task, disk_task, panel, lang) in enumerate(cells, 1):
        res = serve.build_utterances(model, disk_task, lang, panel=panel)
        res.pop("_code", None)
        if res.get("error") or not res.get("rows"):
            no_leaf += 1
        else:
            for row in res["rows"]:
                for field in ("gen_url", "prompt_url"):
                    if field not in row:
                        continue
                    rel = urllib.parse.unquote(row[field].split("path=", 1)[1])
                    key, src = audio_key(rel)
                    if key:
                        row[field] = key
                        keymap.setdefault(key, src)
                    else:
                        del row[field]
                        dropped += 1
            ukey = f"utt/{model}/{board_task}/{lang}.json"
            written.add(ukey)
            if not args.dry_run:
                path = os.path.join(sink, ukey)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(res, separators=(",", ":")))
        if i % 200 == 0:
            log(f"cells {i}/{len(cells)}")

    def summary(transcoded, already, failed, pruned):
        log(f"cells {len(written)} exported, {no_leaf} without a leaf; clips {len(keymap)} "
            f"referenced, {transcoded} transcoded, {already} already present, {failed} failed; "
            f"{dropped} audio links dropped; pruned {pruned}")

    if args.dry_run:
        summary(0, 0, 0, 0)
        return 0

    # pass 2: transcode (+ upload) the clips not yet in the sink, chunk by chunk
    existing = remote_keys(env, BUCKET, "audio/") if upload else local_keys(sink)
    todo = sorted(k for k in keymap if k not in existing)
    failures, done = {}, 0
    for c in range(0, len(todo), args.chunk):
        chunk = todo[c:c + args.chunk]
        with ThreadPoolExecutor(args.jobs) as ex:
            for key, err in zip(chunk, ex.map(lambda k: transcode(k, keymap[k], sink), chunk)):
                if err:
                    failures[key] = err
        if upload:
            move_up(env, os.path.join(stage, "audio"), f"r2:{BUCKET}/audio", AUDIO_HEADERS)
        done += len(chunk)
        log(f"audio {done}/{len(todo)} (failed {len(failures)})")

    # a clip that failed to transcode must not be linked from any cell
    if failures:
        for ukey in sorted(written):
            path = os.path.join(sink, ukey)
            with open(path, encoding="utf-8") as fh:
                res = json.load(fh)
            hit = False
            for row in res["rows"]:
                for field in ("gen_url", "prompt_url"):
                    if row.get(field) in failures:
                        del row[field]
                        hit = True
            if hit:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(res, separators=(",", ":")))
        for key, err in list(failures.items())[:10]:
            log(f"FAILED {key}  {keymap[key]}  {err}")

    if upload:   # cell JSON last: a published cell never links audio not yet uploaded
        move_up(env, os.path.join(stage, "utt"), f"r2:{BUCKET}/utt", UTT_HEADERS)

    pruned = 0
    if args.prune:
        if not written:
            sys.exit("[r2_media] no cell exported; refusing to prune the bucket")
        live = (set(keymap) - set(failures)) | written
        stale = sorted((remote_keys(env, BUCKET, "audio/") | remote_keys(env, BUCKET, "utt/"))
                       - live)
        if stale:
            os.makedirs(stage, exist_ok=True)
            lst = os.path.join(stage, "prune.txt")
            with open(lst, "w", encoding="utf-8") as fh:
                fh.write("".join(k + "\n" for k in stale))
            rclone(env, "delete", f"r2:{BUCKET}", "--files-from", lst)
        pruned = len(stale)

    summary(len(todo) - len(failures), len(keymap) - len(todo), len(failures), pruned)
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------
def cmd_publish(args):
    acct, _ = require_env()
    html = subprocess.run(["gh", "api", f"repos/{GH_REPO}/pages", "--jq", ".html_url"],
                          capture_output=True, text=True)
    if html.returncode or not html.stdout.strip():
        sys.exit(f"[r2_media] GitHub Pages is not enabled on {GH_REPO}: {html.stderr.strip()}")
    u = urllib.parse.urlsplit(html.stdout.strip())
    origin = f"{u.scheme}://{u.netloc}"
    bpath = f"/accounts/{acct}/r2/buckets/{BUCKET}"
    cors = {"rules": [{
        "allowed": {"origins": [origin], "methods": ["GET", "HEAD"], "headers": ["Range"]},
        "exposeHeaders": ["Content-Length", "Content-Range", "Accept-Ranges"],
        "maxAgeSeconds": 86400}]}
    def gh_cmds(media_url):
        return [["gh", "variable", "set", "ZTTS2_MEDIA_URL", "--body", media_url, "-R", GH_REPO],
                ["gh", "workflow", "run", "pages.yml", "-R", GH_REPO, "--ref", "main"]]

    if args.dry_run:
        log(f"pages origin: {origin}")
        log(f"PUT {bpath}/cors {json.dumps(cors)}")
        log(f"PUT {bpath}/domains/managed {json.dumps({'enabled': True})}  "
            f"→ <media url> = https://<result.domain>")
        for c in gh_cmds("https://<result.domain>"):
            log("would run: " + " ".join(c))
        return 0

    got = cf("PUT", f"{bpath}/cors", cors)
    if not got.get("success"):
        sys.exit(f"[r2_media] set CORS: {got}")
    log(f"CORS: GET/HEAD from {origin}")

    # the r2.dev URL Cloudflare assigns the bucket (pub-<bucket id>.r2.dev); enabling it
    # is what makes the bucket publicly readable
    got = cf("PUT", f"{bpath}/domains/managed", {"enabled": True})
    domain = (got.get("result") or {}).get("domain")
    if not got.get("success") or not domain:
        sys.exit(f"[r2_media] enable r2.dev URL: {got}")
    media_url = f"https://{domain}"
    log(f"public r2.dev URL enabled on bucket {BUCKET}: {media_url}")
    for c in gh_cmds(media_url):
        subprocess.run(c, check=True)
    log(f"media public at {media_url}; Pages rebuild dispatched "
        f"(deploys only while the repo is public)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="transcode + upload the drill-down to the bucket")
    ex.add_argument("--root", default=serve.ROOT, help="repo root holding results/ and data/")
    ex.add_argument("--allow-root", action="append", default=[], metavar="DIR",
                    help="also read wavs resolving under DIR (repeatable), as serve.py")
    ex.add_argument("--stage", help="scratch dir for the upload (default <root>/logs/r2-stage)")
    ex.add_argument("--local", metavar="DIR", help="write the tree to DIR instead of uploading")
    ex.add_argument("--model", action="append", default=[], help="only this model (repeatable)")
    ex.add_argument("--lang", action="append", default=[], help="only this language (repeatable)")
    ex.add_argument("--jobs", type=int, default=os.cpu_count(), help="parallel ffmpeg processes")
    ex.add_argument("--chunk", type=int, default=20000, help="clips per transcode+upload round")
    ex.add_argument("--dry-run", action="store_true", help="only count cells and clips")
    ex.add_argument("--prune", action="store_true",
                    help="delete bucket objects no current cell references (full runs only)")
    pub = sub.add_parser("publish", help="make the bucket public and point the site at it")
    pub.add_argument("--dry-run", action="store_true",
                     help="print the Cloudflare requests and gh commands instead of sending them")
    args = ap.parse_args()
    sys.exit(cmd_export(args) if args.cmd == "export" else cmd_publish(args))


if __name__ == "__main__":
    main()
