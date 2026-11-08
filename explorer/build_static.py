#!/usr/bin/env python3
"""Render the explorer as a static site for GitHub Pages.

The page is serve.py's index.html/css/js, marked with `window.ZTTS2_STATIC` (and
`window.ZTTS2_MEDIA` when --media-url is given: the R2 base holding the per-utterance
drill-down, see explorer/r2_media.py). Each aggregate route `GET /api/<name>` becomes
`api/<name>.json` beside the page, produced by serve.py's own builders so the static
payloads equal the server's.

    python explorer/build_static.py --out _site [--media-url https://…]
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import serve  # noqa: E402


def payloads():
    """{name: payload} for every pre-rendered api/<name>.json; exits on a missing or
    unusable snapshot so CI never publishes an empty board."""
    meta = serve.build_meta()
    snap = serve.build_snapshot()
    if meta.get("error") or not snap["rows"]:
        sys.exit(f"[build_static] no usable {serve.SNAPSHOT}: {meta.get('error') or 'no rows'}")
    out = {"meta": meta, "snapshot": snap}
    for route, fname in serve.VIEW_FILES.items():
        data, err = serve._load_cache_json(fname)
        if err:
            sys.exit(f"[build_static] {fname}: {err}")
        out[route.rsplit("/", 1)[1]] = data
    return out


def page_mark(media_url):
    """The <script> injected before </head>: static mode plus the optional media base."""
    media_url = media_url.rstrip("/")
    if media_url and not re.fullmatch(
            r"https?://[A-Za-z0-9.\-]+(:\d+)?(/[A-Za-z0-9._~\-/]*)?", media_url):
        sys.exit(f"[build_static] bad --media-url: {media_url!r}")
    mark = b"<script>window.ZTTS2_STATIC=true;"
    if media_url:
        mark += b"window.ZTTS2_MEDIA=" + json.dumps(media_url).encode() + b";"
    return mark + b"</script>"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(serve.ROOT, "_site"),
                    help="output directory (created; existing files are overwritten in place)")
    ap.add_argument("--media-url", default="",
                    help="R2 base URL of the per-utterance drill-down (explorer/r2_media.py)")
    args = ap.parse_args()

    # everything that can fail runs before the first write
    data = payloads()
    mark = page_mark(args.media_url)

    out = args.out
    os.makedirs(os.path.join(out, "api"), exist_ok=True)
    files = sorted({f for f, _ in serve.STATIC.values()})
    for fname in files:
        with open(os.path.join(HERE, fname), "rb") as fh:
            body = fh.read()
        if fname == "index.html":
            if b"</head>" not in body:
                sys.exit("[build_static] explorer/index.html has no </head>")
            body = body.replace(b"</head>", mark + b"</head>", 1)
        with open(os.path.join(out, fname), "wb") as fh:
            fh.write(body)
    for name, obj in data.items():
        with open(os.path.join(out, "api", name + ".json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, separators=(",", ":")))
    media = args.media_url.rstrip("/") or "none"
    print(f"[build_static] {len(files)} page files + {len(data)} api/*.json → {out} "
          f"(media: {media})")


if __name__ == "__main__":
    main()
