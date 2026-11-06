#!/usr/bin/env python3
"""Capability index over time: assets/plots/capability_index_time.png.

The Epoch-Capabilities-Index headline chart for this benchmark: each system's index
(build_capability_index.py) at its release date (rd_common.RELEASES), with
its 90% bootstrap interval, and the frontier — the running best among systems
scored on every language. Real speech is the dashed level at its anchor value. As in the
explorer, the y range spans the point estimates: intervals of systems scored on one or
two languages run off the edge rather than squeezing everything else flat.
"""
import datetime as dt
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUT = os.path.join(REPO_ROOT, "assets", "plots", "capability_index_time.png")
INDEX = os.path.join(SCRIPT_DIR, "_cache", "capability_index.json")

sys.path.insert(0, SCRIPT_DIR)
from rd_common import label  # noqa: E402

INK, MID, ZYPHRA = "#161616", "#9a9a95", "#e7620f"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    try:
        with open(INDEX, encoding="utf-8") as f:
            d = json.load(f)
    except OSError:
        sys.exit("plot_capability_time: capability_index.json missing -- run build_capability_index.py first")
    n_langs = d["method"]["n_langs"]
    partial = lambda s: any(s["langs"][t] < n for t, n in n_langs.items())
    pts = sorted((s for s in d["systems"] if s.get("release")),
                 key=lambda s: (s["release"], -s["index"]))
    date = lambda s: dt.date.fromisoformat(s["release"])

    frontier = []
    for s in pts:
        if not partial(s) and (not frontier or s["index"] > frontier[-1]["index"]):
            frontier.append(s)

    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    same_day = {}
    for s in pts:
        same_day.setdefault(s["release"], []).append(s)
    for s in pts:
        grp = same_day[s["release"]]
        x = mdates.date2num(date(s)) + (grp.index(s) - (len(grp) - 1) / 2) * 2.5   # fan out variants (days)
        col = ZYPHRA if s["name"].startswith("zonos") else MID if partial(s) else INK
        if s["ci"]:
            ax.plot([x, x], s["ci"], color=col, alpha=0.3, lw=1.2, zorder=2)
        ax.scatter([x], [s["index"]], s=30, zorder=3, color=col)
    if frontier:
        fx = [mdates.date2num(date(s)) for s in frontier]
        fy = [s["index"] for s in frontier]
        ax.step(fx + [mdates.date2num(date(pts[-1]))], fy + fy[-1:], where="post",
                color=INK, alpha=0.55, lw=1.5, zorder=1)
        for s, x, y, nxt in zip(frontier, fx, fy, fx[1:] + [None]):
            # right of the dot unless the next frontier point sits under the label
            left = nxt is not None and nxt - x < 60
            ax.annotate(label(s["name"]), (x, y), xytext=(-6 if left else 6, 6),
                        textcoords="offset points", ha="right" if left else "left", fontsize=8)
    real = next((s for s in d["systems"] if s["name"] == "real speech"), None)
    if real:
        ax.axhline(real["index"], ls="--", lw=1, color=ZYPHRA, zorder=1)
        ax.text(0.005, real["index"], f" real speech = {real['index']:g}", color=ZYPHRA, fontsize=8,
                va="top", transform=ax.get_yaxis_transform())

    step = 25
    lo = min(0, min(s["index"] for s in pts)) // step * step
    hi = -(-max(100, max(s["index"] for s in pts)) // step) * step
    ax.set_ylim(lo - 3, hi + 3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("public release date")
    ax.set_ylabel("capability index (higher = better)")
    ax.grid(alpha=0.25)
    ax.legend(handles=[
        Line2D([], [], color=INK, alpha=0.55, lw=1.5, label="frontier (every language)"),
        Line2D([], [], marker="o", ls="", color=INK, label="scored on every language"),
        Line2D([], [], marker="o", ls="", color=MID, label="some languages only"),
        Line2D([], [], marker="o", ls="", color=ZYPHRA, label="Zyphra"),
    ], loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=8)
    ax.set_title("ZTTS2-Eval capability index over time (90% bootstrap intervals)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
