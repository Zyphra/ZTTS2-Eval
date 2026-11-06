#!/usr/bin/env python3
"""Assemble a self-contained HTML gallery of the reliability plates rendered by
the tufte_* scripts. Reads the PNGs from assets/plots/ and embeds them as data
URIs so the page needs no external files (open it directly, or publish it as an
Artifact).

    python scripts/plots/build_reliability_gallery.py
    python scripts/plots/build_reliability_gallery.py --plots-dir assets/plots --out assets/reliability_gallery.html

Regenerate the plates first with:
    for s in p99 survival_overlay survival_smallmultiples slopegraph; do ...   # see scripts/plots/tufte_*.py
"""
import argparse
import base64
import glob
import os
import sys

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.per_uid import parse_wer  # noqa: E402


def _series_stats(raw_glob):
    """(n_langs, n_utts) counted from the same .raw inputs the plates are built
    from, so the captions can never drift from the images beside them."""
    files = sorted(glob.glob(os.path.join(REPO, raw_glob)))
    n_utts = 0
    for p in files:
        n_utts += len(parse_wer(p))
    return len(files), n_utts


def _fmt_stats(n_langs, n_utts):
    if not n_langs:
        return "inputs not present in this checkout"
    return f"{n_langs} languages · {n_utts:,} ground-truth utterances"


def plates_spec():
    """(section tag, title, blurb, [(png, caption, reading-note), ...]) — the
    dataset statistics are computed from the .raw inputs, not hard-coded."""
    clean = _series_stats("results/gt_clean/*/wav_res_ref_text.wer.raw")
    itw = _series_stats("results/gt_in-the-wild/*/wav_res_ref_text.wer.raw")
    return [
        ("A", "FLEURS-R — clean read-aloud",
         f"{_fmt_stats(*clean)}. Edits reconstructed from "
         "<code>wav_res_ref_text.wer.raw</code> with run_wer.py's own tokenization — "
         "words, or characters (CER) for zh · ja · ko · th.",
         [
            ("reliability_p99_clean.png", "p99 edit distance, by language",
             "Dot = the 99th-percentile utterance; line = bootstrap 95% CI. Thai sits "
             "far right because it is scored per character, not per word."),
            ("reliability_overlay_clean.png",
             f"Survival tails, all {clean[0] or '—'} overlaid",
             "P(edits ≥ k) on log–log. The bundle fans out in the moderate tail — that "
             "spread is the real ranking; the plate orders most → least reliable."),
            ("reliability_smallmultiples_clean.png", "Small multiples — each language against the rest",
             "One language dark, the rest faint, ordered most → least reliable. "
             "Red dots flag the catastrophic worst cases."),
            ("reliability_slopegraph_clean.png", "The ranking, column by column (worst 12)",
             "Where each language sits at ≥1, ≥3, ≥5, ≥10 edits. Line crossings are "
             "rank changes as severity rises."),
         ]),
        ("B", "VoxBlink2 — in-the-wild",
         f"{_fmt_stats(*itw)} of noisy, real-world speech. Same reconstruction, "
         "same four views.",
         [
            ("reliability_p99_in-the-wild.png", "p99 edit distance, by language", ""),
            ("reliability_overlay_in-the-wild.png",
             f"Survival tails, all {itw[0] or '—'} overlaid", ""),
            ("reliability_smallmultiples_in-the-wild.png", "Small multiples — each language against the rest", ""),
            ("reliability_slopegraph_in-the-wild.png", "The ranking, column by column (worst 12)", ""),
         ]),
        ("·", "Method — same audio, two ASR backends",
         "The generic <code>--label</code> switch turns the series axis into anything. "
         "Here it is the ASR backend on identical English ground-truth audio.",
         [
            ("reliability_backend_en.png", "Qwen3-ASR vs Whisper large-v3, English GT",
             "Qwen (blue) edges Whisper in the tail — which is exactly why English stays "
             "on Qwen under the routing rule."),
         ]),
    ], clean, itw


def uri(plots_dir, name):
    """data: URI for a plate PNG, or None when the file is absent (e.g. its
    regeneration was skipped by run_reliability.sh)."""
    path = os.path.join(plots_dir, name)
    if not os.path.isfile(path):
        print(f"warning: {path} missing — plate skipped", file=sys.stderr)
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def build(plots_dir, plates_list):
    sections, plate_no = [], 0
    for sec, title, blurb, plates in plates_list:
        figs = []
        for img, cap, note in plates:
            plate_no += 1
            note_html = f'<p class="note">{note}</p>' if note else ""
            src = uri(plots_dir, img)
            if src is None:
                mount = (f'<div class="mount"><p class="note">Plate not rendered — '
                         f'<code>{img}</code> is missing from the plots directory '
                         '(its inputs were absent when the plates were built).</p></div>')
            else:
                mount = f'<div class="mount"><img src="{src}" alt="{cap}" loading="lazy"></div>'
            figs.append(f'''<figure class="plate">
        {mount}
        <figcaption>
          <span class="pnum">Plate {plate_no:02d}</span>
          <h3>{cap}</h3>{note_html}
        </figcaption>
      </figure>''')
        sections.append(f'''<section class="set">
      <header class="set-head">
        <span class="eyebrow">Set {sec}</span>
        <h2>{title}</h2>
        <p class="blurb">{blurb}</p>
      </header>
      <div class="plates">{''.join(figs)}</div>
    </section>''')
    return "\n  ".join(sections)


PAGE = '''<title>Reliability plates — ZTTS2 GT eval</title>
<style>
  :root {{
    --paper:#FBF9F4; --ink:#2A2622; --muted:#8A8074; --faint:#B4ABA0;
    --rule:#E2DBCE; --mount:#FFFFFF; --accent:#A5321E;
    --shadow:0 1px 2px rgba(42,38,34,.06),0 8px 24px rgba(42,38,34,.08);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --paper:#1C1A17; --ink:#ECE5D8; --muted:#A79D8C; --faint:#7E7568;
      --rule:#332F29; --mount:#F6F3EC; --accent:#D7714F;
      --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.45);
    }}
  }}
  :root[data-theme="light"] {{
    --paper:#FBF9F4; --ink:#2A2622; --muted:#8A8074; --faint:#B4ABA0;
    --rule:#E2DBCE; --mount:#FFFFFF; --accent:#A5321E;
    --shadow:0 1px 2px rgba(42,38,34,.06),0 8px 24px rgba(42,38,34,.08);
  }}
  :root[data-theme="dark"] {{
    --paper:#1C1A17; --ink:#ECE5D8; --muted:#A79D8C; --faint:#7E7568;
    --rule:#332F29; --mount:#F6F3EC; --accent:#D7714F;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.45);
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
    font-family:"Iowan Old Style",Georgia,"Times New Roman",serif;
    line-height:1.5; -webkit-font-smoothing:antialiased; }}
  .mono {{ font-family:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace; }}
  code {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.86em;
    background:color-mix(in srgb,var(--faint) 18%,transparent);
    padding:.05em .35em; border-radius:3px; }}
  .wrap {{ max-width:940px; margin:0 auto; padding:clamp(2rem,5vw,5rem) clamp(1.1rem,4vw,2.5rem); }}
  header.top {{ border-bottom:1px solid var(--rule); padding-bottom:2rem; margin-bottom:3rem; }}
  .eyebrow {{ font-family:ui-monospace,Menlo,monospace; font-size:.7rem; letter-spacing:.22em;
    text-transform:uppercase; color:var(--accent); display:block; }}
  header.top h1 {{ font-size:clamp(1.9rem,4.5vw,3rem); line-height:1.08; margin:.5rem 0 0;
    font-weight:600; text-wrap:balance; letter-spacing:-.01em; }}
  header.top .lede {{ color:var(--muted); max-width:60ch; margin:1.1rem 0 0; font-size:1.02rem; }}
  header.top .meta {{ display:flex; flex-wrap:wrap; gap:.4rem 1.4rem; margin-top:1.4rem;
    font-family:ui-monospace,Menlo,monospace; font-size:.72rem; letter-spacing:.04em;
    color:var(--faint); text-transform:uppercase; }}
  .set {{ margin:0 0 3.5rem; }}
  .set-head {{ margin-bottom:1.8rem; }}
  .set-head h2 {{ font-size:1.5rem; font-weight:600; margin:.35rem 0 0; letter-spacing:-.01em; }}
  .set-head .blurb {{ color:var(--muted); max-width:66ch; margin:.6rem 0 0; font-size:.95rem; }}
  .plates {{ display:grid; gap:2.6rem; }}
  .plate {{ margin:0; }}
  .mount {{ background:var(--mount); border:1px solid var(--rule); border-radius:2px;
    padding:clamp(.5rem,2vw,1.1rem); box-shadow:var(--shadow); overflow-x:auto; }}
  .mount img {{ display:block; width:100%; height:auto; border-radius:1px; }}
  figcaption {{ display:grid; grid-template-columns:auto 1fr; column-gap:1.1rem; row-gap:.15rem;
    align-items:baseline; margin-top:.9rem; padding-left:.1rem; }}
  .pnum {{ font-family:ui-monospace,Menlo,monospace; font-size:.68rem; letter-spacing:.14em;
    text-transform:uppercase; color:var(--accent); white-space:nowrap;
    border-top:2px solid var(--accent); padding-top:.35rem; }}
  figcaption h3 {{ font-size:1.06rem; font-weight:600; margin:0; grid-column:2; }}
  figcaption .note {{ grid-column:2; color:var(--muted); font-size:.9rem; margin:.15rem 0 0; max-width:64ch; }}
  figcaption em {{ font-style:italic; color:var(--ink); }}
  footer {{ border-top:1px solid var(--rule); margin-top:2rem; padding-top:1.6rem;
    color:var(--faint); font-size:.82rem; }}
  footer .mono {{ font-size:.76rem; letter-spacing:.03em; }}
  @media (max-width:560px) {{
    figcaption {{ grid-template-columns:1fr; }}
    figcaption h3, figcaption .note {{ grid-column:1; }}
  }}
</style>

<div class="wrap">
  <header class="top">
    <span class="eyebrow">ZTTS2-Eval · ground-truth baselines</span>
    <h1>Reliability plates: the tail is the story</h1>
    <p class="lede">{n_plates} figures on the same question — how often does an utterance go badly wrong,
      and by how much? Rebuilt from the <span class="mono">tufte_*</span> scripts, now generic over
      any results tree: a “series” is a language, an ASR backend, or a model, chosen on the command line.</p>
    <div class="meta">
      <span>{langs_meta}</span><span>Edit-distance survival</span>
      <span>Bootstrap 95% CI</span><span>Tufte range-frames</span>
    </div>
  </header>
  {sections}
  <footer>
    Reconstructed edit counts match <span class="mono">scripts/score/run_wer.py</span> exactly (word units,
    or CER for zh · ja · ko · th). Rendered by <span class="mono">tufte_p99 · _survival_overlay ·
    _survival_smallmultiples · _slopegraph</span> via the shared <span class="mono">tufte_common.py</span>;
    assembled by <span class="mono">build_reliability_gallery.py</span>.
  </footer>
</div>
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots-dir", default=os.path.join(REPO, "assets", "plots"))
    ap.add_argument("--out", default=os.path.join(REPO, "assets", "reliability_gallery.html"))
    args = ap.parse_args()
    plates_list, clean, itw = plates_spec()
    n_plates = sum(len(p[3]) for p in plates_list)
    total_langs = clean[0] + itw[0]
    langs_meta = f"{total_langs} languages" if total_langs else "GT reliability"
    html = PAGE.format(sections=build(args.plots_dir, plates_list),
                       n_plates=n_plates, langs_meta=langs_meta)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
