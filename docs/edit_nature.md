# The nature of the ground-truth WER edits

The GT baselines carry a heavy WER tail on some languages (see the reliability
plates in `assets/plots/reliability_*`). Because these are *real human
recordings*, a high GT WER is an artifact of the reference ASR + scoring, not a
defect in the audio. This note quantifies **what kind** of edit the tail is made
of — how much is normalization (numbers, diacritics, spelling conventions,
word-segmentation) versus a genuine mishearing.

Reproduce with:

```
python scripts/classify_edits.py results/gt_zero_shot            # clean
python scripts/classify_edits.py results/gt_zero_shot_voxblink   # in-the-wild
```

Every utterance is re-tokenized with `run_wer.py`'s own `prepare()` (words, or
CER characters for zh/ja/ko/th), aligned with jiwer, and each substitution /
insertion / deletion is bucketed.

## Overall breakdown

| category | clean (124,951 edits) | VoxBlink2 (33,315) |
|---|--:|--:|
| content: different word (mishearing) | 33.6% | 28.6% |
| content insertion | 12.5% | 24.6% |
| content deletion | 10.9% | 17.9% |
| **minor: ≤2-char spelling / inflection** | **33.5%** | **22.5%** |
| orthographic: diacritic / ZWNJ / width *(norm.)* | 4.3% | 2.4% |
| number: digit vs word, e.g. `70`↔`siebzig` *(norm.)* | 3.4% | 2.1% |
| number: digit vs digit (misheard) | 0.5% | 0.1% |
| number insertion/deletion | 1.0% | 0.3% |
| segmentation: CJK/Thai word-boundary only *(norm.)* | 0.0% | 0.0% |
| repeated-adjacent-token ins/del | 0.3% | 1.4% |

**Unambiguously normalization-driven** (digit↔word numbers + diacritics/ZWNJ +
pure segmentation) is only **7.7%** of clean edits and **4.5%** in-the-wild.

Two results correct intuitions worth flagging:

- **Segmentation is *not* what inflates the CJK/Thai tail.** The strict test
  (utterance identical once inter-character spaces are dropped) fires on ~0% of
  edits. Thai's large p99 is genuine character-level differences — the CER
  denominator, Thai ASR *inserting* characters (41.8% of `th` edits are content
  insertions), and English loanwords transliterated into Thai script.
- **The big ambiguous bucket is `minor: ≤2-char`, and its composition is
  script-specific** — the overall 7.7% is a *floor*, not the whole normalization
  story (see below).

## Per-language tail signatures (clean set)

The ≤2-char bucket means different things in different writing systems, and that
is where the tail languages diverge:

| lang | p99 | dominant edit character |
|---|--:|---|
| **ar** | 12 | **~72% orthographic** — 30% pure diacritic + 41% ≤2-char, almost all hamza-seat / alef / ta-marbuta spelling variants (pronunciation-neutral). The most normalization-inflated language. |
| **he** | 18 | **~51% ≤2-char = matres lectionis** (ktiv male/haser: optional vav/yod). Same pronunciation, different spelling → effectively normalization. |
| **fa** | 17 | 41.6% genuine content, plus 4.7% ZWNJ (`آن‌ها`↔`آنها`) and 26% ≤2-char (mixed spelling/real). |
| **th** | 36 | **94% genuine content** (ins 41.8%, sub 30.6%, del 21.6%). Almost no normalization — a real CER tail. |
| **el / hu / bg / ru / cs / sk** | 10–16 | ≤2-char here is mostly **inflectional morphology** — a real grammatical difference (case/agreement endings), not a mishearing but not a spelling artifact either. |

## Takeaway for interpreting GT as a ceiling

- The **abjad scripts (ar, he)** — and to a lesser extent Persian's ZWNJ — carry
  a WER tail that is *largely orthographic convention*, not audio quality. Their
  GT numbers should be read with that discount, and a script-level normalization
  pass (unify hamza seats / ta-marbuta, strip ZWNJ, fold matres lectionis) would
  bring them down substantially without touching pronunciation.
- **Thai** is the opposite: its tail is real CER content divergence; no cheap
  normalization removes it. The per-character unit simply is not comparable to
  the word unit used elsewhere — compare `th` only against `th`.
- Everywhere else the tail is dominated by **genuine content mishearings** plus
  **inflectional near-misses**, i.e. the honest limit of the reference ASR on
  real speech. Digits-as-words and diacritics are a small, easily-normalized
  slice (~8% clean).

## Applied — orthographic normalization for abjad scripts

`utils/text_norm.py` now folds the pronunciation-neutral orthography of `ar`, `fa`,
`he` (gated by `ORTHO_NORM_LANGS`, default on), applied identically to reference and
hypothesis inside `run_wer.py`'s `prepare()`. Folds: strip Arabic harakat / Persian
ZWNJ / Hebrew niqqud; hamza-carriers → bare letter (`أإآٱ→ا`, `ؤ→و`, `ئ→ي`); ta-marbuta
`ة→ه`; alef-maqsura `ى→ي`; Arabic↔Persian `ي→ی`, `ك→ک`; Hebrew final forms → base.
Hebrew matres lectionis (optional vav/yod) is **deliberately not folded** — it changes
consonantal spelling and could hide a real ASR error.

Re-scoring the committed GT with these folds (offline from the stored ref/hyp via
`scripts/rescore_ortho.py`; the `ORTHO_NORM_LANGS=""` control reproduces the old WER
exactly, so the deltas are pure normalization):

| set | lang | WER before | WER after | Δ |
|---|---|--:|--:|--:|
| FLEURS-R clean | ar | 20.46 | **13.98** | −6.48 |
| FLEURS-R clean | fa | 31.18 | **29.20** | −1.98 |
| FLEURS-R clean | he | 29.49 | 29.49 | 0.00 |
| VoxBlink2 | ar | 14.19 | **12.66** | −1.53 |

Arabic drops by a third (its tail was ~72% orthographic); Persian gains from ZWNJ +
yeh/kaf unification; Hebrew is unchanged, as predicted — its tail is not safely
normalizable. These four committed `.wer` files (+ their CI footers) were refreshed;
MSR-UTMOS / spk are untouched (ASR-independent).
