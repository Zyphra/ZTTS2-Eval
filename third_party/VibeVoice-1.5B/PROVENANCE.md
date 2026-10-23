# VibeVoice-1.5B inference code — provenance

## Why this tree exists

`microsoft/VibeVoice-1.5B` **weights** are still published on the Hub (MIT, ungated,
3 safetensors shards). The **TTS inference code** is not: Microsoft withdrew it and
rewrote the repository history. In the current upstream checkout
(`third_party/VibeVoice`, our clone of github.com/microsoft/VibeVoice):

- `docs/vibevoice-tts.md` § "Installation and Usage" reads, in full:
  *"Disabled due to widespread misuse."*
- `vibevoice/modular/modeling_vibevoice.py` ships `VibeVoiceForConditionalGeneration`
  with `forward()` only — no `generate()`, no diffusion sampling loop.
- `VibeVoiceForConditionalGenerationInference` appears in **no reachable commit**.
  History begins 2025-09-05 with a commit named `statement`; the takedown predates it.

So the model cannot be run from upstream alone. This tree supplies the missing code.

## Source

PyPI `vibevoice==0.0.1` (the only release), uploaded 2025-09-26.

    wheel   vibevoice-0.0.1-py3-none-any.whl
    sha256  fe42a8ee9adcd4fa87b2abd72be502f635bb847387e682b672022d7a9d606add

**This package is NOT published by Microsoft.** Its description says so in the first
person — *"This is a backup of the code (as of ~2 days ago). I have also backed up the
models here on Hugging Face"* — while its `author_email` is set to
`vibepod@microsoft.com`. PyPI author metadata is free-form and is **not** evidence of
origin. Treat the publisher as unknown.

## Why we trust the contents anyway

The wheel overlaps Microsoft's still-published code in 8 files. Diffed against the
current official checkout, ignoring CRLF (the wheel is CRLF, upstream is LF):

| file | differing lines |
|---|---|
| `modular/modular_vibevoice_diffusion_head.py` | **0 — identical** |
| `processor/vibevoice_processor.py` | **0 — identical** |
| `modular/streamer.py` | 4 |
| `modular/modeling_vibevoice.py` | 17 |
| `modular/modular_vibevoice_tokenizer.py` | 46 |
| `processor/vibevoice_tokenizer_processor.py` | 72 |
| `modular/modular_vibevoice_text_tokenizer.py` | 116 |
| `modular/configuration_vibevoice.py` | 146 |

Two files are byte-identical to code Microsoft ships today; the rest differ by amounts
consistent with ~5 months of upstream drift (the backup is Sep 2025, upstream has since
added the realtime and ASR variants). That is consistent with a genuine snapshot and
inconsistent with injected code.

`modeling_vibevoice_inference.py` (715 lines) has no upstream counterpart to diff, so it
was read directly. It is a `GenerationMixin` subclass: `generate()` plus
`sample_speech_tokens()`, the latter a standard classifier-free-guidance DPM-solver
sampling loop. Reviewed 2026-08-05, whole tree:

- no network calls (`socket`, `requests`, `urllib`, no URLs outside docstrings)
- no `exec`/`eval`/`__import__`/`subprocess`/`os.system`
- no `pickle.loads`/`marshal`/`base64` decoding
- no file I/O at all in the inference module
- `schedule/dpm_solver.py` carries its Apache-2.0 header and credits
  huggingface/diffusers + LuChengTHU/dpm-solver

## Why a separate tree instead of merging into `third_party/VibeVoice`

The 1.5B and the realtime 0.5B need *different* versions of the same module names —
`configuration_vibevoice.py` alone differs by 146 lines between them. Merging would
break one model to serve the other. `run_generate_vibevoice.sh` (realtime 0.5B) points
at `third_party/VibeVoice`; `run_generate_vibevoice15.sh` points here. Neither tree is
on the other's `PYTHONPATH`.

## The 7B variant — weights provenance differs, and is weaker

`run_generate_vibevoice15.sh --model 7b` runs **`vibevoice/VibeVoice-7B`**, which is
*not* a Microsoft repo. Microsoft withdrew VibeVoice-Large outright — `docs/vibevoice-tts.md`
lists its weights as `Disabled` — and this is a third-party re-upload by the same party
that published the code backup above.

The code audit above does **not** transfer to these weights. For the 1.5B the weights are
still officially published, and the vendored *code* could be diffed against files
Microsoft still ships. For the 7B there is no official copy of either, so there is
nothing to diff and no way to prove the tensors are the originals.

What can be established:

- **All 10 shards are `safetensors`, not pickle.** Loading them cannot execute code —
  safetensors is a pure tensor container. This is the reason the variant is usable at
  all; a `.bin` re-upload would be an arbitrary-code-execution risk and should have been
  refused.
- `config.json` declares the same `VibeVoiceForConditionalGeneration` architecture and
  the same tokenizer dims (`acoustic.vae_dim=64`, `semantic.vae_dim=128`) as the 1.5B,
  differing only in backbone width (`hidden_size` 3584 vs 1536) — consistent with
  Qwen2.5-7B vs 1.5B, as VibeVoice-Large was documented to be.
- Behavioural smoke on GPU (2026-08-06): `3 ok, 0 failed`, 6.27s/8.80s/8.00s for
  15/21/20-word prompts, peaks 0.35–0.68. Sane speech, no clipping.
- MIT, ungated, 18.69 GB, 12.6k downloads.

**Config quirk:** the 7B's `config.json` misspells the top-level key as `acostic_vae_dim`
(no `u`), where the 1.5B has `acoustic_vae_dim`. Harmless here —
`configuration_vibevoice.py:238` derives the value from
`acoustic_tokenizer_config.vae_dim` and never reads the top-level key — but worth knowing
before trusting any other tool that reads that field.

Treat 7B numbers as coming from **unverifiable weights**. They are reproducible and the
model behaves correctly, but "this is VibeVoice-Large" rests on the uploader's word plus
the architectural match, not on provenance.

## Local changes

CRLF → LF on all `.py` files, so the tree produces readable git diffs. No other edits.
