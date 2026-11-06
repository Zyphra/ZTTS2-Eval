"""Generate ZONOS3 Mini audio in-process (zonos2.tts.llm.TTSLLM) WITH the prompt transcript, by continuation.

The checkpoint has no transcript input (the server's /tts/generate has no ref-text field), so the prompt transcript
is given by continuation:
    text   = prompt transcript + target text (one space; no separator for zh / ja), normalized as a whole;
    audio  = the 0.2 s codec-silence prefill (as on the server), then the codec frames of the prompt wav with its edge
             silence trimmed (librosa.effects.trim, top_db 35) and its last 2 frames dropped, prefilled into the audio
             stream; the LM continues from them;
    output = the audio after the prefill (silence and prompt are decoded as context, never emitted).
Why the last frames are dropped: the prompt is encoded as a clip of its own, so its final frames carry the encoder's
end-of-file edge. In training a target's audio is a slice of a whole-file encode and EOA follows its last frame; in the
speaker-embedding source (zv3sub10) 85% of targets end exactly on the file-encode's last frame, so the LM reads that
edge as "utterance over" and emits EOA on the first generated row (in-the-wild_mini: 8.6% of items with no audio
keeping every frame; 0.3% dropping 1, 0% dropping 2). The trim matters on the clean (FLEURS-R) prompts, which end in
~1-2 s of silence. --drop-prompt-frames 0 / --no-trim-prompt keep the prompt as given.
Speaker conditioning is the ownmect_s2 speaker embedding of the prompt wav (the server's speaker_mode=embedding).
--no-ref-text drops the transcript and the prompt frames (silence prefill only): the in-process twin of the server's
embedding row, for parity checks.

Sampling, language / text-normalization policy, silence prefill and output gain match scripts/gen/generate_zonos3.py
against the release server (gz.POLICY, gz.TN_LANG, the matched decoder's headroom).

Usage (inside the runtime env; see scripts/gen/run_zonos3_reftext.sh):
    python scripts/gen/generate_zonos3_reftext.py --data-dir data/clean/en \\
        --out-dir results/zonos3-rate30-20k-r2grpo-reftext_embedding/clean/en --model-path "$ZONOS3_MODEL"
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from scripts.gen import generate_zonos3 as gz  # noqa: E402
from utils.genclient import (  # noqa: E402
    apply_limit, benchmark_lang, format_error, load_test_set, pending_uids, save_wav_atomic,
    write_gen_meta_once)

SR = 44100
#: Languages whose prompt transcript and target text are joined without a separator.
NO_SPACE_JOIN = {"zh", "ja"}
SAMPLING = {k: gz.POLICY[k] for k in ("temperature", "topk", "top_p", "min_p", "repetition_window",
                                      "repetition_penalty", "repetition_codebooks", "speaker_cfg_scale")}
SILENCE_FRAMES = gz.POLICY["silence_prefill_frames"]
#: --trim-prompt threshold: librosa.effects.trim's top_db, as generate_ming3.py --trim-prompt uses it.
TRIM_TOP_DB = 35.0


def join_text(lang, prompt_text, text):
    """Prompt transcript + target text, as one raw string (the runtime normalizes it as a whole)."""
    p, t = prompt_text.strip(), text.strip()
    return p + t if lang in NO_SPACE_JOIN else f"{p} {t}"


def prefill_rows(codes, cfg):
    """Codec frames [F, C] -> int32 [F, C + 1] audio-stream rows in the training layout (sheared, WITHOUT the flushed
    tail, so the LM continues the delay pattern; text column = the text pad): zonos2.tts.prompt.silence_rows_from_codes
    for arbitrary frames."""
    import torch
    from zonos2.tts.prompt import shear

    n = int(codes.shape[0])
    sheared = shear(codes.to(torch.int32), cfg.audio_pad_id)[:n]
    return torch.cat((sheared, torch.full((n, 1), cfg.text_vocab, dtype=torch.int32)), dim=1)


def trim_edges(wav, top_db=TRIM_TOP_DB, frame_length=2048, hop_length=512):
    """[channels, samples] -> the span between the first and last non-silent frame (frame RMS within top_db of the
    loudest frame): librosa.effects.trim's defaults, in numpy (the runtime venv has no librosa)."""
    import numpy as np

    y = np.pad(wav.mean(axis=0), frame_length // 2)
    n = 1 + (len(y) - frame_length) // hop_length if len(y) >= frame_length else 0
    if n <= 0:
        return wav
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length][:n]
    power = np.maximum(np.mean(frames.astype(np.float64) ** 2, axis=1), 1e-10)
    loud = np.flatnonzero(10 * np.log10(power / power.max()) > -top_db)
    if loud.size == 0:
        return wav
    return wav[:, loud[0] * hop_length:min(wav.shape[1], (loud[-1] + 1) * hop_length)]


def main():
    ap = argparse.ArgumentParser(description="Generate ZONOS3 Mini audio in-process, continuing from the prompt "
                                             "(transcript + codec frames) with the speaker embedding.")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text', 'prompt_wav.scp', 'prompt_text'; its basename is the language")
    ap.add_argument("--out-dir", required=True, help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--model-path", default=os.environ.get("ZONOS3_MODEL"))
    ap.add_argument("--ref-text", action=argparse.BooleanOptionalAction, default=True,
                    help="continue from the prompt transcript + prompt frames (--no-ref-text: silence prefill only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1350, help="generated frames cap (90 s at 15 frames/s)")
    ap.add_argument("--batch", type=int, default=64, help="requests per engine call")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--overwrite", action="store_true", help="regenerate even if the output wav already exists")
    ap.add_argument("--trim-prompt", action=argparse.BooleanOptionalAction, default=True,
                    help="trim the prompt wav's edge silence (top_db %g) before encoding its frames (the speaker "
                         "embedding always uses the untrimmed wav)" % TRIM_TOP_DB)
    ap.add_argument("--drop-prompt-frames", type=int, default=2,
                    help="drop the last N codec frames of the prompt encoding before the prefill (they carry the "
                         "clip's end-of-file edge, which the LM reads as the end of the utterance)")
    args = ap.parse_args()
    if not args.model_path:
        sys.exit("error: no model path (set ZONOS3_MODEL or pass --model-path)")

    lang = benchmark_lang(args.data_dir)
    code = gz.TN_LANG.get(lang)
    tn = code is not None
    language = code or "en_us"

    texts, prompts, prompt_texts, uids = load_test_set(args.data_dir, need_prompt_text=args.ref_text)
    uids = apply_limit(uids, args.limit)
    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    todo = pending_uids(uids, wav_dir, args.overwrite)
    skip = len(uids) - len(todo)
    print(f"{len(uids)} items, {skip} already generated, {len(todo)} to do", flush=True)
    if not todo:
        return

    import numpy as np
    import soundfile as sf
    import torch
    from zonos2.message.tts import TTSSamplingParams
    from zonos2.tokenizer.matched_codec import codec_names, decoder_headroom_db
    from zonos2.tokenizer.vocoder import shear_up
    from zonos2.tts.llm import TTSLLM

    t0 = time.time()
    # The release server's engine flags (scripts/gen/serve_zonos3.sh).
    tts = TTSLLM(args.model_path, decode_audio=True, max_running_req=args.batch, cuda_graph_max_bs=args.batch,
                 attention_backend="fa", cache_type="naive", memory_ratio=0.6, max_seq_len_override=4096)
    if tn and tts._text_normalizer is None:
        sys.exit("error: runtime text normalization is disabled")
    mc = tts.engine.model_config
    C, size, pad = int(mc.n_codebooks), int(mc.codebook_size), int(mc.audio_pad_id)
    cfg = tts._prompt_builder.config
    vocoder = tts._vocoder
    silence = vocoder.silence_frame_codes().to(torch.int32).reshape(1, C).repeat(SILENCE_FRAMES, 1)
    hop = int(vocoder.hop_length)
    codec = str(getattr(mc, "audio_codec", "dac"))
    headroom_db = decoder_headroom_db(codec) if codec in codec_names() else 0.0
    gain = float(10.0 ** (-float(headroom_db) / 20.0))
    print(f"engine ready in {time.time() - t0:.0f}s: {codec}, hop {hop}, headroom {headroom_db} dB", flush=True)

    # Kept from the FIRST run of a resumed dir unless --overwrite regenerates.
    write_gen_meta_once(
        args.out_dir,
        args.overwrite,
        model="zonos3-mini",
        checkpoint=args.model_path,
        runtime_rev=os.environ.get("ZONOS3_RUNTIME_REV"),
        engine="in-process TTSLLM",
        speaker_mode="embedding",
        ref_text=args.ref_text,
        continuation=("prompt_text + text; prompt codec frames prefilled after the silence prefill; "
                      "output = audio after the prefill") if args.ref_text else None,
        prompt_trim_top_db=TRIM_TOP_DB if args.ref_text and args.trim_prompt else None,
        drop_prompt_frames=args.drop_prompt_frames if args.ref_text else None,
        text_join="none" if lang in NO_SPACE_JOIN else "space",
        language=language,
        text_normalization=tn,
        seed=args.seed,
        max_tokens=args.max_tokens,
        **SAMPLING,
        prepend_silence=True,
        silence_prefill_frames=SILENCE_FRAMES,
        batch=args.batch,
        max_seq_len_override=4096,
        output_headroom_db=headroom_db,
    )

    params = TTSSamplingParams(**SAMPLING, seed=args.seed, max_tokens=args.max_tokens)
    ok = fail = 0
    began = time.time()

    def failed(uid, err):
        nonlocal fail
        fail += 1
        print(f"[gen] {uid}: {err}", flush=True)

    for s in range(0, len(todo), args.batch):
        chunk = []                                  # (uid, text, embedding, prefill rows)
        for uid in todo[s:s + args.batch]:
            try:
                emb = tts.embed_speaker_file(prompts[uid])
                if args.ref_text:
                    if not prompt_texts[uid].strip():
                        raise ValueError("empty prompt transcript")
                    wav, sr = sf.read(prompts[uid], dtype="float32", always_2d=True)
                    wav = wav.T.copy()
                    if args.trim_prompt:
                        wav = np.ascontiguousarray(trim_edges(wav))
                    ref = tts.encode_reference(torch.from_numpy(wav), int(sr))
                    ref = ref[:max(1, ref.shape[0] - args.drop_prompt_frames)]
                    frames = torch.cat((silence, ref.to(torch.int32)))
                    text = join_text(lang, prompt_texts[uid], texts[uid])
                else:
                    frames, text = silence, texts[uid]
                chunk.append((uid, text, emb, prefill_rows(frames, cfg)))
            except Exception as e:
                failed(uid, format_error(e))
        if not chunk:
            continue
        try:
            res = tts.generate([c[1] for c in chunk], params, decode_audio=False, language=language,
                               text_normalization=tn, speaker_embedding=[c[2] for c in chunk],
                               audio_prefill_rows=[c[3] for c in chunk])
        except Exception as e:
            for c in chunk:
                failed(c[0], format_error(e))
            continue
        for (uid, _, _, rows), r in zip(chunk, res):
            try:
                n_prefill = int(rows.shape[0])
                gen = torch.tensor(r["audio_tokens"], dtype=torch.int32).reshape(-1, C)
                gen = torch.cat((gen, torch.full((gen.shape[0], 1), cfg.text_vocab, dtype=torch.int32)), dim=1)
                stream = torch.cat((rows, gen))
                # the scheduler counts generated rows only
                eos = None if r["eos_frame"] is None else int(r["eos_frame"]) + n_prefill
                aligned = shear_up(stream[:, :C].long(), pad)
                if eos is not None:
                    aligned = aligned[:max(0, eos)]
                valid = ((aligned >= 0) & (aligned < size)).all(dim=1)
                kept = int(valid.long().cumprod(0).sum()) if aligned.shape[0] else 0
                if kept <= n_prefill:
                    raise RuntimeError(f"no generated frames (eos_frame {r['eos_frame']}, {gen.shape[0]} generated "
                                       f"rows, {kept} valid frames, {n_prefill} prefill)")
                audio = vocoder.decode_all(aligned[:kept].unsqueeze(0).cuda(), apply_shear_up=False)[0]
                audio = audio.float().cpu().numpy().reshape(-1)
                # prefilled silence + prompt: decoded as context, never emitted (the server's prefill skip)
                save_wav_atomic(audio[n_prefill * hop:] * gain, SR, os.path.join(wav_dir, uid + ".wav"))
                ok += 1
            except Exception as e:
                failed(uid, format_error(e))
        print(f"{min(s + args.batch, len(todo))}/{len(todo)} done, elapsed {time.time() - began:.0f}s", flush=True)

    print(f"done: {ok} ok, {skip} skipped, {fail} failed -> {wav_dir}", flush=True)


if __name__ == "__main__":
    main()
