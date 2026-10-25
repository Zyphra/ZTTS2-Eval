"""Audio-quality scoring with MSR-UTMOS (https://github.com/sarulab-speech/msr-utmos).

CLI:
    python score_msr_utmos.py -t wav.scp -o msr_utmos_scores.txt -m msr_utmos_mean.txt

MSR-UTMOS predicts a per-listener latent and maps the mean to the 1-5 MOS scale via
mos = 2*mean(latent) + 3 (matching the reference demo). Default checkpoint is the
public w2v2_fold0 model on HuggingFace.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from sfi_utmos.model.ssl_mos import SSLMOSLightningModule

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.scp import load_scp_items as load_scp  # noqa: E402
from utils.prefetch import prefetch  # noqa: E402
# 95% CI on the mean MOS via a seeded percentile bootstrap. The canonical
# implementation lives in utils/average_wer.py (memory-chunked, same B/seed);
# ndigits=4 reproduces this scorer's historical rounding bit-for-bit.
from utils.average_wer import CI_BOOT, CI_SEED, boot_ci95  # noqa: E402
from utils.benchmark import BENCHMARK_VERSION, MSR_UTMOS_CKPT  # noqa: E402

# Pinned to a commit of the checkpoint repo (utils/benchmark.py).
DEFAULT_CKPT = MSR_UTMOS_CKPT
N_LISTENERS = 10


def load_model(ckpt=DEFAULT_CKPT, device="cpu"):
    """Load an MSR-UTMOS checkpoint, ready for score_wav / score_wave."""
    model = SSLMOSLightningModule.load_from_checkpoint(
        ckpt, map_location=device, pretrained_model_path=None)
    model.eval()
    return model


@torch.no_grad()
def score_wav(model, wav_path, device):
    wav, sr = torchaudio.load(wav_path)
    return score_wave(model, wav, sr, device)


@torch.no_grad()
def score_wave(model, wav, sr, device):
    """Score an in-memory waveform. `wav` is a (channels, time) float tensor.

    Returns the MOS on the ~1-5 scale (mos = 2*mean(listener latents) + 3)."""
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if model.condition_sr and sr not in model.sr2id:
        raise ValueError(f"sample rate {sr} not in model.sr2id {list(model.sr2id)}")

    is_sfi = getattr(model, "is_sfi", False)

    if is_sfi:
        # SFI models run at the native rate but only for rates with conv params.
        # 44.1 kHz isn't supported, so resample to the nearest supported rate >= sr.
        supported = sorted(int(r) for r in model.ssl_model.config.sfi_conv_parameters)
        if sr not in supported:
            target = next((r for r in supported if r >= sr), supported[-1])
            wav = torchaudio.functional.resample(wav, sr, target)
            sr = target

    waves = [wav.view(-1).to(device)]
    srs = torch.tensor(sr).view(1, -1).to(device)
    if model.condition_sr:
        srs = torch.stack(
            [torch.tensor(model.sr2id[s.detach().cpu().item()]) for s in srs]
        ).to(device)

    if is_sfi:
        model.ssl_model.set_sample_rate(srs[0].item())
        waves = torch.nn.utils.rnn.pad_sequence(
            [w.view(-1) for w in waves], batch_first=True
        ).to(device)
    else:
        waves = [torchaudio.functional.resample(w, sr, 16_000) for w in waves]

    # The model's forward() re-runs the SSL encoder for every listener although the
    # listener only enters after it (ssl_mos.SSLMOSLightningModule.forward: hidden +
    # listener embedding -> linear -> mean -> tanh). Encode once and apply the
    # per-listener head N_LISTENERS times: the same ops per listener as forward(), at
    # ~1/N_LISTENERS of the GPU work.
    hidden = _ssl_hidden(model, waves, srs, is_sfi)
    ratings = []
    for listener in range(1, N_LISTENERS + 1):
        listener_t = torch.tensor(listener).view(-1).to(device)
        h = hidden + model.listenr_embedding(listener_t).unsqueeze(1)
        out = torch.tanh(model.pred_liner(h).mean(dim=1)).squeeze(-1)
        ratings.append(out.cpu().item())
    return 2 * (sum(ratings) / len(ratings)) + 3


def _ssl_hidden(model, waves, srs, is_sfi):
    """forward() up to (and including) the sample-rate embedding: listener-independent."""
    if not is_sfi:
        waves = model.processor([w.detach().cpu().numpy() for w in waves],
                                return_tensors="pt", sampling_rate=16000,
                                padding=True).to(model.device)
        outputs = model.ssl_model(**waves)
    else:
        outputs = model.ssl_model(input_values=waves)
    hidden = outputs.last_hidden_state
    if model.condition_sr:
        hidden = hidden + model.sr_embedding(srs).unsqueeze(1)
    return hidden


def main():
    ap = argparse.ArgumentParser(description="Score audio quality with MSR-UTMOS.")
    ap.add_argument("-t", "--wav_scp", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-m", "--mean_out", required=True)
    ap.add_argument("--ci_out", default=None,
                    help="95%% CI file (default: sibling msr_utmos_ci.txt). The mean "
                         "file stays a bare float for backward compatibility.")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = load_model(args.ckpt, args.device)

    items = load_scp(args.wav_scp)
    scores = []
    # Decode the next files on worker threads while the GPU scores the current one.
    with open(args.out, "w") as fout:
        for (uid, wav_path), loaded, err in tqdm(
                prefetch(lambda it: torchaudio.load(it[1]), items), total=len(items)):
            try:
                if err is not None:
                    raise err
                score = score_wave(model, loaded[0], loaded[1], args.device)
            except Exception as e:
                print(f"[msr-utmos] {wav_path}: {type(e).__name__}: {e}")
                continue
            scores.append(score)
            fout.write(f"{uid}\t{wav_path}\t{score:.4f}\n")
            fout.flush()
    if not scores:
        # Do NOT leave a 0.0000 mean behind: readers treat the file's existence as
        # "scored" and a literal 0.0 poisons any average built from it (5 such
        # stubs shipped before this guard). Remove any stale mean and bail.
        for stale in (args.mean_out,):
            if os.path.isfile(stale):
                os.remove(stale)
        sys.exit(f"error: MSR-UTMOS scored 0/{len(items)} utterances -- "
                 f"no mean written")
    mean = sum(scores) / len(scores)
    with open(args.mean_out, "w") as f:
        f.write(f"{mean:.4f}\n")

    # 95% CI to a separate file so mean_out stays a single float (parsed by
    # float(open(...).read()) in build_results.py / run_full_zonos2_generic.sh).
    ci_out = args.ci_out
    if ci_out is None:
        mp = Path(args.mean_out)
        ci_out = str(mp.with_name("msr_utmos_ci.txt")) if "mean" not in mp.name \
            else str(mp.with_name(mp.name.replace("mean", "ci")))
    lo, hi = boot_ci95(scores, ndigits=4)
    with open(ci_out, "w") as f:
        f.write(f"CI95 {lo} {hi} n {len(scores)} boot {CI_BOOT} seed {CI_SEED}\n")
        # Coverage footer -- additive lines; the CI95 line above is unchanged and
        # its regex parsers (build_results.MSR_CI_RE, report_gt) match anywhere.
        f.write(f"N_SCORED: {len(scores)}\n")
        f.write(f"N_EXPECTED: {len(items)}\n")
        f.write(f"BENCHMARK_VERSION: {BENCHMARK_VERSION}\n")
        f.write(f"MSR_UTMOS_CKPT: {args.ckpt}\n")
    print(f"MSR-UTMOS mean: {mean:.4f}  95%CI [{lo}, {hi}]  (n={len(scores)}/{len(items)})")

    min_cov = float(os.environ.get("SCORE_MIN_COVERAGE", "0.98"))
    if not items or len(scores) / len(items) < min_cov:
        sys.exit(f"error: MSR-UTMOS coverage {len(scores)}/{len(items)} < {min_cov} "
                 f"(SCORE_MIN_COVERAGE) -- outputs written but the leaf is not trustworthy")


if __name__ == "__main__":
    main()
