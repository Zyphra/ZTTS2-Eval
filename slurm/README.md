# SLURM launcher for local-model eval runs

Submit local-model eval runs (the vLLM-Omni family and Zonos2) to SLURM instead of
running them by hand on whatever compute node you happen to be on. The launcher fans
the model sweep out **one job per model**, so 8 omni models run in parallel across
GPUs/nodes rather than cycling sequentially on a single GPU. Each job **generates then
scores** its model inside one 1-GPU allocation.

This is a thin layer over the existing entry points (`run_generate_*`, `run_infer_*`,
`scripts/serve_*.sh`); it does not change any generation or scoring logic.

## Quick start

Run from the repo root.

```bash
# all omni models, one 1-GPU job each (gen + score):
bash slurm/submit_eval.sh --engine omni --model all

# one omni model, one language, quick smoke test:
bash slurm/submit_eval.sh --engine omni --model qwen3-tts-0.6b \
     --task zero_shot --test_set en --limit 3

# the VoxBlink2 task instead of FLEURS-R:
bash slurm/submit_eval.sh --engine omni --model all --task zero_shot_voxblink

# print the sbatch commands without submitting:
bash slurm/submit_eval.sh --engine omni --model all --dry-run true
```

Watch and inspect:

```bash
squeue -u $USER -o '%.18i %.9P %.28j %.8T %.10M %R'
tail -f logs/slurm/ztts-omni-qwen3-tts-0.6b-<jobid>.out
```

Outputs land in the usual place, e.g. `results/omni_<model>/zero_shot/<lang>/wavs/`
plus the per-lang `wav_res_ref_text.wer`, `spk_simi_scores.txt`, `msr_utmos_mean.txt`.

## Files

- `submit_eval.sh` — front door. Kaldi-style `--name value` options; expands the omni
  model list and submits one job per model. See its header for the full option list.
- `eval_job.sh` — the per-model job body run by `sbatch`. Carries `#SBATCH` defaults
  (overridden by the submitter's CLI flags) and reads its config from `EJ_*` env vars.

## How GPUs are handled

On a cluster with `ConstrainDevices=yes` and `task/cgroup`, a job requesting
`--gpus=1` is cgroup-isolated to exactly one GPU, which the CUDA runtime presents as
index **0**. The repo's server scripts already default to GPU 0
(`serve_omni.sh` pins `CUDA_VISIBLE_DEVICES=0`; `serve_asr.sh` auto-picks and finds
only GPU 0), so the allocated GPU is used with no extra plumbing. The job body sets
`ASR_GPU=0`/`SCORE_GPU=0` only to skip their `nvidia-smi` free-memory scans.

Because the omni TTS server is stopped before the ASR server starts, a single GPU
serves both the generation and scoring phases sequentially with no contention — the
same shape as `scripts/run_full_zonos2_generic.sh`.

## Ports: why they are per-job

SLURM will pack several 1-GPU jobs onto one H100×8 node, and they share the node's
`127.0.0.1`. If every job used the default ports (`8091` omni, `8000` ASR), the
lifecycle scripts' "reuse a healthy server on this port" logic would make one job
silently reuse another job's **wrong-model** server. `eval_job.sh` therefore derives a
unique port base from `SLURM_JOB_ID`:

```
port_base = 20000 + (SLURM_JOB_ID % 10000) * 4
OMNI_PORT=port_base   ASR_PORT=port_base+1   WHISPER_PORT=port_base+2
```

Each job owns a **4-port window** `[base .. base+3]`: `base` is the omni/zonos2
HTTP port, `base+1` the ASR server (and the zonos2 server's Mini-SGLang IPC
socket), `base+2` Whisper. The stride of 4 keeps consecutive job IDs from
overlapping on a sub-port (a stride of 1 collided: zonos2 job N's IPC on
`base+1` was job N+1's HTTP port).

## Server logs & the MPS workaround

Per-job vLLM server logs (omni, ASR, Whisper) go to **shared** storage under
`logs/slurm/servers/<jobid>/` — the repo is on shared storage, whereas `/tmp`
is node-local and can't be read back from the login node once the job moves on. The
job body sets `OMNI_LOG_DIR`/`ASR_LOG`/`WHISPER_LOG` (the lifecycle scripts honor
these; the omni driver falls back to `/tmp` when `OMNI_LOG_DIR` is unset, so non-SLURM
use is unchanged).

The job also sets `CUDA_MPS_PIPE_DIRECTORY` to a private, daemon-less path. Some nodes
carry a **stale foreign NVIDIA MPS control daemon** (another user's `/tmp/nvidia-mps`
from a non-SLURM run); CUDA otherwise tries to join it and dies with `Error 805: MPS
client failed to connect`. Pointing at a private dir makes CUDA drive the GPU directly.

## Partition, account, and time

`--partition` and `--account` are unset by default, so jobs go to the cluster's defaults; pass them (or `METRICS_PARTITION` / `METRICS_ACCOUNT` for the metric submitters) to target a queue. With `--time` empty, jobs get 24 h; set `--time` to change it. Override resources with `--cpus`, `--mem`, `--gpus`.

## zonos2 server

Each zonos2 job now **launches its own** Zonos2 Mini-SGLang server on the allocated
GPU (`scripts/serve_zonos2.sh` → `python -m zonos2`), generates through it, stops it,
then reuses the same GPU for scoring — the self-managed shape as the omni engine. The
server code lives in a separate checkout (set `ZONOS2_HOME` to a ZONOS2 checkout with its own venv); override via `ZONOS2_*` env or
`--model_path`. Pass `--server <URL>` to reuse an already-running external server
instead of launching one. Run the two modes as separate jobs with
`--accurate_mode true` (accurate/"stability") and `--accurate_mode false`
(expressive) → `results/zonos2_accurate` / `results/zonos2_expressive`.

```bash
bash slurm/submit_eval.sh --engine zonos2 --server http://<host>:<port> --task zero_shot
```
