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
- `metrics_job.sh` / `metrics_finalize.sh`, `ground_truth_metrics_job.sh` /
  `ground_truth_metrics_finalize.sh` — array/finalizer bodies of the distributional
  metrics (below).
- `score_job.sh`, `submit_release_run.sh` — scoring and release jobs (below).

## Scoring and release jobs (benchmark version 2)

These rescore existing result leaves instead of generating. Submit every one with
`--chdir=<repo root>` (`--export=ALL`); partition, account and QoS are never
hard-coded and come from the submitter.

- **`score_job.sh`** — one-GPU job body that runs any scoring driver passed as its
  arguments, with its own ASR/Whisper port window (the same `20000 + (job id % 10000) * 4`
  scheme as `eval_job.sh`) and the allocated GPU pinned as index 0:
  ```bash
  sbatch --chdir="$(pwd -P)" --export=ALL slurm/score_job.sh \
         bash scripts/run_gt_zero_shot.sh --task zero_shot_voxblink
  sbatch --chdir="$(pwd -P)" --export=ALL slurm/score_job.sh \
         env DECODE_DIR=logs/smoke/zero_shot_voxblink bash run_infer_voxblink_eval.sh --test_set en
  ```
- **`scripts/rescore_all.sh`** — array job (8 GPUs per node, `--requeue`) that fully
  rescores every `results/<system>/<task>/<lang>` leaf — WER with the routed ASR, both
  speaker-similarity passes, MSR-UTMOS — so every score file carries the benchmark
  version stamp. Leaves are dealt round-robin over node × GPU shards
  (`sbatch --array=0-3 --chdir="$(pwd -P)" --export=ALL scripts/rescore_all.sh`), each
  shard with its own ASR and Whisper servers. The resume gates are version-aware, so a
  requeued or rerun job resumes at the first stale leaf. Leaf failures are listed in
  `logs/slurm/rescore-all-<array job>-failed.txt` and do not fail the job (a shard
  whose server cannot start does). `RESCORE_SYSTEMS="a b"` restricts the leaves to those
  `results/` dirs.
- **`submit_metrics.sh` / `submit_ground_truth_metrics.sh`** — arrays for the
  distributional metrics: `speaking` (char/s), `mpm` (Clean only), `rhythm` (GPU) and
  `pitch_range` (CPU; on the In-The-Wild set the same job also runs
  `scripts/paired_itw.py`), and for the ground-truth halves `speaking_gt`, `mpm_gt`.
  They compose through the environment: `METRICS_PARTITION` / `METRICS_ACCOUNT` /
  `METRICS_QOS` (site; `submit_metrics.sh` also takes `METRICS_RESERVATION`), `METRICS_KINDS` (which arrays to
  submit), `METRICS_DEPENDENCY` (an `--dependency` string applied to every array), and
  `METRICS_FINALIZE` (`0` skips the finalizer). The last stdout line is
  `METRICS_JOBS=<id>[:<id>...]`.
- **`metrics_node_job.sh`** — `submit_metrics.sh` with `METRICS_NODES=N` submits one
  array of N whole-node tasks instead of the per-kind arrays (where jobs asking for
  fewer than 5 GPUs are routed to a small partition, the 1-GPU arrays starve). Each task
  runs `metrics_job.sh` rows locally, GPU kinds one per GPU and CPU kinds
  `METRICS_CPU_LANES` (default 3) at a time; rows are claimed under
  `<run dir>/claims/`, so the tasks share one list. With `METRICS_WAIT_JOBS="<ids>"` and
  `METRICS_WAIT_SYSTEMS="a b"` a row starts as soon as those systems' leaves for its
  language are complete, instead of after the generation jobs end.
- **`zonos3_gen_job.sh`** — ZONOS3 Mini generation on 8-GPU nodes (array job; always
  submit with `--array`): one `scripts/serve_zonos3.sh` server per GPU serving every mode
  and language it is dealt. Leaves (mode × task × lang) go largest-first round-robin over
  array tasks × GPUs; existing wavs are skipped, so a requeue or resubmit resumes, and a
  dead server is restarted (at most 3 times per GPU). Environment: `ZONOS3_HOME`,
  `ZONOS3_ENV`, `Z3_TASKS` (required), `Z3_MODES` (default `reference embedding`),
  `Z3_LIMIT`, `Z3_MAX_RUNNING` (default 16), `Z3_SYSTEM` (default `zonos3-mini`),
  `Z3_LACI`, and `Z3_REVERSE=1` for a helper job that walks the same leaves
  smallest-first. It prints a `COVERAGE <system> <task> <lang> <got>/<want>` line per
  leaf and exits 1 iff a leaf is left in `logs/slurm/ztts-zonos3-gen-<tag>-failed.txt`.
  ```bash
  Z3_TASKS="zero_shot_mini zero_shot_voxblink_mini" sbatch --chdir="$(pwd -P)" \
      --export=ALL --array=0-0 slurm/zonos3_gen_job.sh
  ```
- **`submit_release_run.sh`** — the whole release DAG, run from the repo root on the
  login node: build the target recordings (`build_voxblink_targets.py`), upload them to
  the HF dataset when `HF_TOKEN` is set, full rescore (`rescore_all.sh`), the two GT
  scorings (`run_gt_zero_shot.sh`), the `mpm`, `rhythm`, `pitch_range` and `mpm_gt`
  metrics with the right dependencies, and a final job that runs
  `scripts/results_doc/run.sh` and rebuilds `RESULTS_CORESET.md`. Site values come from
  `RELEASE_PARTITION`, `RELEASE_ACCOUNT`, `RELEASE_QOS`; `RELEASE_RESCORE_NODES` (default
  4) sizes the rescore array and `RELEASE_SKIP_TARGETS=1` skips the target/upload jobs on
  a restart. Job ids are written to `logs/slurm/release-run-<timestamp>/jobs.env`. Run
  the pinned-model downloads (Qwen3-ASR, Whisper-large-v3, mHuBERT-147, ReDimNet2,
  MSR-UTMOS) on the login node first if compute nodes are offline.

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

`--partition` and `--account` are unset by default, so jobs go to the cluster's defaults; pass them (or `METRICS_PARTITION` / `METRICS_ACCOUNT` / `METRICS_QOS` for the metric submitters, `RELEASE_PARTITION` / `RELEASE_ACCOUNT` / `RELEASE_QOS` for `submit_release_run.sh`) to target a queue. With `--time` empty, jobs get 24 h; set `--time` to change it. Override resources with `--cpus`, `--mem`, `--gpus`.

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
