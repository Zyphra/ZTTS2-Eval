# SLURM launchers

Job bodies and submitters for running ZTTS2-Eval on a SLURM cluster: per-model
generate + score jobs, 8-GPU generation arrays, scoring and rescoring jobs, the
distributional-metric arrays, and the whole release DAG. They are a thin layer over
the regular entry points (`scripts/gen/run_*.sh`, `scripts/score/run_score.sh`,
`scripts/score/run_gt.sh`, `scripts/gen/serve_*.sh`, `scripts/score/serve_asr.sh`)
and do not change any generation or scoring logic.

Run everything from the repo root and submit with `--chdir="$(pwd -P)" --export=ALL`.
Job bodies run `.venv/bin/python`. Partition, account and QoS are never hard-coded:
they come from the submitter's flags or `METRICS_*` / `RELEASE_*` env (see
[Partition, account, and time](#partition-account-and-time)).

## Quick start

```bash
# all omni models, one 1-GPU job each (gen + score):
bash slurm/submit_eval.sh --engine omni --model all

# one omni model, one language, quick smoke test:
bash slurm/submit_eval.sh --engine omni --model qwen3-tts-0.6b \
     --task clean --test-set en --limit 3

# the In-The-Wild (VoxBlink2) task instead of Clean (FLEURS-R):
bash slurm/submit_eval.sh --engine omni --model all --task in-the-wild

# print the sbatch commands without submitting:
bash slurm/submit_eval.sh --engine omni --model all --dry-run true
```

Watch and inspect:

```bash
squeue -u $USER -o '%.18i %.9P %.28j %.8T %.10M %R'
tail -f logs/slurm/ztts-omni-qwen3-tts-0.6b-<jobid>.out
```

Outputs land in the usual place, e.g. `results/omni_<model>/clean/<lang>/wavs/`
plus the per-lang `wav_res_ref_text.wer`, `spk_simi_scores.txt`, `msr_utmos_mean.txt`.

## Files

- `lib_slurm.sh` — plumbing shared by every job body (ports, GPU count, server log
  dirs, MPS dir, claims, list-once, leaf coverage); see [below](#lib_slurmsh).
- `submit_eval.sh` / `eval_job.sh` — one 1-GPU job per model: generate, then score.
- `gen_job.sh`, `zonos3_gen_job.sh` — generation arrays on 8-GPU nodes.
- `score_job.sh` — one-GPU wrapper for any scoring driver.
- `submit_metrics.sh` / `metrics_job.sh` / `metrics_node_job.sh` /
  `metrics_finalize.sh` — the distributional metrics, system and ground truth.
- `submit_release_run.sh` — the whole release DAG.

`scripts/score/rescore_all.sh` and the control scripts under `scripts/analysis/` are
sbatch scripts too and use the same `lib_slurm.sh` plumbing.

## Generation and eval jobs

- **`submit_eval.sh`** — front door. Kaldi-style `--name value` options; engines `omni
  zonos2 kokoro cosyvoice3 moss-tts neutts vibevoice vibevoice15 audio8`. It expands
  the model list (`--model all`) and `sbatch`es one `eval_job.sh` per model. See its
  header for the full option list.
- **`eval_job.sh`** — the per-model job body. It maps the engine to its
  `scripts/gen/run_<driver>.sh`, writes to the driver's default out root
  (`gen_default_out_root`, `scripts/gen/lib_gen.sh`), then scores with
  `scripts/score/run_score.sh --decode-dir … --test-set …`. Config comes from `EJ_*`
  env set by the submitter (`EJ_OVERWRITE` regenerates existing wavs). The TTS server
  is stopped before the ASR servers start, so one GPU serves both phases.
- **`gen_job.sh`** — generation on 8-GPU nodes (array job; always submit with
  `--array`) for any driver taking `--model --task --test-set --out-root --limit
  --list-langs`. `GEN_SERVER=0` (default) is for in-process drivers such as
  `scripts/gen/run_moss_tts.sh` and `scripts/gen/run_ming3.sh`, each GPU's process
  pinned with `CUDA_VISIBLE_DEVICES`; `GEN_SERVER=1` is for drivers that start their
  own server and also take `--gpu --port --workers` (`scripts/gen/run_omni.sh`), one
  server per leaf on its GPU, run with `HF_HUB_OFFLINE=1` (download the model on a
  networked host first). The leaf languages come from the driver itself
  (`<driver> --model M --task T --list-langs true`, intersected with `GEN_LANGS`), so
  unsupported languages are never queued. The pending leaves (task × lang with a text
  uid lacking a wav) are listed once, largest first; each GPU claims the next leaf, so
  late tasks never leave a share idle. Existing wavs are skipped, a leaf still missing
  wavs is rerun once, and a requeued task releases its own unfinished claims.
  Environment: `GEN_DRIVER` and `GEN_MODEL` (required), `GEN_TASKS`, `GEN_LANGS`,
  `GEN_LIMIT`, `GEN_SERVER`, `GEN_WORKERS` (default 16), `GEN_OUT_ROOT` (default
  `gen_default_out_root <driver> GEN_MODEL`), `OMNI_VENV`. It prints a `COVERAGE
  <system> <task> <lang> <got>/<want>` line per leaf and exits 1 iff a leaf is left in
  `logs/slurm/ztts-gen-<array job>_<task>-failed.txt` (driver exited non-zero or the
  leaf ends below 98% coverage).
  ```bash
  GEN_DRIVER=scripts/gen/run_moss_tts.sh GEN_MODEL=moss-tts-v1.5 sbatch --chdir="$(pwd -P)" \
      --export=ALL --array=0-3 slurm/gen_job.sh
  GEN_DRIVER=scripts/gen/run_omni.sh GEN_MODEL=omnivoice GEN_SERVER=1 sbatch --chdir="$(pwd -P)" \
      --export=ALL --array=0-1 slurm/gen_job.sh
  ```
- **`zonos3_gen_job.sh`** — ZONOS3 Mini generation on 8-GPU nodes (array job; always
  submit with `--array`): one `scripts/gen/serve_zonos3.sh` server per GPU serving every
  mode and language it claims. The pending (mode × task × lang) leaves are listed once
  per array job in `logs/slurm/ztts-zonos3-gen-<array job>-leaves.txt`, largest first;
  each GPU walks the list and claims leaves under
  `logs/slurm/ztts-zonos3-gen-<array job>-claims/`. Existing wavs are skipped, so a
  requeue or resubmit resumes, and a dead server is restarted (at most 3 times per
  GPU). Environment: `ZONOS3_HOME`, `ZONOS3_ENV`, `Z3_TASKS` (required), `Z3_MODES`
  (default `reference embedding`), `Z3_LIMIT`, `Z3_MAX_RUNNING` (default 16),
  `Z3_SYSTEM` (default `zonos3-mini`), `Z3_LACI`, and `Z3_REVERSE=1` for a helper job
  that walks the same list from the other end. It prints a `COVERAGE <system> <task>
  <lang> <got>/<want>` line per claimed leaf (coverage below 98% is reported, not
  failed) and exits 1 iff a GPU's server never started or a leaf is left in
  `logs/slurm/ztts-zonos3-gen-<array job>_<task>-failed.txt`.
  ```bash
  Z3_TASKS="clean_mini in-the-wild_mini" sbatch --chdir="$(pwd -P)" \
      --export=ALL --array=0-0 slurm/zonos3_gen_job.sh
  ```

## Scoring, metrics and release jobs

These score existing result leaves instead of generating.

- **`score_job.sh`** — one-GPU job body that runs any scoring driver passed as its
  arguments, with its own ASR/Whisper ports (slot 0 of the job's port window) and the
  allocated GPU pinned as index 0:
  ```bash
  sbatch --chdir="$(pwd -P)" --export=ALL slurm/score_job.sh \
         bash scripts/score/run_gt.sh --task in-the-wild
  sbatch --chdir="$(pwd -P)" --export=ALL slurm/score_job.sh \
         bash scripts/score/run_score.sh --decode-dir logs/smoke/in-the-wild --test-set en
  ```
- **`scripts/score/rescore_all.sh`** — array job (8 GPUs per node, `--requeue`) that
  rescores every `results/<system>/<task>/<lang>` leaf with a stale score kind — WER with
  the routed ASR, both speaker-similarity passes, MSR-UTMOS — so every score file
  carries the current benchmark version stamp. The leaf list is built once (only leaves
  where `utils.benchmark.stale_kinds` is non-empty, largest first), and each GPU slot
  runs its own ASR and Whisper servers and claims leaves under
  `logs/slurm/rescore-all-<array job>-claims/`
  (`sbatch --array=0-3 --chdir="$(pwd -P)" --export=ALL scripts/score/rescore_all.sh`).
  The gates are version-aware, so a requeued or rerun job resumes. Leaf failures are
  listed in `logs/slurm/rescore-all-<array job>-failed.txt` and do not fail the job (a
  slot whose server cannot start does). `RESCORE_SYSTEMS="a b"` restricts the leaves
  to those `results/` dirs.
- **`submit_metrics.sh`** — one submitter for every distributional metric, system and
  ground truth. Kinds: `speaking` (char/s), `mpm` (Clean only), `rhythm` (GPU) and
  `pitch_range` (CPU; on the In-The-Wild set the same job also runs
  `scripts/prosody/paired_itw.py`), plus the ground-truth split halves `speaking_gt`
  and `mpm_gt` (Clean only; written to `results/_mpm_ground_truth/`), which are opt-in.
  Each kind is an array of `metrics_job.sh` rows, and a dependent `metrics_finalize.sh`
  merges the shards and publishes the report and explorer payloads (with
  `METRICS_FINALIZE_MODE=gt` when every requested kind is a ground-truth kind, so the
  GT speaking-rate shards are overlaid on the published payload). Environment:
  `METRICS_PARTITION` / `METRICS_ACCOUNT` / `METRICS_QOS` / `METRICS_RESERVATION`
  (site), `METRICS_KINDS` (default `speaking mpm rhythm pitch_range`),
  `METRICS_DEPENDENCY` (an `--dependency` string applied to every array),
  `METRICS_FINALIZE` (`0` skips the finalizer), `METRICS_LANGS` (only these
  languages, also for the GT kinds), `METRICS_EXCLUDE`, `GT_SPLIT_SEED` (default 0),
  `GT_SPEAKING_THROTTLE` / `GT_MPM_THROTTLE` / `GT_MPM_TIME`. The run dir is
  `logs/slurm/metrics-<run id>/`, and the last stdout line is
  `METRICS_JOBS=<id>[:<id>...]`.
  ```bash
  bash slurm/submit_metrics.sh                                  # every system kind
  METRICS_KINDS="speaking_gt mpm_gt" bash slurm/submit_metrics.sh   # GT halves
  ```
- **`metrics_node_job.sh`** — `submit_metrics.sh` with `METRICS_NODES=N` submits one
  array of N whole-node tasks instead of the per-kind arrays for the system kinds
  (where jobs asking for fewer than 5 GPUs are routed to a small partition, the 1-GPU
  arrays starve); ground-truth kinds are always arrays. Each task runs `metrics_job.sh`
  rows locally, GPU kinds one per GPU and CPU kinds `METRICS_CPU_LANES` (default 3) at
  a time; rows are claimed under `<run dir>/claims/`, so the tasks share one list. With
  `METRICS_WAIT_JOBS="<ids>"` and `METRICS_WAIT_SYSTEMS="a b"` a row starts as soon as
  those systems' leaves for its language are complete, instead of after the generation
  jobs end.
- **`submit_release_run.sh`** — the whole release DAG, run from the repo root on the
  login node: build the target recordings (`scripts/prep/build_voxblink_targets.py`),
  upload them to the HF dataset when `HF_TOKEN` is set, full rescore
  (`scripts/score/rescore_all.sh`), the two GT scorings (`scripts/score/run_gt.sh`),
  the `mpm`, `rhythm` and `pitch_range` metrics and the `mpm_gt` array (both via
  `submit_metrics.sh`) with the right dependencies, and a final job that runs
  `scripts/results_doc/run.sh` and rebuilds `RESULTS_CORESET.md`. Site values come from
  `RELEASE_PARTITION`, `RELEASE_ACCOUNT`, `RELEASE_QOS`; `RELEASE_RESCORE_NODES` (default
  4) sizes the rescore array and `RELEASE_SKIP_TARGETS=1` skips the target/upload jobs on
  a restart. Job ids are written to `logs/slurm/release-run-<timestamp>/jobs.env`. Run
  the pinned-model downloads (Qwen3-ASR, Whisper-large-v3, mHuBERT-147, ReDimNet2,
  MSR-UTMOS, Silero VAD) on the login node first if compute nodes are offline.

## `lib_slurm.sh`

Every sbatch job body sources `slurm/lib_slurm.sh` from the repo root.

**Ports.** SLURM packs several jobs onto one node, and they share the node's
`127.0.0.1`. Each job therefore owns a 40-port window keyed on its job id, and each
GPU slot of the job owns 4 ports of it, one per role:

```
slurm_port_base SLOT = 20000 + (SLURM_JOB_ID % 1000) * 40 + SLOT * 4
slurm_role_port BASE ROLE:  tts = +0   tts_aux = +1   asr = +2   whisper = +3
```

`tts` is the TTS server's HTTP port, `tts_aux` its second port (the zonos2/zonos3
IPC socket or rendezvous), `asr` the Qwen3-ASR server and `whisper` the
Whisper-large-v3 server. Single-GPU jobs use slot 0; the 8-GPU jobs use one slot per
GPU. A port is only a preference: the servers (`utils/lib_serve.sh`) never adopt a
server that is already listening, and move to an OS-chosen free port when it is busy.

**Work distribution.** `slurm_list_once` builds a job's work list once per array job
(the first task to start builds it under a `mkdir` lock; the others wait for it), and
`slurm_claim` / `slurm_claim_done` hand its entries out through atomic `mkdir` claims
recording their owner. `slurm_claims_init` releases a requeued task's unfinished
claims. `slurm_want` / `slurm_missing` give the uid-exact coverage of a leaf
(wanted text uids lacking a wav). `slurm_gpu_count` counts the node's GPUs.

## Server logs & the MPS workaround

Each job sets `SERVE_LOG_DIR=$(slurm_server_log_dir <tag>)`, so every server log of the
job (TTS, ASR, Whisper) goes to **shared** storage as
`logs/slurm/servers/<tag>/<role>.log` — `/tmp` is node-local and can't be read back
from the login node once the job moves on. The tag is the job id (`eval_job.sh`),
`score-<job id>` (`score_job.sh`), `<job>-g<gpu>` (`gen_job.sh`),
`ztts-zonos3-gen-<tag>-g<gpu>` (`zonos3_gen_job.sh`) or
`rescore-all-<array job>_<node>-s<slot>` (`rescore_all.sh`). Outside SLURM, with
`SERVE_LOG_DIR` unset, servers log to `/tmp/<role>_<port>.log`.

Each job also sets `CUDA_MPS_PIPE_DIRECTORY` to a private, daemon-less path
(`slurm_mps_dir`). Some nodes carry a **stale foreign NVIDIA MPS control daemon**
(another user's `/tmp/nvidia-mps` from a non-SLURM run); CUDA otherwise tries to join it
and dies with `Error 805: MPS client failed to connect`. Pointing at a private dir
makes CUDA drive the GPU directly.

## How GPUs are handled

On a cluster with `ConstrainDevices=yes` and `task/cgroup`, a job requesting
`--gpus=1` is cgroup-isolated to exactly one GPU, which the CUDA runtime presents as
index **0**. The server scripts default to GPU 0, so the allocated GPU is used with
no extra plumbing. The job body sets `ASR_GPU=0`/`SCORE_GPU=0` only to skip their
`nvidia-smi` free-memory scans.

## Partition, account, and time

`--partition` and `--account` are unset by default, so jobs go to the cluster's defaults; pass them (or `METRICS_PARTITION` / `METRICS_ACCOUNT` / `METRICS_QOS` for `submit_metrics.sh`, `RELEASE_PARTITION` / `RELEASE_ACCOUNT` / `RELEASE_QOS` for `submit_release_run.sh`) to target a queue. With `--time` empty, jobs get 24 h; set `--time` to change it. Override resources with `--cpus`, `--mem`, `--gpus`.

## zonos2 server

Each zonos2 job **launches its own** Zonos2 Mini-SGLang server on the allocated
GPU (`scripts/gen/serve_zonos2.sh` → `python -m zonos2`), generates through it, stops it,
then reuses the same GPU for scoring. The server code lives in a separate checkout (set
`ZONOS2_HOME` to a ZONOS2 checkout with its own venv); override via `ZONOS2_*` env or
`--model-path`. Pass `--server <URL>` to use an already-running external server
instead of launching one. Run the two modes as separate jobs with
`--accurate-mode true` (accurate/"stability") and `--accurate-mode false`
(expressive) → `results/zonos2_accurate` / `results/zonos2_expressive`.

```bash
bash slurm/submit_eval.sh --engine zonos2 --server http://<host>:<port> --task clean
```
