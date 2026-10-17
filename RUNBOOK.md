# Runbook: distributed neuron map (RunPod)

The code is in `src/dist/`. A run lives in one store, either a local folder or an S3-compatible bucket. The store holds the
config, corpus shards, task manifest, results, claims and logs. Workers are interruptible RunPod Community pods.
Each one runs `src/dist/worker.py`: it takes pending tasks at random until none are left, then terminates its own
pod. The launcher runs on the desktop. It keeps N pods alive, replaces interrupted ones and enforces the spend cap.

| File | Does |
|---|---|
| `storage.py` | The store: local folder or `s3://bucket/prefix` (R2, AWS, …) |
| `make_run.py` | Writes config (model and pinned revision), corpus shards, promptset and manifest |
| `tasks.py` | `run_task`: `observe:<shard>`, `merge:observe`, `ablate:<L>:<a>-<b>:<ps>` |
| `manifest.py` | Pending, ready and claimed state (no torch) |
| `worker.py` | Container entrypoint: the worker loop, logs, self-termination |
| `launcher.py` | `run`, `status`, `terminate`, `billing` |
| `runpod_api.py` | REST calls: create, list, terminate |
| `compare_runs.py` | Two runs side by side, e.g. 5090 vs rented GPU, or merge vs single shard |

## Credentials

Put these in `.env` (gitignored) or the environment. Never put them in the image or in git.

- `RUNPOD_API_KEY`: used on the desktop only. Pods get their own pod-scoped key from RunPod.
- `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`: an R2 token limited to the one bucket. The launcher
  passes these to pods as env vars.
- `HF_TOKEN`: optional, for faster hub downloads.

## 1. Set up a run

```
python src/dist/make_run.py --store s3://<bucket>/<run> --model Qwen/Qwen3-8B \
    --shard-tokens 2000000 --n-shards 5 --k 40 --batch-tokens 4096 --ablate 18:0-63 --window 128
```

The revision defaults to the pinned hash in `make_run.py` (Qwen3-8B `b968826d…`). Every result records the model,
revision, dtype, GPU, driver, CUDA, torch and git commit in `results/<task>/meta.json`.

## 2. Test locally first (5090, or `MODEL_MAPPER_DEVICE=cpu DIST_DTYPE=float32` for a CPU smoke test)

```
python src/dist/worker.py --store <store> --no-terminate            # all tasks
python src/dist/worker.py --store <store> --task observe:shard_00000 --no-terminate
```

A local copy in `$MODEL_MAPPER_MODELS` is used only if its safetensors match the pinned revision's sha256.

## 3. Code on the pods (no custom image needed)

By default pods start from the public `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` image, which has the same torch
and CUDA as the local `.venv`. The launcher uploads the committed worker code (`git archive` of `src/dist`,
`src/config.py`, `src/neurons.py`, `requirements-dist.txt`) to `code/<commit>.tar.gz` in the bucket. Each pod's start
command downloads it, installs the pinned packages and runs the worker. If any boot step fails, the pod terminates
itself, and after two failed boots the launcher stops. The launcher refuses to run with uncommitted worker code.

Optional: `Dockerfile` builds an image with the code baked in. Use it with `--custom-image --image <registry>/<image>`.

## 4. Launch

```
python src/dist/launcher.py run --store <store> --gpu "NVIDIA GeForce RTX 4090" \
    --pods 3 --max-pods 3 --max-spend 5 --interval 120 --drain-to D:/model-mapper-runs/<run>
```

- Pods are Community Cloud, interruptible, 40 GB container disk, `volumeInGb 0`, no network volume, one GPU type.
- Spend is estimated as each pod's `costPerHr` × its lifetime. The launcher won't start a pod that could take the
  estimate past `--max-spend` within one interval, and terminates everything if the cap is reached.
- `--max-launches` (default 20) caps the total number of pods started, replacements included.
- Ctrl-C terminates every pod of the run.

## 5. Monitor

```
python src/dist/launcher.py status --store <store>
```

This lists the pods plus pending, ready and claimed tasks. In the store:

- `logs/<pod>/boot.json`: model ready time, first task start;
- `logs/<pod>/task_*.json`: seconds, tokens, tokens/s per task;
- `logs/<pod>/heartbeat.json`, `logs/<pod>/error_*.json`;
- `logs/launcher.json`: the spend ledger.

Afterwards, `python src/dist/timeline.py <store>` shows where each pod's time and money went: create call,
scheduling and image pull, each boot step, model download (GB, MB/s), load onto the GPU, every task (compute vs input
download vs result upload, tokens/s, peak GPU memory) and teardown, each with its dollar cost at the pod's rate.

## 6. Stop

```
python src/dist/launcher.py terminate --store <store>   # terminates every pod of the run, then lists pods (expect 0)
```

Every session ends with this, or with the launcher's own final listing, showing zero pods. Then check the RunPod
billing page, or `launcher.py billing`.

## 7. Merge

`merge:observe` runs as a task once every observe shard is done, on a worker or locally with `worker.py --task
merge:observe`. Ablate tasks wait for it. To compare two runs:

```
python src/dist/compare_runs.py <store A> <store B> --out compare.json
```

## R2 free tier (automatic caps, `quota.py`)

Each month R2 gives 10 GB-month of storage, 1M Class A requests (writes, listings) and 10M Class B requests (reads)
free. Deletes are free.

- **Storage.** Storage is averaged over the month, so what counts is how much sits in the bucket. Run the launcher with
  `--drain-to D:/model-mapper-runs/<run>`: finished results nothing else needs are copied there and deleted from the
  bucket, while the small `meta.json` and `_DONE` files stay. Bytes held are capped at `DIST_MAX_GB` (default 9).
- **Requests.** These add up over the calendar month, and deleting refunds nothing. Each process saves its counts to
  `usage/<YYYY-MM>/`. The caps are `DIST_MAX_CLASS_A=900000` and `DIST_MAX_CLASS_B=9000000`.
- **Enforcement.** A worker checks before every task and stops if the task would pass a cap. The launcher checks
  every interval and terminates everything if a cap is reached.
- **Assumption.** This assumes this bucket is the account's only R2 use, because the free tier is per account.

## Behaviour to know

- **Restartable tasks.** A task is done only when `results/<task>/_DONE` exists, and that marker is written last. A
  killed task is simply recomputed. Duplicate runs are harmless.
- **Soft claims.** `claims/<task>` steers workers apart and expires after 45 minutes, so an interrupted worker's
  task gets picked up again.
- **Self-termination.** A worker with nothing left to do terminates its pod with `runpodctl pod delete $RUNPOD_POD_ID`,
  falling back to REST `DELETE`. The launcher also terminates finished and stopped pods as a backstop. It is not yet
  verified that the pod-scoped key can delete its own pod; phase C checks this.
