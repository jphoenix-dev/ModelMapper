# Setup and install check

A step-by-step guide to installing Model Mapper on a new machine and proving the install works. It is written so that
either a person or a coding agent (Claude Code or similar) can follow it top to bottom. Each step ends with a
**Check**; don't move on until it passes.

**Done looks like:** the smoke test prints `PASS: 6/6 stages` for a Qwen model and for a model from a second family,
the answer-key check (step 8) passes, and you have the short report from step 9.

## Rules for an agent following this file
- Run the steps in order and stop at the first failed **Check**. Report the exact command and the last ~30 lines of
  output; don't paraphrase them.
- Don't edit code under `src/` or `scripts/` to make a test pass. A failure is the result to report.
- Don't commit, push or open pull requests.
- Never type, paste or store a password or token yourself. Where a step needs one (Hugging Face login), ask the
  human to run that command.
- Skip optional steps unless the human asked for them.
- Prefer the commands as written. If one doesn't exist on this machine, say so rather than improvising a workaround.

## 1. Check the machine
```bash
git --version
python --version          # Windows: py -0p  (lists installed Pythons)
nvidia-smi                # only if there is an NVIDIA GPU
```
Model Mapper was developed on Python 3.11.9. Use 3.11 or 3.12 if one is installed; otherwise use 3.10, which is
also supported (a couple of packages then install slightly older versions). Python 3.13 and newer are untested.

Note these three facts for the report: the OS, the Python version, and the GPU. For the GPU, record the name, the
memory and the driver version from `nvidia-smi`; write "none" if there is no NVIDIA GPU, or "Apple Silicon".

**Check:** git and a Python 3.10–3.12 are available. You need at least 10 GB of free disk space.

## 2. Get the code
Clone into an ordinary folder, for example `C:\Users\<you>\Documents\GitHub` or `~/src`. Avoid deeply nested
folders: on Windows, pip fails once paths pass 260 characters.
```bash
git clone https://github.com/jphoenix-dev/ModelMapper.git
cd ModelMapper
```
If you were given a `.bundle` file instead, run `git clone -b release <path>/model-mapper.bundle model-mapper`.

**Check:** `README.md`, `src/` and `scripts/smoke_e2e.py` exist.

## 3. Create a virtual environment
Windows (PowerShell):
```powershell
py -3.11 -m venv .venv          # or -3.12 / -3.10, whichever step 1 found
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
If PowerShell refuses to run the script, run `Set-ExecutionPolicy -Scope Process Bypass` first. That only affects
this window.

Linux / macOS:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
Don't skip the pip upgrade. The pip that ships with older Pythons (e.g. 23.0 with 3.10) fails to install the CUDA
build of torch.

**Check:** `python -c "import sys; print(sys.prefix)"` prints a path ending in `.venv`.

## 4. Install PyTorch (pick one)
| Machine | Command |
|---|---|
| NVIDIA GPU, driver 570 or newer | `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128` |
| NVIDIA GPU with an older driver | Update the driver (preferred), or use the CPU line below |
| No NVIDIA GPU (Windows / Linux) | `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu` |
| Mac (Apple Silicon) | `pip install torch==2.11.0` |

Then install the rest:
```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available(), '| mps:', torch.backends.mps.is_available())"
```
**Check:** on an NVIDIA machine `cuda: True`; on a Mac `mps: True`. On CPU-only machines both are `False`, which is
expected.

## 5. Smoke test, same model family (no API keys)
```bash
python scripts/smoke_e2e.py
```
This runs every pipeline stage at toy scale with Qwen3-0.6B. First it checks that the model's files are on disk or
downloadable, and says why if not. Then:
1. neuron ranges;
2. the activation atlas;
3. blind dossiers;
4. recall, attribution, knockout and log-odds;
5. a steering grid.

It downloads Qwen3-0.6B (~1.5 GB) on first use. Typical time: about 5 min on a GPU, 12–20 min on CPU. On CPU, add
`MODEL_MAPPER_DTYPE=float32` (PowerShell: `$env:MODEL_MAPPER_DTYPE="float32"`) to make it several times faster.

**Check:** the last line starts with `PASS: 6/6 stages`. Copy the whole stage table into the report.

## 6. Smoke test, a second model family (no API keys)
This shows the code isn't tied to Qwen. SmolLM2 uses the Llama architecture and needs no login:
```bash
python scripts/smoke_e2e.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct
```
**Check:** `PASS: 6/6 stages`.

A failure in the last stage that says `the model did not recall it` means this model doesn't know enough of the test
questions about Paris. That is a limit of the test, not a bug: report it as such.

**Optional: Gemma.** Gemma is gated, so the human needs to do two things first:
1. Accept the licence at https://huggingface.co/google/gemma-3-1b-it.
2. Run `hf auth login` and paste a read token.

Then:
```bash
python scripts/smoke_e2e.py --model google/gemma-3-1b-it
```

## 7. Optional: local labeller and judge endpoints (no paid APIs)
Only do this if the human asks; it needs a local LLM server. It checks that the labeller and judge endpoint
settings work, not label quality.
1. Start a llama.cpp server with a small chat model: `llama-server -m <model.gguf> --port 8080 --jinja`.
2. Copy `.env.example` to `.env`, then set these lines:
   ```
   OPENAI_BASE_URL=http://127.0.0.1:8080/v1
   OAI_MODEL=local
   JUDGE_BACKEND=llm
   JUDGE_BASE_URL=http://127.0.0.1:8080/v1
   JUDGE_MODEL=local
   JUDGE_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}
   ```
3. Run `python scripts/smoke_e2e.py --apis`.

**Check:** `PASS: 8/8 stages`.

## 8. Verify against the answer key
The smoke test shows the code runs. This step shows it measures the same thing it measured for us. The answer key,
[answer_keys/Qwen3-0.6B.json](answer_keys/Qwen3-0.6B.json), lists marker neurons: for a few concepts, one neuron each
that was needed for recall in both of the independent prompt sets the key was built from.
```bash
python scripts/verify_answer_key.py answer_keys/Qwen3-0.6B.json --tier 1
```
Tier 1 re-runs the marker concepts on the key's own recall questions. It needs no API keys and takes about a minute
on a GPU, a few minutes on CPU.

**Check:** `PASS: tier 1, 3/3 markers found`. Each marker's rank should match the key's, and its knockout drop should
match to about ±0.02.

**Optional, only if the human allows API use** (any labeller endpoint works, including a local one, as in step 7):
- `--tier 2`: the labeller writes new recall questions. Pass: at least 2 of 3 markers found.
- `--tier 3`: the labeller labels each marker neuron blind from this install's own atlas. Build the atlas first,
  ideally from text we never used:
  ```bash
  SHARD=001_00000 bash scripts/download_corpus.sh
  MODEL_MAPPER_CORPUS=data/fineweb_001_00000.parquet python src/build_atlas.py Qwen/Qwen3-0.6B 2e6 40
  ```
  That takes a few minutes on a GPU. Close other programs using the GPU first (a local LLM server, for example): on a
  6 GB card they push the atlas into shared memory and make it several times slower.
  The label must name the concept.

With a big enough GPU (about 20 GB for Qwen3-8B), `answer_keys/Qwen3-8B.json` is the stronger test: 10 markers.

## 9. Report
Send back this, filled in:
```
Machine: <OS> | Python <x.y.z> | GPU <name, memory, driver> or none
torch <version> (cuda <True/False>, mps <True/False>) | transformers <version>
Step 5 (Qwen3-0.6B):  PASS/FAIL  <stage table>
Step 6 (SmolLM2-1.7B): PASS/FAIL <stage table>
Step 8 (answer key, tier 1): PASS/FAIL <the marker lines>
Optional (Gemma / endpoints / answer-key tiers 2-3): PASS/FAIL/skipped
Problems: <exact command + last ~30 lines of output, for anything that failed>
```
After a failed run, the smoke test keeps its working folder and prints where it is; each stage's full log is there.
After a passed run, it deletes the folder.

## Troubleshooting
- **`torch.cuda.is_available()` is False on an NVIDIA machine.** Either the CPU wheel was installed, or the driver is
  older than the CUDA 12.8 wheels need. Check with `pip show torch` (the version ends in `+cu128` or `+cpu`) and
  `nvidia-smi`.
- **`CERTIFICATE_VERIFY_FAILED` behind a corporate proxy** (Zscaler, Netskope and similar re-sign HTTPS traffic).
  Get the proxy's root certificate from IT, or export it from the Windows certificate store. Combine it with the
  public CA bundle into one `.pem` file, then point the tools at that file:
  - pip: `PIP_CERT=<bundle.pem>`;
  - model downloads and API calls (Hugging Face, httpx): `SSL_CERT_FILE=<bundle.pem>` and `REQUESTS_CA_BUNDLE=<bundle.pem>`.

  Set all three before running the smoke test.
- **The "model files reachable" stage fails with 403 and a proxy's block page.** The network blocks huggingface.co
  by policy, and certificates won't help. Don't try to get around it. Ask whoever runs the network, or copy each
  model folder (`config.json`, tokenizer files, `*.safetensors`) in from a machine that can download it, into
  `models/<name>`, if your organisation allows that.
- **Model download stalls or fails.** Download it explicitly, then point the code at the folder:
  ```bash
  hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
  ```
  The default models folder is `./models`; set `MODEL_MAPPER_MODELS` to use another.
- **`Filename too long` or pip `OSError` on Windows.** The clone or `.venv` is in too deep a folder. Re-clone
  somewhere shorter.
- **Out of GPU memory.** The smoke test already uses small batches (`MODEL_MAPPER_ATLAS_BATCH=2048`,
  `MODEL_MAPPER_GEN_BATCH=25`). If it still runs out:
  1. halve those values;
  2. use a smaller model;
  3. run on CPU with `MODEL_MAPPER_DEVICE=cpu` and `MODEL_MAPPER_DTYPE=float32`.

  Other apps' windows also take GPU memory on Windows laptops, so close what you can. For full-size runs on a small
  GPU, set the same two batch variables in `.env`.
- **Other settings** (model folder, device, dtype, endpoints) are all in [.env.example](.env.example).

## After the check
- To reproduce the published experiments (Qwen3-8B, the full FineWeb shard, about a day on a 32 GB GPU, and paid
  APIs), see [README.md](README.md), "Main pipelines".
- To map another model, see README, "Use it on another model or endpoint".
