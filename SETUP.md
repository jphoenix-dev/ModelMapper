# Setup

Install Model Mapper and check it measures what ours measured. A person or a coding agent can follow this top to
bottom. Each step ends with a **Check**; don't go on until it passes.

**Done looks like:** `modelmapper check` ends with `PASS`, and you have its `report.txt`.

## Rules for an agent following this file
- Run the steps in order and stop at the first failed **Check**. Report the exact command and the last ~30 lines of
  output; don't paraphrase them.
- Don't edit the code to make a check pass. A failure is the result to report.
- Don't commit, push or open pull requests.
- Never type, paste or store a password or key yourself. Step 5 is for the human.
- Run `modelmapper check --apis` only if the human allows API use (about $0.005 of labeller calls with the default labeller).
- Labelling (step 7) spends money and can take hours. Always run `modelmapper label ... --plan` first and show the human
  the plan (its cost, time and disk). Pass `--yes`, `--all` or a larger `--budget` only after the human agrees to that
  plan. Never raise a budget on your own. A run that stops at its budget (exit code 3) is a result to report.

## 1. Check the machine
```bash
git --version
python --version          # Windows: py -0p  (lists installed Pythons)
nvidia-smi                # only if there is an NVIDIA GPU
```
**Check:** git and Python 3.10, 3.11 or 3.12 (3.11.9 is what we used). At least 10 GB of free disk.

## 2. Get the code and make a virtual environment
Use a short folder path: on Windows pip fails once paths pass 260 characters.
```bash
git clone https://github.com/jphoenix-dev/ModelMapper.git
cd ModelMapper
python -m venv .venv
```
Activate it: `source .venv/bin/activate` (Linux / macOS) or `.venv\Scripts\Activate.ps1` (Windows PowerShell; if
it refuses, run `Set-ExecutionPolicy -Scope Process Bypass` first).

**Check:** `python --version` inside the environment shows the version from step 1.

## 3. Install
Update pip first: the pip that comes with Python 3.10 (23.0) fails on this install ("inconsistent Name: expected
'jinja2'", then "Could not find a version that satisfies the requirement flit_core").
```bash
python -m pip install --upgrade pip
```
Then torch, from the index that matches the machine:
```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128     # NVIDIA GPU
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu       # no NVIDIA GPU (Windows / Linux)
pip install torch==2.11.0                                                         # macOS
```
Then the package (add `-e` for an editable install, and `[test]` for the unit tests):
```bash
pip install .
```
**Check:** `modelmapper --version` prints `modelmapper 0.2.0`.

## 4. Quick check (seconds)
```bash
modelmapper check --quick
```
It lists Python, the packages (against the versions we used), the GPU, the device and whether a labeller key is set
up (never the key itself).

**Check:** no FAIL lines. A package version that differs from ours is a WARN; note it in the report.

## 5. Labeller key (the human does this step)
Needed only for `map-concept`, `map-batch` and `check --apis`. **An agent must not type, paste or store the key.**

Put the key in one of these places, in the folder you run `modelmapper` from (all are gitignored):
- a line `OPENAI_API_KEY=...` in `.env` (copy `.env.example` to `.env` first);
- the key alone in a file `OAI.env`;
- a key file elsewhere, named by `OAI_KEY_FILE=<path>` in `.env`.

For another OpenAI-compatible endpoint (including a local one), also set `OPENAI_BASE_URL` and `OAI_MODEL`. It must
support JSON mode. For Azure OpenAI, use its OpenAI-compatible endpoint:
`OPENAI_BASE_URL=https://<resource>.openai.azure.com/openai/v1` (a regional endpoint such as
`https://<region>.api.cognitive.microsoft.com/openai/v1` works too), `OAI_MODEL=<your deployment name>`, and the
Azure key as `OPENAI_API_KEY` or in `OAI.env`. With a custom endpoint and no key, the check WARNs: fine for a local
server, but Azure and other hosted endpoints need the key. The judge for `steer --judge` is optional and set up the same way (`.env.example`, "Judge").

If the model is already on disk (for example `models/Qwen3-0.6B/config.json` in an earlier clone), set
`MODEL_MAPPER_MODELS` to the folder that holds `Qwen3-0.6B/` instead of downloading it again.

**Check:** `modelmapper check --quick` shows `labeller key  found`.

## 6. The full check (about 10 minutes on a CPU, 1-2 on a GPU; `--apis` adds about 25)
```bash
modelmapper check            # or, if the human allows API use:  modelmapper check --apis
```
It downloads Qwen3-0.6B (1.5 GB) the first time. If the download fails (a proxy that blocks huggingface.co), copy the
model folder from another machine and set `MODEL_MAPPER_MODELS` (step 5). It runs the answer key, an offline map of Bitcoin, and a steering
test, and compares each with our numbers (METHOD.md, "How check is calibrated").

**Check:** the last line says `PASS` (a WARN or SKIPPED inside is fine; the report says why).

## 7. Labelling (optional)
Labelling needs a labeller key (step 5) and a **judge** key, which the human puts in the same way: `JEV_API_KEY=...`
in `.env`, or a file `jevkey.env` holding the key alone or a line `JEV_API_KEY=...` or `TYPESAFE_API_KEY=...`. For another judge, set `JUDGE_BACKEND=llm` with `JUDGE_BASE_URL` and
`JUDGE_MODEL` (`.env.example`, "Judge"). What has been tested: the labeller `gpt-6-luna` with the judge Jev,
labelling Qwen3-8B and Qwen3-0.6B. Any other labeller, judge or model is untested, and results from them say so.

```bash
modelmapper check --labels       # does labelling behave here as it did for us? (about $0.55 of API calls, a few minutes)
modelmapper label L20:100 L27:3000 --model Qwen/Qwen3-0.6B --quick --with-fakes 10 --plan
```
`check --labels` compares a small sweep and a re-judged exam with our values. It also labels fake neurons, which
must fail (LABELLING.md).

`label --plan` prints what a run would do and cost without starting it: the cost as a range (measured on Qwen3-8B
and on Qwen3-0.6B, where the judge cost about 4 times more per neuron), the sweep's time on a 32 GB and on a 6 GB GPU,
and the disk. `--with-fakes` adds a second sweep for the fake neurons; the plan includes it. Without `--budget`, spend
stops at twice the plan's upper estimate. When the human agrees, run the same command without `--plan` (an agent adds
`--yes`). Re-running a stopped run resumes it.

At the end the run gives the falsifier's verdict per part, by the same rule as `check --labels`. It exits with code 4
when fake neurons pass too often (FAIL): the labels are kept, but report that verdict with them. With 20 fakes or fewer
this can be chance.

`modelmapper export-pack <labels folder> --out pack --contributor "<name>"` writes the run as a ledger pack; the model
comes from the run's results (`--model` only for files that don't record it).

**Check:** `check --labels` ends with `PASS` (or SKIPPED steps with a reason), and the plan prints.

## 8. Report
Send `check_reports/<machine>_<time>/report.txt`. It holds no keys, user names, host names or home-folder paths.

## Troubleshooting
- **No GPU found on an NVIDIA machine.** The CPU wheel was installed, or the driver is too old for CUDA 12.8. `pip
  show torch` shows `+cu128` or `+cpu`; `nvidia-smi` shows the driver.
- **`CERTIFICATE_VERIFY_FAILED` behind a corporate proxy.** Get the proxy's root certificate from IT, combine it with
  the public CA bundle into one `.pem` file, and set `PIP_CERT`, `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` to it.
- **The model download is blocked or stalls.** Download it on a machine that can (`hf download Qwen/Qwen3-0.6B
  --local-dir <dir>/Qwen3-0.6B`), copy the folder over, and set `MODEL_MAPPER_MODELS=<dir>`. Don't get around a
  network block.
- **`[WinError 6] The handle is invalid` from `py -m venv`.** Endpoint-control software on managed laptops (Avecto
  and similar) can block making or running a virtual environment outside approved folders. Clone into a folder your
  IT allows (on our test laptop, `C:\dev`) and make the environment there.
- **Out of GPU memory.** Use Qwen3-0.6B, or the CPU: `MODEL_MAPPER_DEVICE=cpu` and `MODEL_MAPPER_DTYPE=float32`.
- **Every setting** is in [.env.example](.env.example); `modelmapper settings` shows what each resolves to.
