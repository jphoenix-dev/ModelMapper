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
- Run `modelmapper check --apis` only if the human allows API use (under a cent of labeller calls with the default labeller).

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
**Check:** `modelmapper --version` prints `modelmapper 0.1.0`.

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
`OPENAI_BASE_URL=https://<resource>.openai.azure.com/openai/v1`, `OAI_MODEL=<your deployment name>`, and the Azure
key as `OPENAI_API_KEY`. The judge for `steer --judge` is optional and set up the same way (`.env.example`, "Judge").

If the model is already on disk (for example `models/Qwen3-0.6B/config.json` in an earlier clone), set
`MODEL_MAPPER_MODELS` to the folder that holds `Qwen3-0.6B/` instead of downloading it again.

**Check:** `modelmapper check --quick` shows `labeller key  found`.

## 6. The full check (about 10 minutes on a CPU; `--apis` adds about 25)
```bash
modelmapper check            # or, if the human allows API use:  modelmapper check --apis
```
It downloads Qwen3-0.6B (1.5 GB) the first time. It runs the answer key, an offline map of Bitcoin, and a steering
test, and compares each with our numbers (METHOD.md, "How check is calibrated").

**Check:** the last line says `PASS` (a WARN or SKIPPED inside is fine; the report says why).

## 7. Report
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
