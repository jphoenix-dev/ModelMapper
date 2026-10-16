"""End-to-end smoke test: every pipeline stage at toy scale, on any model and device, in a throwaway copy of the repo.

It checks that the code installs and runs on this machine: paths, device, model loading, the MLP hooks, the atlas,
attribution, ablation and log-odds, steering and every output file. It does NOT check that results are right: it
uses a toy corpus (this repo's own text) and tiny token counts.

    python scripts/smoke_e2e.py                                # offline: Qwen/Qwen3-0.6B, no API calls
    python scripts/smoke_e2e.py --model google/gemma-3-1b-it   # another model family
    python scripts/smoke_e2e.py --apis                         # also one labeller + judge round (endpoints from .env)

Your .env / environment settings apply (MODEL_MAPPER_MODELS, MODEL_MAPPER_DEVICE, MODEL_MAPPER_DTYPE, the labeller
and judge endpoints; with --apis the keys must be in .env or the environment, not OAI.env / jevkey.env).
Everything is written under --workdir (default: a new temp folder), never into this checkout's
data/ or results/. Exit code 0 = every stage passed.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# A concept small models know, with hand-written prompts, so no labeller call is needed offline.
CONCEPT = "Paris"
QUESTIONS = ["Which city is the capital of France?", "What is the French capital?", "Name the capital city of France.",
             "What is the capital and largest city of France?", "In what city does the French president live?",
             "What is the largest city in France?", "In which city is the Arc de Triomphe?", "What city is the Louvre in?",
             "In which city is the Louvre museum?", "In which city is the Notre-Dame cathedral?"]
COMPLETIONS = ["The capital of France is", "The French capital is", "The capital city of France is",
               "The capital and largest city of France is", "The French president lives in the city of",
               "The largest city in France is", "The Arc de Triomphe stands in", "The Louvre is in",
               "The Louvre museum is in", "The Notre-Dame cathedral is in"]


def toy_corpus(path: Path):
    """~1,000 short documents cut from this repo's own markdown and Python, as a FineWeb-shaped parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    docs = []
    for f in sorted(REPO.glob("*.md")) + sorted((REPO / "src").glob("*.py")):
        text = f.read_text(encoding="utf-8", errors="replace")
        docs += [text[i:i + 800] for i in range(0, len(text), 800) if len(text[i:i + 800]) > 400]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": docs}), path)
    return len(docs)


HUB_HINTS = {
    401: "the model is gated or private (accept its licence on huggingface.co, then run `hf auth login`), or the id "
         "is misspelt: the hub answers 401 for models that don't exist",
    403: "access denied. For a gated model, accept its licence. Otherwise a network proxy or firewall may be blocking "
         "huggingface.co; then copy the model folder in from elsewhere (SETUP.md, Troubleshooting)",
    404: "no such model on the hub: check the id's spelling",
}


def check_model(model: str) -> int:
    """Is the model on disk, or can its files be downloaded? Says which, with the HTTP status, before any slow stage."""
    sys.path.insert(0, str(REPO / "src"))
    from config import MODELS_DIR, model_path
    if model_path(model) != model or os.path.isdir(model):
        print(f"local copy: {model_path(model)}")
        return 0
    import httpx
    from huggingface_hub import get_token
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{model}/resolve/main/config.json"
    token = get_token()
    print(f"no local copy in {MODELS_DIR}; checking the hub ({'with' if token else 'without'} a login token)")
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {token}"} if token else {}, follow_redirects=True,
                      timeout=30)
    except httpx.HTTPError as e:
        print(f"cannot reach {endpoint}: {e!r}\nhint: offline, DNS, or certificate trouble behind a proxy "
              "(SETUP.md, Troubleshooting)")
        return 1
    print(f"GET {url} -> HTTP {r.status_code} (server: {r.headers.get('server', '?')}, "
          f"x-error-code: {r.headers.get('x-error-code', '-')})")
    if r.status_code == 200:
        return 0
    print("response starts: " + " ".join(r.text[:300].split()))
    print("hint: " + HUB_HINTS.get(r.status_code, "unexpected status; see the response above"))
    return 1


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--check-model":   # internal: run as a stage of its own
        sys.exit(check_model(sys.argv[2]))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--workdir", help="where to put the throwaway copy (default: a new temp folder)")
    ap.add_argument("--tokens", type=int, default=20000, help="tokens streamed for neuron ranges and the atlas")
    ap.add_argument("--apis", action="store_true", help="also call the labeller and judge (costs a few API calls)")
    ap.add_argument("--keep", action="store_true", help="keep the workdir afterwards")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # logs hold progress bars, token text

    work = Path(a.workdir or tempfile.mkdtemp(prefix="model_mapper_smoke_")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    for d in ("src", "prompts"):
        shutil.copytree(REPO / d, work / d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    short = a.model.split("/")[-1]
    out = work / "results" / "concepts_smoke"
    out.mkdir(parents=True, exist_ok=True)

    # this checkout's .env applies (real environment variables still win); no files with keys are copied
    sys.path.insert(0, str(REPO / "src"))
    import config
    env = {**config._read_dotenv(), **os.environ}
    # the copy's own src/ would resolve ./models inside the throwaway folder: point it at this checkout's models
    env.setdefault("MODEL_MAPPER_MODELS", config.MODELS_DIR)
    # toy scale, so small batches: fits ~6 GB GPUs next to a 1-2B model (batch size doesn't change what is computed)
    env.setdefault("MODEL_MAPPER_ATLAS_BATCH", "2048")
    env.setdefault("MODEL_MAPPER_GEN_BATCH", "25")
    env.update(PYTHONIOENCODING="utf-8", MODEL_MAPPER_MODEL=a.model, MODEL=a.model,
               MODEL_MAPPER_CORPUS=str(work / "data" / "toy_corpus.parquet"), CONCEPT_OUT_DIR=str(out),
               RECALL_VARIANT="V3", ANSWER_DISCOVERY="0")
    n_docs = toy_corpus(work / "data" / "toy_corpus.parquet")
    slug = re.sub(r"[^a-z0-9]+", "_", CONCEPT.lower()).strip("_")
    (out / f"{slug}_spec.json").write_text(json.dumps({
        "answer": " " + CONCEPT, "recall": COMPLETIONS, "recall_v2": COMPLETIONS, "recall_v3": QUESTIONS,
        "controls": [{"prompt": "The capital of Germany is", "answer": " Berlin"},
                     {"prompt": "The capital of Italy is", "answer": " Rome"}],
        "answer_forms_extra": [CONCEPT]}, indent=1), encoding="utf-8")
    (work / "prompts" / "smoke_wave.json").write_text(json.dumps({"concepts": {"smoke": [CONCEPT]}}), encoding="utf-8")

    py = sys.executable
    results = []

    def stage(name, cmd, check=None):
        log = work / f"smoke_{len(results) + 1:02d}_{re.sub(r'[^a-z0-9]+', '_', name.lower())}.log"
        t = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=work, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t
        problem = f"exit code {rc}" if rc else (check() if check else None)
        results.append((name, problem, dt))
        print(f"[{'FAIL' if problem else ' ok '}] {name:<34} {dt:7.1f}s  {problem or ''}", flush=True)
        if problem:
            print("      last lines of " + str(log) + ":")
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]:
                print("      | " + line)
        return not problem

    def exists(*paths):
        return lambda: next((f"missing {p}" for p in paths if not (work / p).exists()), None)

    print(f"Model Mapper smoke test: {a.model} | workdir {work} | toy corpus {n_docs} docs", flush=True)
    ok = stage("environment", [py, "-c", (
        "import sys, torch, transformers; sys.path.insert(0, 'src'); import neurons, config; "
        "print('python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__); "
        "print('device', neurons.DEVICE, '| dtype', neurons.DTYPE, '| model', config.DEFAULT_MODEL, '->', config.model_path(config.DEFAULT_MODEL))")])
    ok = ok and stage("model files reachable", [py, str(Path(__file__).resolve()), "--check-model", a.model])
    ok = ok and stage("neuron ranges (neuron_stats.py)", [py, "-u", "src/neuron_stats.py", a.model, str(a.tokens)],
                      exists(f"data/stats_{short}.pt"))
    ok = ok and stage("atlas (build_atlas.py)", [py, "-u", "src/build_atlas.py", a.model, str(a.tokens), "20"],
                      exists(f"data/atlas_{short}.pt", f"data/atlas_{short}_docs.npz"))
    ok = ok and stage("atlas readout + blind dossiers", [py, "-c", (
        "import sys; sys.path.insert(0, 'src'); from config import DEFAULT_MODEL; from atlas import Atlas; "
        "at = Atlas(DEFAULT_MODEL); L = at.a['top_v'].shape[0]; "
        "ex = at.examples(L // 2, 0, 'pos', n=3); assert ex and '[[' in ex[0]['text'], ex; "
        "at.write_dossiers([(L // 2, 0), (L - 1, 1)], 'results/smoke_dossiers', n=5); print(ex[0]['text'][-200:])")],
        exists("results/smoke_dossiers"))

    def wave_check():
        row = json.loads((out / "wave_free.json").read_text(encoding="utf-8")).get(CONCEPT, {})
        if row.get("status") != "ok":
            return f"concept status {row.get('status')!r} (recall {row.get('recall')}/10): the model did not recall it"
        missing = [k for k in ("k_mean", "drops", "random_drops", "steer") if k not in row]
        return f"missing fields {missing}" if missing else None
    ok = ok and stage("recall, ablation, steering (wave_free)",
                      [py, "-u", "src/wave_free.py", "prompts/smoke_wave.json"], wave_check)
    if ok and a.apis:
        ok = stage("labeller + judge endpoints", [py, "-c", (
            "import sys; sys.path.insert(0, 'src'); import oai, jev; "
            "r = oai.chat_json('Reply with the JSON object {\"ok\": true}.'); print('labeller:', r); "
            "p = jev.noul('The sky is blue today.', 'Does the text mention a colour?'); print('judge P(yes):', p); "
            "assert 0.0 <= p <= 1.0")])
        ok = ok and stage("census of 3 neurons (label_batch)", [py, "-u", "src/label_batch.py", a.model, "random:3",
                                                                 "results/smoke_census.json", "--workers", "4"],
                          exists("results/smoke_census.json"))

    total = sum(dt for _, _, dt in results)
    print(f"\n{'PASS' if ok else 'FAIL'}: {sum(p is None for _, p, _ in results)}/{len(results)} stages in {total:.0f}s"
          f" ({'offline' if not a.apis else 'with APIs'}). Logs: {work}")
    if ok and not a.keep and not a.workdir:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
