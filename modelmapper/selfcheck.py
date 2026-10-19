"""modelmapper check: does this install work, and does it measure what ours measured? Writes a report to send back.

    modelmapper check                 # environment + offline checks on Qwen3-0.6B   (no API calls; ~5-15 min on a CPU)
    modelmapper check --apis          # also live map-concept for the answer key's 3 concepts (labeller calls; under a cent)
    modelmapper check --quick         # environment only (seconds)
    modelmapper check --only map      # just these steps: key1, map, steer, live, judge

Every item is PASS, FAIL, WARN (works, but differs from ours or varies run to run) or SKIPPED (with the reason), next
to the value our machine got (CALIBRATION below). The report is written as text and JSON to
check_reports/<machine id>_<time>/ (or --out). It holds no keys, user name, host name, home-folder paths or endpoint
error bodies (the machine id is a short hash); send report.txt back if something fails.

Steps, all on Qwen3-0.6B (it needs the model files: the Hugging Face hub, or MODEL_MAPPER_MODELS):
  key1    answer key tier 1: each marker's concept on the key's pinned questions; the marker must come back near the
          key's rank with a similar drop (the 16 Oct rule; modelmapper/answerkey.py)
  map     the whole map_concept, offline, on the key's pinned Bitcoin question set: the marker L23:2392 near the top
  steer   steering smoke test: L23:2392 held at 8x its peak on two prompts changes the replies, deterministically
  live    --apis: map-concept for Bitcoin, YouTube and photosynthesis with fresh labeller questions
  judge   --apis, if a judge is set up (Jev key or JUDGE_BACKEND=llm): one judged steering reply
If nothing sets MODEL_MAPPER_DTYPE (environment or .env) and there is no GPU, float32 is used (much faster on a CPU).
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

from modelmapper import guards, settings as S

MODEL = "Qwen/Qwen3-0.6B"
STEPS = ("key1", "map", "steer", "live", "judge")
TAIL = 300            # characters of an error kept in the report (METHOD.md, "Limits")
LIVE_PASS_RANK = 20   # live step: a marker counts as found in its concept's top 20 (the key's tier-2 bar)
LIVE_PASS_FOUND = 2   # ... and the step passes with 2 of the 3 markers found (else WARN: fresh questions vary)
# What our machine got (CALIBRATION.md has the runs; CPU, float32, the setting check picks with no GPU).
OURS = {
    "python": "3.11.9 (3.10-3.12 supported)",
    "key1": "3/3 at the key's ranks: Bitcoin L23:2392 rank 2, YouTube L23:2069 rank 1, photosynthesis L23:487 rank 4; "
            "drops 0.160 / 0.150 / 0.211 on a CPU in float32 (0.155 / 0.146 / 0.208 on the GPU in bfloat16, the key's)",
    "map": "ok, recall 12; L23:2392 at position 2, drop 0.838; top 5 L27:81, L23:2392, L9:1456, L4:2581, L27:39 "
           "(CPU, float32; ~6.5 min)",
    "steer": "value 692.0 (8 x its peak); 2/2 replies changed; repeat run identical; first steered reply 'Fun fact: among "
             "the address999 address mapping mapping ...' (CPU, float32)",
    "live": "3/3 in 2 runs: Bitcoin L23:2392 at 2 and 2, YouTube L23:2069 at 2 and 3, photosynthesis L23:487 at 4 and 4 "
            "(CPU, ~22 min, under a cent of labeller calls)",
    "judge": "a probability in [0, 1] for each reply",
}
PINS = {"torch": "2.11.0", "transformers": "5.20.0", "accelerate": "1.15.0", "tokenizers": "0.23.3", "safetensors": "0.8.0",
        "huggingface_hub": "2.2.0", "numpy": "2.4.6" if sys.version_info >= (3, 11) else "2.2.6",
        "pyarrow": "26.0.0" if sys.version_info >= (3, 11) else "25.0.1", "httpx": "0.28.1"}
MAP_PASS_RANK = 5     # offline map: the marker in the top 5 (calibrated value in OURS["map"])
STEER_UNIT, STEER_PROMPTS = "L23:2392=x8", ["Tell me a fun fact.", "What should I do this weekend?"]


# ------------------------------------------------------------------------------------------------ privacy
def machine_id():
    return "machine-" + hashlib.sha1(platform.node().encode("utf-8")).hexdigest()[:6]


def private(s):
    """Text safe to send back: no home folder, user name, host name, endpoint error bodies or key-like strings."""
    s = str(s)
    home = str(Path.home())
    for h in {home, home.replace("\\", "/")}:
        if len(h) > 3:
            s = re.sub(re.escape(h), "~", s, flags=re.I)
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = ""
    if len(user) >= 3:
        s = re.sub(rf"(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])", "<user>", s, flags=re.I)
    node = platform.node()
    if len(node) >= 3:
        s = re.sub(re.escape(node), "<machine>", s, flags=re.I)
    s = re.sub(r"(HTTP \d{3}):.*", r"\1 (response body left out)", s)
    s = re.sub(r"\b(sk|key|pk)-[A-Za-z0-9_\-]{6,}", "[key-like text left out]", s)
    return s


class Report:
    def __init__(self, out):
        self.out = out; self.items = []; self.t0 = time.time()

    def add(self, id_, name, status, value="", expected="", detail="", seconds=None):
        it = {"id": id_, "name": name, "status": status, "value": private(value), "expected": private(expected),
              "detail": private(detail), "seconds": seconds}
        self.items.append(it)
        print(f"[{status:<7}] {name}: {it['value']}" + (f"   ({it['detail']})" if it["detail"] else ""), flush=True)
        return it

    def overall(self):
        if any(i["status"] == "FAIL" for i in self.items):
            return "FAIL"
        notes = [f"{sum(i['status'] == s for i in self.items)} {s.lower()}" for s in ("WARN", "SKIPPED")
                 if any(i["status"] == s for i in self.items)]
        return "PASS" + (f" ({', '.join(notes)}: see below)" if notes else "")

    def write(self, meta):
        self.out.mkdir(parents=True, exist_ok=True)
        data = {"overall": self.overall(), **meta, "seconds": round(time.time() - self.t0), "items": self.items}
        (self.out / "report.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        w = max(len(i["name"]) for i in self.items)
        lines = [f"Model Mapper install check: {data['overall']}",
                 f"{meta['when']} | {meta['machine']} | modelmapper {meta['version']} | steps: {meta['steps']} | "
                 f"{data['seconds']} s", ""]
        for i in self.items:
            lines.append(f"{i['status']:<7} {i['name']:<{w}}  {i['value']}")
            if i["expected"]:
                lines.append(f"{'':<7} {'':<{w}}  ours: {i['expected']}")
            if i["detail"]:
                lines.append(f"{'':<7} {'':<{w}}  {i['detail']}")
        lines += ["", "To report back: send this file (report.txt). It holds no keys, passwords, user or host names."]
        (self.out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return data, "\n".join(lines)


# ------------------------------------------------------------------------------------------------ environment
def _version(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return f"MISSING ({type(e).__name__})"


def _key_source(env_names, key_file):
    """Where a usable key would come from (never its value): 'environment', '.env', '<file>' or None."""
    for n in env_names:
        if os.environ.get(n, "").strip():
            return "environment"
        if S._read_dotenv().get(n):
            return ".env"
    try:
        if key_file.exists() and any(l.strip() and not l.strip().startswith("#")
                                     for l in key_file.read_text(encoding="utf-8-sig").splitlines()):
            return key_file.name
    except OSError:
        pass
    return None


def environment(rep, apis):
    v = platform.python_version()
    ok = (3, 10) <= sys.version_info[:2] <= (3, 12)
    rep.add("python", "Python", "PASS" if ok else "WARN", f"{v} ({guards.os_name()}, {platform.machine()})",
            OURS["python"], "" if ok else "untested Python version")
    import modelmapper
    rep.add("install", "modelmapper", "PASS", f"{modelmapper.__version__}", "0.1.0")
    got = {mod: _version(mod) for mod in PINS}
    missing = [m for m, x in got.items() if x.startswith("MISSING")]
    other = [f"{m} {x} (tested {PINS[m]})" for m, x in got.items() if m not in missing and x.split("+")[0] != PINS[m]]
    rep.add("packages", f"packages ({len(PINS)})", "FAIL" if missing else "WARN" if other else "PASS",
            f"missing: {', '.join(missing)}" if missing else f"different versions: {'; '.join(other)}" if other else
            f"all at the tested versions (torch {got['torch']}, transformers {got['transformers']})",
            f"the pins in requirements.txt for Python {sys.version_info[0]}.{sys.version_info[1]}",
            "pip install -r requirements.txt" if missing else "untested versions; the checks below show whether they matter" if other else "")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        rep.add("gpu", "torch", "FAIL", f"cannot import torch: {type(e).__name__}"[:TAIL], "torch 2.11.0",
                "install torch first (SETUP.md)")
        return
    try:
        cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if cuda:
            dev = S.env("MODEL_MAPPER_DEVICE") or "cuda"
            idx = int(dev.split(":")[1]) if dev.startswith("cuda:") and dev.split(":")[1].isdigit() else 0
            pr = torch.cuda.get_device_properties(idx)
            gpu = f"{pr.name}, {pr.total_memory / 2**30:.1f} GB, CUDA {torch.version.cuda}" + (f" (device {idx})" if idx else "")
        else:
            gpu = "Apple MPS" if mps else "none (CPU only)"
        rep.add("gpu", "GPU", "PASS" if cuda or mps else "WARN", gpu, "NVIDIA GeForce RTX 5090, 31.8 GB, CUDA 12.8",
                "" if cuda or mps else "everything runs on the CPU, slowly; Qwen3-8B needs a GPU with >= 24 GB")
        chose = []
        if not (cuda or mps) and torch.cuda.is_available():
            # e.g. CUDA_VISIBLE_DEVICES="": torch says CUDA is available but has no device; hide CUDA properly
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; chose.append("CUDA_VISIBLE_DEVICES=-1 (CUDA reported no usable device)")
            if not S.env("MODEL_MAPPER_DEVICE"):
                os.environ["MODEL_MAPPER_DEVICE"] = "cpu"; chose.append("MODEL_MAPPER_DEVICE=cpu")
        if not (cuda or mps) and not S.env("MODEL_MAPPER_DTYPE"):
            os.environ["MODEL_MAPPER_DTYPE"] = "float32"; chose.append("MODEL_MAPPER_DTYPE=float32 (no GPU; nothing set it)")
        from modelmapper import model as M
        rep.add("device", "device / dtype used", "PASS", f"{M.DEVICE} / {str(M.DTYPE).replace('torch.', '')}",
                "cuda / bfloat16", ("chosen by check: " + "; ".join(chose)) if chose else "from MODEL_MAPPER_DEVICE / MODEL_MAPPER_DTYPE or the defaults")
    except Exception as e:  # noqa: BLE001
        rep.add("gpu", "GPU / device", "FAIL", f"{type(e).__name__}: {e}"[:TAIL], "a working CUDA, MPS or CPU device")
    local = Path(S.MODELS_DIR) / S.short_name(MODEL) if S.MODELS_DIR else None
    if local and (local / "config.json").exists():
        rep.add("model", "model files Qwen3-0.6B", "PASS", "local copy in MODEL_MAPPER_MODELS", "local copy or the hub")
    else:
        rep.add("model", "model files Qwen3-0.6B", "WARN", "no local copy: loaded from the Hugging Face hub (~1.5 GB, cached)",
                "local copy or the hub", "a blocked hub makes the model steps fail; see SETUP.md, Troubleshooting")
    from modelmapper import judge, labeller
    src = _key_source(["OPENAI_API_KEY"], labeller.KEY_FILE)
    custom = labeller.BASE != labeller.DEFAULT_BASE
    host = "custom endpoint" if custom else "api.openai.com"     # never the host itself: it can name the provider and region
    rep.add("labeller", "labeller key", "PASS" if (src or custom) else ("FAIL" if apis else "WARN"),
            f"{'found (' + src + ')' if src else 'not found'}; endpoint {host}, model {labeller.MODEL}",
            "found; endpoint api.openai.com, model gpt-6-luna",
            "" if src else ("custom endpoint: a key may not be needed" if custom else
                            "needed for map-concept, map-batch and --apis: SETUP.md, 'Labeller key' (the human sets it up)"))
    jsrc = _key_source(["JEV_API_KEY", "TYPESAFE_API_KEY"], judge.KEY_FILE)
    llm = judge.BACKEND == "llm"
    jhost = "api.openai.com" if S.env("JUDGE_BASE_URL", "https://api.openai.com/v1").rstrip("/") == "https://api.openai.com/v1" else "custom endpoint"
    rep.add("judge", "judge (optional)", "PASS" if jsrc or llm else "SKIPPED",
            f"backend {judge.BACKEND}; " + ("key found (" + jsrc + ")" if jsrc else
                                           f"model {S.env('JUDGE_MODEL')}, endpoint {jhost}" if llm else "no key"),
            "Jev (the published results' judge)", "" if jsrc or llm else "only `steer --judge` uses it")
    for label, path in (("disk free", Path.cwd()),):
        try:
            free = shutil.disk_usage(path).free / 2**30
            rep.add("disk", label, "PASS" if free >= 5 else "WARN", f"{free:.0f} GB on {Path(path).anchor or '/'}", ">= 5 GB",
                    "" if free >= 5 else "low: the model and its arrays need a few GB")
        except OSError as e:
            rep.add("disk", label, "WARN", type(e).__name__)


# ------------------------------------------------------------------------------------------------ steps
def _step(rep, id_, name, fn):
    """Run one step; an exception is that step's FAIL (with its last line), never a crash of the whole check."""
    t0 = time.time()
    try:
        fn(t0)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        (rep.out / f"{id_}_error.log").write_text(tb, encoding="utf-8")
        rep.add(id_, name, "FAIL", f"did not finish: {type(e).__name__}: {e}"[:TAIL], OURS.get(id_, ""),
                f"traceback in {id_}_error.log", round(time.time() - t0))


def step_key1(rep, m):
    def run(t0):
        from modelmapper import answerkey
        r = answerkey.tier1(m)
        (rep.out / "answer_key_tier1.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        rows = "; ".join(f"{x['concept']} {x['unit'].replace('_N', ':')} " + (f"rank {x['rank']}, drop {x['drop']:.3f}" if x["rank"] else x["detail"])
                         for x in r["rows"])
        rep.add("key1", "answer key tier 1", "PASS" if r["passed"] else "FAIL", f"{r['found']}/{r['markers']} markers ({rows})",
                OURS["key1"], f"needs {r['pass_fraction']:.0%}", round(time.time() - t0))
    _step(rep, "key1", "answer key tier 1", run)


def step_map(rep, m):
    def run(t0):
        from modelmapper import answerkey
        from modelmapper.api import map_concept
        key = answerkey.load_key(MODEL)
        mk = next(x for x in key["markers"] if x["concept"] == "Bitcoin")
        res = map_concept("Bitcoin", out=rep.out / "map_offline", spec=json.loads(json.dumps(mk["spec"])), measurer=m,
                          log=lambda s: print("    map | " + s, flush=True))
        top = [t.neuron for t in res.top]
        unit = f"L{mk['layer']}:{mk['neuron']}"
        pos = top.index(unit) + 1 if unit in top else None
        drop = next((t.drop for t in res.top if t.neuron == unit), None)
        rep.add("map", "map_concept offline (pinned Bitcoin)", "PASS" if pos and pos <= MAP_PASS_RANK else "FAIL",
                f"{res.status}, recall {res.recall}; {unit} " + (f"at position {pos}, drop {drop:.3f}" if pos else "not in the top 30")
                + f"; top 5 {', '.join(top[:5])}", OURS["map"], f"needs the marker in the top {MAP_PASS_RANK}", round(time.time() - t0))
    _step(rep, "map", "map_concept offline (pinned Bitcoin)", run)


def step_steer(rep, m):
    def run(t0):
        from modelmapper.api import steer
        r = steer([STEER_UNIT], prompts=STEER_PROMPTS, max_new=24, measurer=m)
        again = steer([STEER_UNIT], prompts=STEER_PROMPTS, max_new=24, measurer=m, baseline=False)
        changed = sum(a != b for a, b in zip(r["baseline"], r["steered"]))
        same = again["steered"] == r["steered"]
        (rep.out / "steer.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
        ok = changed == len(STEER_PROMPTS) and same and all(s.strip() for s in r["steered"])
        rep.add("steer", "steering smoke test", "PASS" if ok else "FAIL",
                f"{STEER_UNIT} (value {r['units'][0][2]}): {changed}/{len(STEER_PROMPTS)} replies changed; repeat run "
                f"{'identical' if same else 'DIFFERENT'}; first steered reply: {r['steered'][0][:80]!r}",
                OURS["steer"], "needs every reply changed and a repeat to be identical (greedy decoding)", round(time.time() - t0))
    _step(rep, "steer", "steering smoke test", run)


def step_live(rep, m, budget):
    def run(t0):
        from modelmapper import answerkey
        from modelmapper.api import map_concept
        key = answerkey.load_key(MODEL)
        found, parts = 0, []
        for mk in key["markers"]:
            res = map_concept(mk["concept"], out=rep.out / "map_live", budget=budget, measurer=m,
                              log=lambda s, c=mk["concept"]: print(f"    live {c} | " + s, flush=True))
            unit = f"L{mk['layer']}:{mk['neuron']}"
            pos = next((i + 1 for i, t in enumerate(res.candidates) if t.neuron == unit), None)
            hit = res.status == "ok" and pos is not None and pos <= LIVE_PASS_RANK
            found += hit
            parts.append(f"{mk['concept']} {unit} " + (f"position {pos}" if pos else res.status if res.status != "ok" else "not in the 200"))
        st = "PASS" if found >= LIVE_PASS_FOUND else "WARN"
        rep.add("live", "live map-concept (3 key concepts)", st, f"{found}/3 markers in their top {LIVE_PASS_RANK} ({'; '.join(parts)})",
                OURS["live"], f"PASS with >= {LIVE_PASS_FOUND} of 3, else WARN (the questions are written fresh each run)",
                round(time.time() - t0))
    _step(rep, "live", "live map-concept (3 key concepts)", run)


def step_judge(rep, m):
    def run(t0):
        from modelmapper.api import steer
        r = steer([STEER_UNIT], prompts=STEER_PROMPTS[:1], max_new=24, measurer=m, judge="Does the text mention cryptocurrency?")
        j = r["judge"]; ps = j["steered"] + j["baseline"] + j["fluent_steered"]
        ok = all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in ps)
        rep.add("judge", "judge call (steer --judge)", "PASS" if ok else "FAIL",
                f"backend {j['backend']}: P(cryptocurrency) steered {j['steered'][0]:.2f}, baseline {j['baseline'][0]:.2f}; "
                f"P(fluent) steered {j['fluent_steered'][0]:.2f}", OURS["judge"], "", round(time.time() - t0))
    _step(rep, "judge", "judge call (steer --judge)", run)


def run(apis=False, out=None, quick=False, only=None, budget=0.5) -> dict:
    steps = [s for s in STEPS if (apis or s not in ("live", "judge")) and (not only or s in only)]
    if quick:
        steps = []
    sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
    when = time.strftime("%Y-%m-%d %H:%M %Z")
    out = Path(out) if out else Path.cwd() / "check_reports" / f"{machine_id()}_{time.strftime('%Y%m%d_%H%M%S')}"
    rep = Report(out); out.mkdir(parents=True, exist_ok=True)
    plan = "environment" + "".join(", " + s for s in steps)
    print(f"Model Mapper install check -> {private(out)}\n{plan}\n", flush=True)
    environment(rep, apis)
    if steps and not any(i["status"] == "FAIL" and i["id"] in ("gpu", "packages") for i in rep.items):
        no_key = next(i for i in rep.items if i["id"] == "labeller")["status"] == "FAIL"
        judge_ok = next(i for i in rep.items if i["id"] == "judge")["status"] == "PASS"
        m = None
        try:
            from modelmapper.measure import Measurer
            print(f"loading {MODEL} ...", flush=True)
            m = Measurer(MODEL)
        except Exception as e:  # noqa: BLE001
            rep.add("load", f"load {MODEL}", "FAIL", f"{type(e).__name__}: {e}"[:TAIL], "loads", "see SETUP.md, Troubleshooting")
        if m is not None:
            for s in steps:
                if s == "key1":
                    step_key1(rep, m)
                elif s == "map":
                    step_map(rep, m)
                elif s == "steer":
                    step_steer(rep, m)
                elif s == "live":
                    if no_key:
                        rep.add("live", "live map-concept (3 key concepts)", "SKIPPED", "no labeller key", OURS["live"])
                    else:
                        step_live(rep, m, budget)
                elif s == "judge":
                    if not judge_ok:
                        rep.add("judge", "judge call (steer --judge)", "SKIPPED", "no judge set up (optional)", OURS["judge"])
                    else:
                        step_judge(rep, m)
    elif steps:
        rep.add("steps", "model steps", "SKIPPED", "the environment has a FAIL above; fix it first")
    import modelmapper
    data, txt = rep.write({"when": when, "machine": f"{machine_id()} ({guards.os_name()})",
                           "version": modelmapper.__version__, "apis": apis, "quick": quick, "steps": plan})
    print("\n" + txt + f"\n\nReport: {private(out / 'report.txt')} (and report.json)")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(prog="modelmapper check", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__.split("\n\n", 1)[0], epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--apis", action="store_true", help="also the live map-concept step (and the judge, if set up)")
    ap.add_argument("--quick", action="store_true", help="environment only (seconds)")
    ap.add_argument("--only", help=f"only these steps, comma-separated: {', '.join(STEPS)}")
    ap.add_argument("--out", help="report folder (default check_reports/<machine id>_<time>)")
    ap.add_argument("--budget", type=float, default=0.5, help="labeller spend cap per concept for the live step (USD, Luna prices)")
    a = ap.parse_args(argv)
    only = None
    if a.only:
        only = [x.strip() for x in a.only.split(",") if x.strip()]
        bad = [x for x in only if x not in STEPS]
        if bad:
            ap.error(f"--only: unknown step(s) {', '.join(bad)}; steps: {', '.join(STEPS)}")
        if any(x in ("live", "judge") for x in only) and not a.apis:
            ap.error("--only live / judge need --apis (they call the labeller or the judge)")
    data = run(apis=a.apis, out=a.out, quick=a.quick, only=only, budget=a.budget)
    return 1 if data["overall"] == "FAIL" else 0
