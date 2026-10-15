"""Batched, heavily parallel blind labelling for large neuron sets (census, full-model maps).

Same judgement as label.py, but built for thousands of neurons:
  1. Luna writes a short explanation + verdict per pole, from the atlas's top examples (many calls in parallel,
     trimmed context, terse output).
  2. Jev grades coverage: one yes/no per example (N_GRADE per pole), hundreds of calls in flight.
  3. Jev random-token check, BATCHED: each of the 20 random web snippets is sent once per chunk of ~100
     explanations, each explanation as its own isolated question. ~100x fewer calls than label.py.
Clean (same bar as label.py): Luna says ONE, >= 80% of graded examples fit, <= 20% of random tokens fit.

Not used by the running replication (that uses label.py) so results there stay on one code version.

Usage: python src/label_batch.py <model> <units.json|random:N[:seed]> <out.json> [--grade 10] [--workers 128]
"""
import argparse
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jev
import oai
from atlas import Atlas
from intruder_test import random_snippets

ROOT = Path(__file__).resolve().parents[1]
N_RANDOM = 20
Q_PER_CALL = 100
SYSTEM = """You are studying units inside a language model. You get one unit's strongest positive and negative
firings on web text; the firing token is marked [[like this]]. Judge each pole separately. Read the surrounding
text, not just the marked token: units often fire on punctuation or just before something becomes relevant, and
are often better described by a shared pattern or distinction than by a list of topics.
For each pole give: explanation ({LENGTH}, as specific as the evidence allows); verdict:
ONE (one main job, small related offshoots allowed) | TWO_RELATED (two jobs with a nameable connection) |
MIXTURE (unconnected jobs) | NONE (no hypothesis). Be honest about mess; don't stretch an explanation.
Reply with JSON only: {"pos": {"explanation": str, "verdict": str}, "neg": {"explanation": str, "verdict": str}}"""
FIT = ("A detector inside a language model is described as follows: \"{e}\" The detector fired strongly at the "
       "token marked [[like this]]. Taking the surrounding context into account, does the description fit what is "
       "happening at the marked token?")


def clip(text, chars=500):
    i = text.find("[[")
    return text[max(0, i - chars):].replace("«", "").replace("»", "")


class ScanSource:
    """Examples from a scan JSON (scan.py / strata_scan.py format) instead of the atlas; units keyed by id."""
    def __init__(self, path):
        self.units = json.loads(Path(path).read_text(encoding="utf-8"))["units"]

    def examples(self, uid, _unused, pole, n=20):
        return [x for x in self.units[uid][pole] if pole == "neg" or x.get("regime") != "--"][:n]


def dossier(at, l, n, k, chars=400, before=200):
    lines = []
    for title, pole in (("Positive pole", "pos"), ("Negative pole", "neg")):
        exs = at.examples(l, n, pole, n=k) if isinstance(at, ScanSource) else at.examples(l, n, pole, n=k, before=before)
        peak = max(abs(exs[0]["value"]), 1e-9) if exs else 1.0
        lines += [f"## {title}"] + [f"{i}. ({abs(x['value'])/peak:.2f}) …{clip(x['text'], chars).replace(chr(10), ' ⏎ ')}"
                                    for i, x in enumerate(exs, 1)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("units"); ap.add_argument("out")
    ap.add_argument("--examples", type=int, default=15, help="examples shown to Luna per pole")
    ap.add_argument("--grade", type=int, default=10, help="examples graded by Jev per pole")
    ap.add_argument("--workers", type=int, default=256)
    ap.add_argument("--scan", help="take examples from this scan JSON instead of the atlas (units = its ids)")
    # every cap is explicit (CAPS.md); defaults reproduce the original census run
    ap.add_argument("--luna-chars", type=int, default=400, help="chars of left context per example shown to Luna")
    ap.add_argument("--grade-chars", type=int, default=500, help="chars of left context per example graded by Jev")
    ap.add_argument("--random-chars", type=int, default=320, help="chars of left context in random-token snippets")
    ap.add_argument("--before-tokens", type=int, default=200, help="tokens of left context rendered from the atlas")
    ap.add_argument("--explain-length", default="one sentence, max 35 words", help="length instruction to Luna")
    ap.add_argument("--reasoning-effort", help="Luna reasoning_effort (none/low/medium/high/xhigh); default = API default")
    a = ap.parse_args()
    if a.scan:
        at = ScanSource(a.scan)
        units = [(uid, None) for uid in at.units]
    else:
        at = Atlas(a.model)
        L, W = at.a["top_v"].shape[:2]
    if a.scan:
        pass
    elif a.units.startswith("random:"):
        parts = a.units.split(":")
        rng = random.Random(int(parts[2]) if len(parts) > 2 else 0)
        units = sorted({(rng.randrange(L), rng.randrange(W)) for _ in range(int(parts[1]))})
    else:
        units = [(u["layer"], u["neuron"]) for u in json.loads(Path(a.units).read_text(encoding="utf-8"))]
    print(f"{len(units)} units", flush=True)

    # Every API result is appended to <out>.cache.jsonl as it arrives; re-running the same command skips
    # anything already done and retries anything that failed or was never reached.
    from cache import Cache
    import hashlib
    cache = Cache(str(a.out) + ".cache.jsonl")
    uid = lambda u: u[0] if a.scan else f"L{u[0]}_N{u[1]}"   # noqa: E731
    h = lambda e: hashlib.sha1(e.encode()).hexdigest()[:10]   # noqa: E731  (ties grades to the exact explanation)

    def pool(fn, jobs, workers):
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(fn, jobs))

    # 1. Luna explanations
    def explain(u):
        k = f"luna|{uid(u)}"
        if k in cache:
            return True
        try:
            cache.put(k, oai.chat_json(dossier(at, *u, a.examples, a.luna_chars, a.before_tokens),
                                       SYSTEM.replace("{LENGTH}", a.explain_length),
                                       **({"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else {})))
            return True
        except Exception:  # noqa: BLE001
            return False
    for attempt, w in enumerate((min(256, a.workers), 32, 8)):        # main pass, then gentler retry passes
        todo = [u for u in units if f"luna|{uid(u)}" not in cache]
        if not todo:
            break
        pool(explain, todo, w)
    expl = {u: cache.get(f"luna|{uid(u)}", {"error": "luna failed"}) for u in units}
    print(f"Luna: {sum('error' not in v for v in expl.values())}/{len(units)} explained | {oai.usage}", flush=True)

    # 2. coverage grading: one isolated Jev call per example
    jobs = []
    for u in units:
        for pole in ("pos", "neg"):
            e = (expl[u].get(pole) or {}).get("explanation")
            if e:
                exs = at.examples(*u, pole, n=a.grade) if a.scan else at.examples(*u, pole, n=a.grade, before=a.before_tokens)
                for i, x in enumerate(exs):
                    jobs.append((f"cov|{uid(u)}|{pole}|{i}|{h(e)}", clip(x["text"], a.grade_chars), e))

    def grade(job):
        k, text, e = job
        if k in cache:
            return True
        try:
            cache.put(k, jev.noul(text, FIT.format(e=e)))
            return True
        except Exception:  # noqa: BLE001
            return False
    for w in (a.workers, 32, 8):
        todo = [j for j in jobs if j[0] not in cache]
        if not todo:
            break
        pool(grade, todo, w)
    print(f"coverage: {sum(j[0] in cache for j in jobs)}/{len(jobs)} graded | Jev {jev.usage}", flush=True)

    # 3. random-token check, batched: each random snippet x chunks of ~100 explanations (only missing ones)
    from transformers import AutoTokenizer
    from neurons import LOCAL_MODELS
    tok = AutoTokenizer.from_pretrained(f"{LOCAL_MODELS}/Qwen3-8B")
    snippets = random_snippets(tok, N_RANDOM, seed=7, chars=a.random_chars)
    qs = [(uid(u), pole, (expl[u].get(pole) or {}).get("explanation")) for u in units for pole in ("pos", "neg")]
    qs = [(i, p_, e) for i, p_, e in qs if e]
    rkey = lambda i, p_, e, si: f"rnd|{i}|{p_}|{si}|{h(e)}"   # noqa: E731

    def check(job):
        si, chunk = job
        questions = {f"q{j}": {"type": "noul", "instructions": FIT.format(e=e)} for j, (_, _, e) in enumerate(chunk)}
        try:
            ans = jev.ask(snippets[si], questions)
        except Exception:  # noqa: BLE001
            return False
        for j, (i, p_, e) in enumerate(chunk):
            cache.put(rkey(i, p_, e, si), ans[f"q{j}"]["noul"])
        return True
    for w in (a.workers, 16, 4):
        rjobs = []
        for si in range(len(snippets)):
            missing = [q for q in qs if rkey(*q, si) not in cache]
            rjobs += [(si, missing[k:k + Q_PER_CALL]) for k in range(0, len(missing), Q_PER_CALL)]
        if not rjobs:
            break
        pool(check, rjobs, w)
    print(f"random check done | Jev {jev.usage}, ~${jev.cost():.3f}", flush=True)
    cover = {}
    for k, _, _ in jobs:
        if k in cache:
            _, i, p_, *_ = k.split("|")
            cover.setdefault((i, p_), []).append(cache.get(k))
    cover = {(u, pole): cover.get((uid(u), pole), []) for u in units for pole in ("pos", "neg")}
    rnd = {(u, pole): [cache.get(rkey(uid(u), pole, e, si)) for si in range(len(snippets))
                       if rkey(uid(u), pole, e, si) in cache]
           for u in units for pole in ("pos", "neg") for e in [(expl[u].get(pole) or {}).get("explanation") or ""]}

    # 4. assemble
    out = {}
    for u in units:
        r = {"layer": u[0], "neuron": u[1]}
        for pole in ("pos", "neg"):
            l = expl[u].get(pole) or {}
            cv = [p for p in cover.get((u, pole), []) if p is not None]
            rv = rnd.get((u, pole), [])
            cov = sum(p >= 0.5 for p in cv) / len(cv) if cv else None
            rf = sum(p >= 0.5 for p in rv) / len(rv) if rv else None
            if "error" in expl[u] or not cv or not rv:
                r[pole] = {"excluded": "api_failure", "explanation": l.get("explanation"), "verdict": l.get("verdict")}
                continue
            r[pole] = {"explanation": l.get("explanation"), "verdict": l.get("verdict"),
                       "coverage": round(cov, 3) if cov is not None else None,
                       "random_fit": round(rf, 3) if rf is not None else None,
                       "clean": bool(l.get("verdict") == "ONE" and cov is not None and cov >= 0.8
                                     and rf is not None and rf <= 0.2)}
        out[u[0] if a.scan else f"L{u[0]}_N{u[1]}"] = r
    Path(a.out).write_text(json.dumps({"model": a.model, "n": len(units), "units": out,
                                       "settings": {k: getattr(a, k) for k in ("examples", "grade", "luna_chars", "grade_chars",
                                                                               "random_chars", "before_tokens", "explain_length")},
                                       "cost": {"jev_usd": round(jev.cost(), 4), "luna": dict(oai.usage)}},
                                      indent=1, ensure_ascii=False), encoding="utf-8")
    for pole in ("pos", "neg"):
        ok = [r[pole] for r in out.values() if "excluded" not in r[pole]]
        n_clean = sum(x["clean"] for x in ok)
        print(f"{pole}: clean {n_clean}/{len(ok)} = {n_clean/max(len(ok),1):.0%} (excluded for API failures: {len(out)-len(ok)})")
    print(f"Jev ~${jev.cost():.3f} | Luna {oai.usage}")


if __name__ == "__main__":
    main()
