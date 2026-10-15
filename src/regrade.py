"""Re-grade existing explanations with the validated rubric grader, keeping soft scores (ISSUES #23).

Explanations are taken unchanged from a previous labelling run, so only the grader changes. For each unit and pole:
  coverage_soft   mean rubric score (0..1, expected level) over its top-k examples
  coverage_hits   share of examples scoring >= 0.5 ("partly" or better on average)
  random_soft     mean rubric score of the same explanation on 20 random web snippets (same context length)
Every Jev result is cached (resumable). Thresholds are NOT applied here; they get calibrated separately
against known answers (planted mixtures, other-neuron examples).

Usage: python src/regrade.py <labels.json> <out.json> [--scan scan.json] [--k 10] [--chars 500]
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jev
from cache import Cache
from grader_validation import RUBRIC, WORDINGS
from label_batch import ScanSource, clip

ROOT = Path(__file__).resolve().parents[1]
INSTR = WORDINGS["W1_rubric"][1]


def grade(text, e):
    a = jev.ask(text, {"q": {"type": "score", "instructions": INSTR.format(e=e), "criteria": RUBRIC}})["q"]
    probs = {int(k): v for k, v in a["probabilities"].items()}
    return sum(k * v for k, v in probs.items()) / (len(RUBRIC) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("out")
    ap.add_argument("--scan"); ap.add_argument("--k", type=int, default=10); ap.add_argument("--chars", type=int, default=500)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    labels = json.loads(Path(a.labels).read_text(encoding="utf-8"))["units"]
    if a.scan:
        src = ScanSource(a.scan)
        exs = lambda uid, v, pole: src.examples(uid, None, pole, n=a.k)            # noqa: E731
    else:
        from atlas import Atlas
        at = Atlas(a.model)
        exs = lambda uid, v, pole: at.examples(v["layer"], v["neuron"], pole, n=a.k)  # noqa: E731
    from transformers import AutoTokenizer
    from intruder_test import random_snippets
    from neurons import LOCAL_MODELS
    snippets = random_snippets(AutoTokenizer.from_pretrained(f"{LOCAL_MODELS}/Qwen3-8B"), 20, seed=7, chars=a.chars)
    cache = Cache(a.out + ".cache.jsonl")
    h = lambda e: hashlib.sha1(e.encode()).hexdigest()[:10]   # noqa: E731
    jobs = []
    for uid, v in labels.items():
        for pole in ("pos", "neg"):
            e = (v.get(pole) or {}).get("explanation")
            if not e:
                continue
            for i, x in enumerate(exs(uid, v, pole)):
                jobs.append((f"own|{uid}|{pole}|{i}|{h(e)}", clip(x["text"], a.chars), e))
            for si, s in enumerate(snippets):
                jobs.append((f"rnd|{uid}|{pole}|{si}|{h(e)}", s, e))

    def run(j):
        k, text, e = j
        if k in cache:
            return
        try:
            cache.put(k, grade(text, e))
        except Exception:  # noqa: BLE001
            pass

    def run_batch(group):
        """Random-snippet checks share the snippet as state: one Jev call carries up to 50 explanations,
        each as its own isolated question, so the snippet is billed once (Jev's documented cost lever)."""
        text = group[0][1]
        qs = {f"q{i}": {"type": "score", "instructions": INSTR.format(e=e), "criteria": RUBRIC}
              for i, (_, _, e) in enumerate(group)}
        try:
            ans = jev.ask(text, qs)
        except Exception:  # noqa: BLE001
            return
        for i, (k, _, _) in enumerate(group):
            probs = {int(kk): v for kk, v in ans[f"q{i}"]["probabilities"].items()}
            cache.put(k, sum(kk * v for kk, v in probs.items()) / (len(RUBRIC) - 1))
    for w in (256, 32, 8):                      # retry passes pick up anything that failed
        todo = [j for j in jobs if j[0] not in cache]
        if not todo:
            break
        own_todo = [j for j in todo if j[0].startswith("own|")]
        by_snip = {}
        for j in todo:
            if j[0].startswith("rnd|"):
                by_snip.setdefault(j[1], []).append(j)
        groups = [g[i:i + 50] for g in by_snip.values() for i in range(0, len(g), 50)]
        with ThreadPoolExecutor(w) as ex:
            list(ex.map(run, own_todo))
            list(ex.map(run_batch, groups))
    by_key = {}
    for k, _, _ in jobs:
        if k in cache:
            kind, uid_, pole_, *_ = k.split("|")
            by_key.setdefault((kind, uid_, pole_), []).append(cache.get(k))
    out = {}
    for uid, v in labels.items():
        r = {"layer": v.get("layer"), "neuron": v.get("neuron")}
        for pole in ("pos", "neg"):
            e = (v.get(pole) or {}).get("explanation")
            if not e:
                continue
            own = by_key.get(("own", uid, pole), [])
            rnd = by_key.get(("rnd", uid, pole), [])
            if own and rnd:
                r[pole] = {"verdict": v[pole].get("verdict"), "explanation": e,
                           "coverage_soft": round(sum(own) / len(own), 3),
                           "coverage_hits": round(sum(x >= 0.5 for x in own) / len(own), 3),
                           "random_soft": round(sum(rnd) / len(rnd), 3),
                           "random_hits": round(sum(x >= 0.5 for x in rnd) / len(rnd), 3), "n_own": len(own)}
        out[uid] = r
    Path(a.out).write_text(json.dumps({"labels_from": a.labels, "grader": "W1_rubric_soft", "k": a.k, "chars": a.chars,
                                       "units": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} units regraded | Jev {jev.usage}, ~${jev.cost():.3f}")


if __name__ == "__main__":
    main()
