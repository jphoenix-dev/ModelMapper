"""Second grader (ISSUES #16): does Luna, asked the same rubric question, agree with Jev?

Samples (explanation, example) pairs from the census rubric re-grade: own top examples, plus the same explanations on
random snippets. Luna scores each 0/1/2 with the identical rubric text (one call per pair, cached). Reports Spearman
correlation of Luna's score with Jev's soft score, agreement on "fits" (>= 0.5), and whether both separate own
examples from random snippets equally well (AUC).

Usage: python src/grader2.py [n_pairs=400]
"""
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import oai
from cache import Cache
from grader_validation import RUBRIC, WORDINGS
from label_batch import clip

ROOT = Path(__file__).resolve().parents[1]
INSTR = WORDINGS["W1_rubric"][1]
SYS = ("You grade explanations of neurons in a language model. Reply with JSON {\"score\": 0|1|2}, using this scale:\n"
       + "\n".join(f"{i}: {r}" for i, r in enumerate(RUBRIC)))


def auc(pos, neg):
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / ((sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    from atlas import Atlas
    from intruder_test import random_snippets
    from transformers import AutoTokenizer
    from neurons import LOCAL_MODELS
    at = Atlas("Qwen/Qwen3-8B")
    snippets = random_snippets(AutoTokenizer.from_pretrained(f"{LOCAL_MODELS}/Qwen3-8B"), 20, seed=7, chars=500)
    labels = json.loads((ROOT / "results/census_regrade_rubric.json").read_text(encoding="utf-8"))["units"]
    jcache = Cache(str(ROOT / "results/census_regrade_rubric.json.cache.jsonl"))
    h = lambda e: hashlib.sha1(e.encode()).hexdigest()[:10]   # noqa: E731
    rng = random.Random(16)
    items = []
    keys = sorted(labels)
    while len(items) < n:
        uid = rng.choice(keys); pole = rng.choice(("pos", "neg")); v = labels[uid]
        e = (v.get(pole) or {}).get("explanation")
        if not e:
            continue
        if rng.random() < 0.5:
            i = rng.randrange(10)
            exs = at.examples(v["layer"], v["neuron"], pole, n=10)
            if i >= len(exs):
                continue
            k = f"own|{uid}|{pole}|{i}|{h(e)}"; text = clip(exs[i]["text"], 500); kind = "own"
        else:
            si = rng.randrange(len(snippets))
            k = f"rnd|{uid}|{pole}|{si}|{h(e)}"; text = snippets[si]; kind = "rnd"
        if k in jcache:
            items.append((k, kind, text, e, jcache.get(k)))
    cache = Cache(str(ROOT / "results/grader2.cache.jsonl"))

    def luna(it):
        k, _, text, e, _ = it
        if k in cache:
            return
        try:
            r = oai.chat_json(INSTR.format(e=e) + "\n\nText:\n" + text, SYS)
            cache.put(k, int(r["score"]) / 2)
        except Exception:  # noqa: BLE001
            pass
    for w in (64, 16):
        with ThreadPoolExecutor(w) as ex:
            list(ex.map(luna, [it for it in items if it[0] not in cache]))
    got = [(kind, j, cache.get(k)) for k, kind, _, _, j in items if k in cache]
    J = [j for _, j, _ in got]; Lu = [l for _, _, l in got]
    print(f"{len(got)}/{len(items)} pairs graded by both | Luna {oai.usage}")
    print(f"Spearman(Jev soft, Luna 0/0.5/1) = {spearman(J, Lu):.2f}")
    print(f"agreement on fits (>= 0.5): {sum((a >= 0.5) == (b >= 0.5) for a, b in zip(J, Lu)) / len(got):.0%}")
    for name, s in (("Jev", 1), ("Luna", 2)):
        own = [x[s] for x in got if x[0] == "own"]; rnd = [x[s] for x in got if x[0] == "rnd"]
        print(f"{name}: own mean {sum(own)/len(own):.2f}, random mean {sum(rnd)/len(rnd):.2f}, AUC own-vs-random {auc(own, rnd):.2f}")
    (ROOT / "results/grader2.json").write_text(json.dumps(got), encoding="utf-8")


if __name__ == "__main__":
    main()
