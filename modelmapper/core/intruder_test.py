"""Explanation-free cleanliness check: can Jev spot a random snippet hidden among a unit's top firings?

Each trial, per pole: 4 random examples from the unit's top 40 (positive pole: double-negative
firings removed) plus 1 snippet from a random spot in random web text, formatted identically and shuffled.
Score = mean probability Jev puts on the true intruder (chance = 0.2). No explanation is involved,
so a good story can't inflate it. If a unit's top firings share something real, the intruder sticks out.

Usage: python src/intruder_test.py <scan.json> <out.json> [trials=20] [--units id1,id2]
"""
import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from modelmapper.core import jev
from modelmapper.core import neurons  # noqa: F401  (sets UTF-8 stdout on Windows, as before)
from modelmapper.core.config import CORPUS, model_path

CHARS = 320  # characters of context kept before the marked token
INSTR = ("Each numbered snippet is a piece of web text with one token marked [[like this]]. "
         "In all but one snippet, the same detector fired strongly at the marked token. "
         "Which snippet is the odd one out, where the detector did not fire?")


def clip(text):
    """Keep the end of the left context, the marked token and a little after; drop «» highlights."""
    text = text.replace("«", "").replace("»", "")
    i = text.find("[[")
    return text[max(0, i - CHARS):]


def random_snippets(tok, n, seed=0, chars=None):
    chars = chars or CHARS  # CAPS.md #2: should match the context of the examples it's compared with
    rng = random.Random(seed)
    out = []
    for batch in pq.ParquetFile(CORPUS).iter_batches(batch_size=64, columns=["text"]):
        for text in batch.column(0).to_pylist():
            if rng.random() > 0.05:
                continue
            ids = tok(text, add_special_tokens=False).input_ids[:1024]
            if len(ids) < 80:
                continue
            p = rng.randrange(60, len(ids) - 12)
            left, mid, right = tok.decode(ids[:p]), tok.decode(ids[p:p + 1]), tok.decode(ids[p + 1:p + 12])
            out.append(left[-chars:] + f"[[{mid}]]" + right)
            if len(out) >= n:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan"); ap.add_argument("out")
    ap.add_argument("trials", type=int, nargs="?", default=20)
    ap.add_argument("--units")
    a = ap.parse_args()
    scan = json.loads(Path(a.scan).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(model_path(scan['model']))
    unit_ids = a.units.split(",") if a.units else list(scan["units"])
    pool = random_snippets(tok, 2 * a.trials * len(unit_ids))
    rng = random.Random(1)
    results = {}
    for uid in unit_ids:
        results[uid] = {}
        for pole in ("pos", "neg"):
            # positive pole: drop double-negative (gate-, value-) leak firings, per PREREG
            exs = [clip(e["text"]) for e in scan["units"][uid][pole] if pole == "neg" or e.get("regime") != "--"]
            if len(exs) < 4:
                results[uid][pole] = {"mean_p_intruder": None, "n_examples_used": len(exs)}
                continue
            probs = []
            for _ in range(a.trials):
                shown = rng.sample(exs, 4)
                k = rng.randrange(5)
                shown.insert(k, pool.pop() if pool else random_snippets(tok, 1, seed=rng.randrange(10**6))[0])
                probs.append(jev.intruder(shown, INSTR)[k])
            results[uid][pole] = {"mean_p_intruder": round(sum(probs) / len(probs), 3),
                                  "hit_rate": round(sum(p >= 0.5 for p in probs) / len(probs), 3),
                                  "n_examples_used": len(exs)}
        print(f"{uid:>24}: " + " | ".join(f"{p} P(intruder) {results[uid][p]['mean_p_intruder']}" for p in ("pos", "neg")), flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"Jev: {jev.usage}, ~${jev.cost():.4f}")


if __name__ == "__main__":
    main()
