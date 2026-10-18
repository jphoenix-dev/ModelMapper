"""Validate the coverage grader against known answers (ISSUES #16, #23).

For N neurons with blind explanations (from a label_batch run), grade the explanation against
  - its OWN top examples          (should fit)   -> label 1
  - a random OTHER neuron's examples (shouldn't)  -> label 0
with several question wordings. For each wording report:
  AUC         probability a random own-example scores higher than a random other-example (0.5 = chance, 1 = perfect)
  decisive    share of grades below 0.1 or above 0.9
  near-0.5    share of grades in 0.3-0.7
  mean P(own) and mean P(other)
Soft scores are used throughout; nothing is rounded.

Usage: python src/grader_validation.py <labels.json from label_batch> <model> [N=100] [k=8]
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modelmapper.core import jev
from modelmapper.core.atlas import Atlas
from modelmapper.core.label_batch import clip
from modelmapper.core.settings import REPO_ROOT

WORDINGS = {
    "W0_current_noul": ("noul", "A detector inside a language model is described as follows: \"{e}\" The detector fired strongly "
                                "at the token marked [[like this]]. Taking the surrounding context into account, does the "
                                "description fit what is happening at the marked token?"),
    "W1_rubric": ("score", "A detector inside a language model is described as follows: \"{e}\" Look at the token marked "
                           "[[like this]] and its surrounding context. How well does the description explain why the "
                           "detector would fire at that token?"),
    "W2_shows": ("noul", "Description of a pattern: \"{e}\" Does the marked token [[like this]], read in its context, "
                         "show this pattern?"),
}
RUBRIC = ["Does not fit: the description has nothing to do with this token or its context.",
          "Partly fits: related, but the description does not really explain this token.",
          "Clearly fits: this token in this context is a clear example of the description."]


def ask(wording, text, e):
    kind, instr = WORDINGS[wording]
    q = {"type": kind, "instructions": instr.format(e=e)}
    if kind == "score":
        q["criteria"] = RUBRIC
    a = jev.ask(text, {"q": q})["q"]
    if kind == "noul":
        return a["noul"]
    probs = {int(k): v for k, v in a["probabilities"].items()}
    return sum(k * v for k, v in probs.items()) / (len(RUBRIC) - 1)   # expected level, scaled to 0..1


def auc(pos, neg):
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    labels = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["units"]
    at = Atlas(sys.argv[2])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    rng = random.Random(3)
    ids = [u for u, v in labels.items() if (v.get("pos") or {}).get("explanation")]
    pick = rng.sample(ids, min(N, len(ids)))
    jobs = []
    for uid in pick:
        l, n = labels[uid]["layer"], labels[uid]["neuron"]
        other = labels[rng.choice([u for u in ids if u != uid])]
        e = labels[uid]["pos"]["explanation"]
        own = [clip(x["text"]) for x in at.examples(l, n, "pos", n=k)]
        oth = [clip(x["text"]) for x in at.examples(other["layer"], other["neuron"], "pos", n=k)]
        for w in WORDINGS:
            jobs += [(w, uid, 1, t, e) for t in own] + [(w, uid, 0, t, e) for t in oth]

    def run(j):
        w, uid, y, t, e = j
        try:
            return w, uid, y, ask(w, t, e)
        except Exception:  # noqa: BLE001
            return w, uid, y, None
    with ThreadPoolExecutor(256) as ex:
        res = [r for r in ex.map(run, jobs) if r[3] is not None]
    (REPO_ROOT / "results/grader_validation.json").write_text(json.dumps(res), encoding="utf-8")
    print(f"{len(pick)} neurons x {k} own + {k} other examples x {len(WORDINGS)} wordings = {len(res)} grades")
    for w in WORDINGS:
        pos = [p for ww, _, y, p in res if ww == w and y == 1]
        neg = [p for ww, _, y, p in res if ww == w and y == 0]
        allp = pos + neg
        dec = sum(p < 0.1 or p > 0.9 for p in allp) / len(allp)
        mid = sum(0.3 <= p <= 0.7 for p in allp) / len(allp)
        print(f"  {w:<16} AUC {auc(pos, neg):.3f} | decisive {dec:.0%} | near-0.5 {mid:.0%} | "
              f"mean P own {sum(pos)/len(pos):.2f} vs other {sum(neg)/len(neg):.2f}")
    print(f"Jev: {jev.usage}, ~${jev.cost():.3f}")


if __name__ == "__main__":
    main()
