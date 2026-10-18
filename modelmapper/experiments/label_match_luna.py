"""R5b: choose label-matched neurons with a Luna judgement instead of word search (follow-up to label_rule.py).

For each CR-2 concept's top-10 attributed neurons, give Luna the recall-pole label and ask whether the neuron is
specifically ABOUT the concept (its own topic: the thing itself, its parts, works or close associations), not
generic machinery whose examples merely include it. Writes results/label_matched_luna_v3.json in the same shape as
label_matched_v3.json, and prints where it differs from word matching.

Usage: python src/label_match_luna.py
"""
import glob
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modelmapper.core import oai
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
SYS = ("You judge descriptions of neurons in a language model. Given a concept and a neuron description, answer "
       "JSON {\"about\": true|false}. true only if the neuron's own job is this concept (the thing itself, its parts, "
       "works, or close associations); false if it is generic (punctuation, function words, scripts, lists, code, "
       "formatting) even when some examples happen to involve the concept.")


def main():
    word = json.loads((ROOT / "results/label_matched_v3.json").read_text(encoding="utf-8"))
    jobs = []
    for f in sorted(glob.glob(str(ROOT / "results/concepts_v3/*__clamp_reply.json"))):
        x = json.loads(Path(f).read_text(encoding="utf-8"))
        if not isinstance(x, dict) or x.get("status") != "ok":
            continue
        for t in x["top_neurons"][:10]:
            u = f"L{t['layer']}_N{t['neuron']}"
            e = ((x["labels"].get(u) or {}).get(t["sign_on_recall"]) or {}).get("explanation")
            if e:
                jobs.append((x["concept"], t["layer"], t["neuron"], t["rank"], e))

    def ask(j):
        c, l, n, r, e = j
        try:
            return bool(oai.chat_json(f"Concept: {c}\nNeuron description: {e}", SYS).get("about"))
        except Exception:  # noqa: BLE001
            return None
    with ThreadPoolExecutor(32) as ex:
        verdicts = list(ex.map(ask, jobs))
    out = {}
    for (c, l, n, r, e), v in zip(jobs, verdicts):
        out.setdefault(c, [])
        if v:
            out[c].append((l, n, r))
    failed = sum(v is None for v in verdicts)
    for c in sorted(set(word) | set(out)):
        w = {tuple(u[:2]) for u in word.get(c, [])}; L = {tuple(u[:2]) for u in out.get(c, [])}
        flag = "" if w == L else f"   (word match: {sorted(w)})"
        print(f"{c:<26} Luna 'about': {sorted(L)}{flag}")
    print(f"{len(jobs)} labels judged, {failed} failed | Luna {oai.usage}")
    (ROOT / "results/label_matched_luna_v3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
