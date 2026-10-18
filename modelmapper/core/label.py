"""Blind labelling: Luna writes explanations, Jev grades them.

For each dossier (see make_dossiers.py):
  1. Luna (gpt-6-luna) reads the examples and, per pole, writes an explanation, a verdict
     (ONE / TWO_RELATED / MIXTURE / NONE) and which examples it believes it covers.
  2. Jev grades coverage independently: one yes/no per example, "does the explanation fit what
     happens at the marked token?" The writer never grades itself.
  3. Jev gives its own blind verdict from the raw examples (without Luna's explanation).

  4. Specificity: Jev applies the same explanation to 20 random tokens in random web text. A vague
     explanation ("grammatical scaffolding") fits lots of them; a real one fits few. This makes
     explanations able to lose.

A pole is CLEAN if Luna says ONE, Jev finds >= 16/20 of its examples covered, and the explanation
fits <= 4/20 random tokens.

Usage: python src/label.py <dossier_dir> <out.json> [--units U123,U456] [--no-jev]
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import random
import threading

from modelmapper.core import jev
from modelmapper.core import oai
from modelmapper.core.intruder_test import random_snippets
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
SYSTEM = """You are studying units inside a language model. For each unit you get its strongest positive firings and its strongest negative firings on web text. Judge each pole separately.

Read the surrounding text, not just the marked token. Units often fire on punctuation or on the word *before* something becomes relevant, summarising or anticipating the context. A unit may be better described by a shared pattern or distinction than by a list of topics.

For each pole give:
1. explanation: one or two sentences, as specific as the evidence allows.
2. verdict, one of:
   - ONE: one main job covers the examples (small, obviously related offshoots allowed)
   - TWO_RELATED: two jobs, with a stated sensible connection
   - MIXTURE: two or more jobs with no connection you can name
   - NONE: no hypothesis
3. covers: the example numbers your explanation covers.
4. misfits: example numbers that don't fit, each with a few words on what it is about.
5. connection: if TWO_RELATED, the connection in one sentence (it must be testable: a shared pattern, not "both are things"); otherwise null.

Be honest about mess: calling a unit MIXTURE or NONE is as useful as ONE. Don't stretch an explanation to cover everything.

Reply with JSON only:
{"pos": {"explanation": str, "verdict": str, "covers": [int], "misfits": [{"n": int, "about": str}], "connection": str|null},
 "neg": {...same...}}"""

VERDICTS = {
    "ONE": "One main job covers nearly all the snippets (small, obviously related variations allowed).",
    "TWO_RELATED": "Two distinct jobs that are sensibly connected to each other.",
    "MIXTURE": "Two or more jobs with no sensible connection between them.",
    "NONE": "No common pattern can be identified.",
}
EX_RE = re.compile(r"^\*\*(\d+)\.\*\* \(([\d.]+)\) …(.*)$")


def parse_dossier(path):
    poles, cur = {"pos": [], "neg": []}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## Positive"):
            cur = "pos"
        elif line.startswith("## Negative"):
            cur = "neg"
        elif cur and (m := EX_RE.match(line)):
            poles[cur].append({"n": int(m[1]), "strength": float(m[2]), "text": m[3]})
    return poles


def clip(text, chars=600):
    i = text.find("[[")
    return text[max(0, i - chars):].replace("«", "").replace("»", "")


def jev_grade(explanation, examples):
    """P(example fits the explanation), one isolated Jev call per example."""
    instr = (f"A detector inside a language model is described as follows: \"{explanation}\" "
             "The detector fired strongly at the token marked [[like this]]. Taking the surrounding "
             "context into account, does the description fit what is happening at the marked token?")
    with ThreadPoolExecutor(32) as ex:
        return list(ex.map(lambda e: jev.noul(clip(e["text"]), instr), examples))


def jev_verdict(examples, rotations=4):
    """Averaged over shuffled option orders: Jev has a documented first-option bias."""
    state = {str(e["n"]): clip(e["text"], 400) for e in examples}
    instr = ("Each numbered snippet is web text where the same detector inside a language model fired "
             "strongly at the token marked [[like this]]. Looking at what the snippets have in common "
             "around the marked tokens, which description best fits this detector?")
    keys, total = list(VERDICTS), {k: 0.0 for k in VERDICTS}
    for r in range(rotations):
        order = keys[r % len(keys):] + keys[:r % len(keys)]
        for k, p in jev.choice(state, instr, {k: VERDICTS[k] for k in order}).items():
            total[k] += p / rotations
    return {k: round(v, 3) for k, v in total.items()}


_RANDOM = []


def random_fit(explanation, n=20):
    """Share of random web tokens the explanation (wrongly) fits: the false-positive rate."""
    if not _RANDOM:
        from transformers import AutoTokenizer
        from modelmapper.core.config import DEFAULT_MODEL, model_path
        tok = AutoTokenizer.from_pretrained(model_path(DEFAULT_MODEL))
        _RANDOM.extend(random_snippets(tok, n, seed=7))
    return jev_grade(explanation, [{"text": t} for t in _RANDOM[:n]])


def label_unit(path, use_jev=True):
    poles = parse_dossier(path)
    luna = oai.chat_json(Path(path).read_text(encoding="utf-8"), SYSTEM)
    out = {"luna": luna}
    if use_jev:
        for pole in ("pos", "neg"):
            exs = poles[pole]
            expl = (luna.get(pole) or {}).get("explanation") or ""
            fits = jev_grade(expl, exs) if expl and exs else []
            out[f"jev_fit_{pole}"] = [round(p, 3) for p in fits]
            out[f"jev_covered_{pole}"] = sum(p >= 0.5 for p in fits)
            out[f"jev_verdict_{pole}"] = jev_verdict(exs) if exs else {}
            rnd = random_fit(expl) if expl else []
            out[f"random_fit_{pole}"] = sum(p >= 0.5 for p in rnd)
            out[f"clean_{pole}"] = ((luna.get(pole) or {}).get("verdict") == "ONE"
                                    and out[f"jev_covered_{pole}"] >= 0.8 * len(exs)
                                    and out[f"random_fit_{pole}"] <= 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dossiers"); ap.add_argument("out")
    ap.add_argument("--units"); ap.add_argument("--no-jev", action="store_true")
    a = ap.parse_args()
    files = sorted(Path(a.dossiers).glob("U*.md"))
    if a.units:
        files = [f for f in files if f.stem in a.units.split(",")]
    out_path = Path(a.out)
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [f for f in files if f.stem not in results]
    if not a.no_jev:
        random_fit("warm-up")  # load the shared random snippets once, before threads start
    lock = threading.Lock()

    def work(f):
        r = label_unit(f, not a.no_jev)
        line = " | ".join(f"{p}: {r['luna'].get(p, {}).get('verdict')}"
                          + (f" (covers {r.get(f'jev_covered_{p}')}/20, random fit {r.get(f'random_fit_{p}')}/20,"
                             f" clean={r.get(f'clean_{p}')})" if not a.no_jev else "")
                          for p in ("pos", "neg"))
        with lock:
            results[f.stem] = r
            print(f"{f.stem}: {line}", flush=True)
            out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    with ThreadPoolExecutor(16) as ex:  # 16 units at a time; each also fans out its Jev calls
        list(ex.map(work, todo))
    print(f"Luna: {oai.usage} | Jev: {jev.usage}, ~${jev.cost():.4f}")


if __name__ == "__main__":
    main()
