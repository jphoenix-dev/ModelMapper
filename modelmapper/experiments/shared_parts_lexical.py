"""Lexical control for "related concepts share neurons" (LITERATURE.md follow-up 1; Ait Hou & Hwa 2026).

For every non-generic neuron shared by a related pair (results/shared_parts.json), read its top-20 atlas firings on
each pole and check the marked token. A shared neuron counts as LEXICAL (a word-form detector) if, on either pole, >= 50% of its
top-20 marked (firing) tokens contain a word stem the two concepts' names literally share (e.g. "moon" for the Moon / the Moon
landing). Non-lexical shared neurons are the evidence for shared meaning. Also prints each shared neuron's most common
marked tokens so the call can be checked by eye. CPU only.

Usage: python src/shared_parts_lexical.py
"""
import json
import re
from collections import Counter
from pathlib import Path

from modelmapper.core.config import DEFAULT_MODEL
from modelmapper.core.atlas import Atlas
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
STOP = {"the", "of", "a", "an"}


def stems(name):
    return {w.lower()[:4] for w in re.findall(r"[A-Za-z]{3,}", name) if w.lower() not in STOP}


def main():
    d = json.loads((ROOT / "results/shared_parts.json").read_text(encoding="utf-8"))
    generic = {tuple(u) for u in d["generic"]}
    at = Atlas(DEFAULT_MODEL)
    rows, lex_total, sem_total = [], 0, 0
    for p in d["pairs"]:
        a, b = p["a"], p["b"]
        common = stems(a) & stems(b)
        shared = [tuple(u) for u in p["shared"] if tuple(u) not in generic]
        lex, sem = [], []
        for l, n in shared:
            fracs, ctx_fracs, toks_all, missing = [], [], Counter(), 0
            for pole in ("pos", "neg"):
                toks, ctxs = [], []
                # after >= 1: with after=0 the atlas renderer omits the [[ ]] marker (found the hard way)
                for e in at.examples(l, n, pole, n=20, before=10, after=3):
                    m = re.search(r"\[\[(.*?)\]\]", e["text"], re.S)   # the firing token may be a newline
                    if not m:
                        missing += 1; continue
                    toks.append(m.group(1).strip().lower()); ctxs.append(e["text"].lower())
                toks_all.update(toks)
                fracs.append(sum(any(s in t for s in common) for t in toks) / max(1, len(toks)) if common else 0.0)
                ctx_fracs.append(sum(any(s in c for s in common) for c in ctxs) / max(1, len(ctxs)) if common else 0.0)
            assert missing == 0, f"marker missing in {missing} examples for L{l}:{n}"
            is_lex = max(fracs) >= 0.5
            (lex if is_lex else sem).append(((l, n), [t for t, _ in toks_all.most_common(5)] +
                                             ([f"word-in-context {max(ctx_fracs):.0%}"] if common else [])))
        lex_total += len(lex); sem_total += len(sem)
        rows.append({"a": a, "b": b, "common_stems": sorted(common), "lexical": lex, "semantic": sem})
        print(f"{a:<20} ~ {b:<22} common word {sorted(common) or '-'}: {len(shared)} shared -> lexical {len(lex)}, "
              f"not lexical {len(sem)}")
        for u, t in lex + sem:
            print(f"     L{u[0]}:{u[1]:<6} {'LEX ' if (u, t) in lex else 'sem '} top marked tokens {t}")
    n = len(rows)
    print(f"\nnon-generic shared neurons: {lex_total + sem_total}; lexical {lex_total}; not lexical {sem_total} "
          f"-> related pairs share {sem_total / n:.1f} non-lexical neurons on average (unrelated baseline incl. lexical: "
          f"{sum(d['unrelated']) / len(d['unrelated']):.2f} all / see shared_parts.txt for non-generic 0.13)")
    (ROOT / "results/shared_parts_lexical.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
