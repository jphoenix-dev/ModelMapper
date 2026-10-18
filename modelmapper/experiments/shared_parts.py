"""Do related concepts share neurons? (composition check; no grader calls)

For each pair of related concepts, re-run attribution on each concept's own recall prompts (saved specs) and compare
their top-50 attributed neurons: overlap count and which neurons are shared. Baseline: the same concepts re-paired at
random (unrelated pairs). Composition from shared parts predicts related >> unrelated.

Usage: python src/shared_parts.py
"""
import json
import os
import random
from pathlib import Path
from modelmapper.core.settings import REPO_ROOT

os.environ["RECALL_VARIANT"] = "V3"
os.environ["ANSWER_DISCOVERY"] = "0"
from modelmapper.core.config import DEFAULT_MODEL  # noqa: E402
from modelmapper.core import concept_pipeline as cp  # noqa: E402
from modelmapper.core import pipeline_fast  # noqa: E402
from modelmapper.core.pipeline_fast import FastPipeline  # noqa: E402

ROOT = REPO_ROOT
PAIRS = [(("concepts_v3", "chess"), ("concepts_w4", "chess grandmasters")),
         (("concepts_v3", "the Moon"), ("concepts_w4", "the Moon landing")),
         (("concepts_v3", "Eiffel Tower"), ("concepts_w4", "Paris")),
         (("concepts_v3", "coffee"), ("concepts_w4", "tea")),
         (("concepts_w4", "the dolphin"), ("concepts_w4", "the shark")),
         (("concepts_w4", "the colour red"), ("concepts_w4", "the colour blue")),
         (("concepts_w2", "the steam engine"), ("concepts_w2", "the printing press")),
         (("concepts_w4", "snow"), ("concepts_w4", "thunderstorms")),
         (("concepts_v3", "Golden Gate Bridge"), ("concepts_w4", "gold")),
         (("concepts_w3", "Zeus"), ("concepts_w3", "Thor"))]
K = 50


def main():
    pipe = FastPipeline(DEFAULT_MODEL)
    tops = {}
    for pair in PAIRS:
        for d, c in pair:
            if (d, c) in tops:
                continue
            cp.OUT = ROOT / "results" / d; pipeline_fast.OUT = cp.OUT
            spec, tid, recall, _ = pipe.make_prompts(c)
            if len(recall) < 5:
                print(f"{c}: not recalled here, skipped", flush=True); continue
            tops[(d, c)] = pipe.attribute([q for q, _ in recall], tid)["top"][:K]
    ok = [p for p in PAIRS if p[0] in tops and p[1] in tops]
    # neurons in the top-50 of >= 3 different concepts are generic machinery, not shared meaning: report both ways
    from collections import Counter
    freq = Counter(u for t in tops.values() for u in set(t))
    generic = {u for u, n in freq.items() if n >= 3}
    print(f"{len(generic)} generic neurons (in >= 3 of {len(tops)} concepts' top-{K})", flush=True)

    def sh(a, b, drop):
        return set(tops[a]) & set(tops[b]) - (generic if drop else set())
    rel_nogen = [len(sh(a, b, True)) for a, b in ok]
    rel = []
    for a, b in ok:
        shared = sorted(set(tops[a]) & set(tops[b]))
        rel.append(len(shared))
        print(f"{a[1]:<20} ~ {b[1]:<22} shared top-{K}: {len(shared):>2} (non-generic {len(sh(a, b, True))})  "
              f"{sorted(sh(a, b, True))[:8]}", flush=True)
    items = sorted({x for p in ok for x in p})
    rng = random.Random(0); unrel, unrel_ng = [], []
    pairset = {frozenset(p) for p in ok}
    while len(unrel) < 200:
        a, b = rng.sample(items, 2)
        if frozenset((a, b)) not in pairset:
            unrel.append(len(set(tops[a]) & set(tops[b]))); unrel_ng.append(len(sh(a, b, True)))
    print(f"\nrelated pairs: mean shared {sum(rel)/len(rel):.1f}/{K} | unrelated (200 random re-pairings): mean "
          f"{sum(unrel)/len(unrel):.1f}, 95th pct {sorted(unrel)[189]}", flush=True)
    print(f"excluding generic neurons: related {sum(rel_nogen)/len(rel_nogen):.1f} vs unrelated "
          f"{sum(unrel_ng)/len(unrel_ng):.2f} (95th pct {sorted(unrel_ng)[189]})", flush=True)
    (ROOT / "results/shared_parts.json").write_text(json.dumps(
        {"pairs": [{"a": a[1], "b": b[1], "shared": sorted(set(tops[a]) & set(tops[b]))} for a, b in ok],
         "related_mean": sum(rel) / len(rel), "unrelated": unrel, "generic": sorted(generic),
         "tops": {f"{d}/{c}": t for (d, c), t in tops.items()}}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
