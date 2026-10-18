"""PREREG_MAP P5: do the corpus descriptions of concept neurons hold on NEW text? (no model, no API)

Compares results/corpus_profile_summary.json (atlas 10M tokens) with corpus_profile_summary_new.json (the next ~2M
FineWeb tokens, same neurons, thresholds and keyword rule). Spearman over neurons of (a) firing rate per million tokens
(all profiled neurons), (b) on-concept share (neurons with >= 20 firings in both).
Prediction: (a) >= 0.8 and (b) >= 0.6.
Usage: python src/p5_score.py  -> results/p5_score.txt
"""
import json
from pathlib import Path

from fame_summary import spearman

ROOT = Path(__file__).resolve().parents[1]


def main():
    a = json.loads((ROOT / "results/corpus_profile_summary.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "results/corpus_profile_summary_new.json").read_text(encoding="utf-8"))
    both = [u for u in a if u in b]
    r_rate = spearman([a[u]["rate"] for u in both], [b[u]["rate"] for u in both])
    on = [u for u in both if a[u]["n"] >= 20 and b[u]["n"] >= 20 and a[u]["on"] is not None and b[u]["on"] is not None]
    r_on = spearman([a[u]["on"] for u in on], [b[u]["on"] for u in on]) if len(on) >= 5 else float("nan")
    ok = r_rate >= 0.8 and r_on >= 0.6
    txt = (f"P5 corpus descriptions on new text: {len(both)} neurons\n"
           f"  firing rate: Spearman {r_rate:.3f} (n={len(both)})\n"
           f"  on-concept share: Spearman {r_on:.3f} (n={len(on)}, >= 20 firings in both texts)\n"
           f"  PREDICTION {'HOLDS' if ok else 'FAILS'} (needs >= 0.8 and >= 0.6)")
    (ROOT / "results/p5_score.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
