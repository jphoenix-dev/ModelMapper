"""Merge a random-corpus scan with a concept-enriched scan of the same units.

The top 40 of (A's top 40 + B's top 40) is exactly the top 40 of the combined text, so this is
lossless. Each example is tagged with its source, and we count how many of each pole's top 40
fire within a few words of the target phrase (e.g. "golden gate").

Usage: python src/merge_scans.py <random_scan.json> <enriched_scan.json> <out.json> "<phrase>" [units.json]
"""
import json
import re
import sys
from pathlib import Path

a_path, b_path, out_path, phrase = sys.argv[1:5]
A = json.loads(Path(a_path).read_text(encoding="utf-8"))
B = json.loads(Path(b_path).read_text(encoding="utf-8"))
pat = re.compile(re.escape(phrase), re.I)
K = 40


def near_phrase(text, window=60):
    """Does the phrase occur within ~window characters of the marked firing token?"""
    i = text.find("[[")
    j = text.find("]]", i)
    return bool(pat.search(text[max(0, i - window):j + window].replace("«", "").replace("»", "")))


merged = {"model": A["model"], "tokens_scanned": A["tokens_scanned"],
          "enriched_tokens": B["tokens_scanned"], "ctx": A["ctx"], "phrase": phrase, "units": {}}
for uid, ea in A["units"].items():
    eb = B["units"][uid]
    e = {"unit": ea["unit"], "quantiles": ea["quantiles"],
         "max": max(ea["max"], eb["max"]), "min": min(ea["min"], eb["min"])}
    for pole, rev in (("pos", True), ("neg", False)):
        pool = [dict(x, source="random") for x in ea[pole]] + [dict(x, source="enriched") for x in eb[pole]]
        top = sorted(pool, key=lambda x: x["value"], reverse=rev)[:K]
        for x in top:
            x["near_phrase"] = near_phrase(x["text"])
        e[pole] = top
        e[f"{pole}_near_phrase"] = sum(x["near_phrase"] for x in top)
        e[f"{pole}_from_enriched"] = sum(x["source"] == "enriched" for x in top)
    merged["units"][uid] = e
Path(out_path).write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")

units_path = Path(sys.argv[5]) if len(sys.argv) > 5 else Path(a_path).parent / "units_Qwen3-8B.json"
units = {u["id"]: u["role"] for u in json.loads(units_path.read_text(encoding="utf-8"))}
print(f"{'unit':>22} {'role':>28}  pos: near/enriched  neg: near/enriched   (of top {K})")
for uid, e in sorted(merged["units"].items(), key=lambda kv: -max(kv[1]["pos_near_phrase"], kv[1]["neg_near_phrase"])):
    print(f"{uid:>22} {units.get(uid, '?'):>28}  {e['pos_near_phrase']:>4}/{e['pos_from_enriched']:<10} "
          f"{e['neg_near_phrase']:>4}/{e['neg_from_enriched']}")
