"""Blind dossiers for the labeller: one markdown file per unit, identified only by a random code.

Each pole shows its top 20 firings with ~200 tokens of left context. The firing token is [[marked]];
other tokens where the unit fires at > 30% of this example's strength are «marked».
Double-negative (gate-, value-) firings are removed from the positive pole before judging (per PREREG),
and counted in the answer key only.

The key (code -> unit id, role) is written separately and must never be shown to the labeller.

Usage: python src/make_dossiers.py <scan.json> <out_dir> [n=20] [units.json]
"""
import json
import random
import sys
from pathlib import Path

scan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
units_file = Path(sys.argv[4]) if len(sys.argv) > 4 else     Path(sys.argv[1]).parent / sys.argv[1].split("scan_")[-1].replace(".json", "").join(["units_", ".json"])
units_meta = {u["id"]: u for u in json.loads(units_file.read_text(encoding="utf-8"))}

rng = random.Random(42)
ids = list(scan["units"])
codes = rng.sample(range(100, 1000), len(ids))
key = {}
for uid, code in zip(ids, codes):
    e = scan["units"][uid]
    c = f"U{code}"
    pos_all = e["pos"]
    pos = [x for x in pos_all if x.get("regime") != "--"][:n]
    neg = e["neg"][:n]
    key[c] = {"unit": uid, "role": units_meta.get(uid, {}).get("role", "?"),
              "double_negative_removed_from_top40": sum(x.get("regime") == "--" for x in pos_all)}
    lines = [f"# Unit {c}", "",
             "Below are the text positions where this unit fires most strongly in a large sample of web text.",
             "The firing token is marked [[like this]]. Other nearby tokens where it also fires noticeably are marked «like this».",
             "Strength is shown relative to the unit's strongest firing (1.00).", ""]
    for title, exs in (("Positive pole: strongest positive firings", pos), ("Negative pole: strongest negative firings", neg)):
        lines += [f"## {title}", ""]
        peak = abs(exs[0]["value"]) if exs else 1.0
        for i, x in enumerate(exs, 1):
            text = x["text"].replace("\n", " ⏎ ")
            lines += [f"**{i}.** ({abs(x['value'])/peak:.2f}) …{text[-900:]}", ""]
    (out / f"{c}.md").write_text("\n".join(lines), encoding="utf-8")
(out.parent / f"{out.name}_KEY.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
print(f"{len(ids)} dossiers -> {out}; key -> {out.parent / (out.name + '_KEY.json')}")
