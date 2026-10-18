"""PREREG_MAP P1, step 1: independent (non-Luna) cloze prompts from Wikipedia opening paragraphs. (web, read-only)

Concepts: recalled Wikipedia-drawn concepts with saved rankings and answer forms (coverage-3: rank_save_fame3.json +
wave_free.json; rank_save_fame3m.json + map_loop_fame3m.json), up to 100 at random (seed 51).
For each: the plain-text intro of its English Wikipedia article; sentences that mention the article name (title
without any parenthetical) are turned into cloze prompts by replacing every occurrence of the name with '___'; a
sentence is dropped if the model's answer form (or the name) still appears after blanking, or it is shorter than 6
words; at most 8 prompts per concept. Prompt text: '<sentence> Which name fills the blank? Answer with just the name.'
Output: results/p1_prompts.json {concept: {"dir", "answer_form", "prompts": [...]}}
Usage: python src/p1_prompts.py
"""
import json
import random
import re
from pathlib import Path

import httpx

from wiki_sample import UA
from wiki_sitelinks import get

ROOT = Path(__file__).resolve().parents[1]


def main():
    pool = {}
    d3 = ROOT / "results/concepts_fame3"
    wf = json.loads((d3 / "wave_free.json").read_text(encoding="utf-8"))
    for c, v in json.loads((d3 / "rank_save_fame3.json").read_text(encoding="utf-8")).items():
        if "top" in v and wf.get(c, {}).get("answer_form"):
            pool[c] = {"rank_file": "rank_save_fame3.json", "answer_form": wf[c]["answer_form"]}
    ml = json.loads((d3 / "map_loop_fame3m.json").read_text(encoding="utf-8"))
    for c, v in json.loads((d3 / "rank_save_fame3m.json").read_text(encoding="utf-8")).items():
        if "top" in v and ml.get(c, {}).get("answer_form"):
            pool[c] = {"rank_file": "rank_save_fame3m.json", "answer_form": ml[c]["answer_form"]}
    names = sorted(pool); random.Random(51).shuffle(names); names = names[:100]
    out = {}
    with httpx.Client(headers=UA, timeout=30) as cl:
        for c in names:
            d = get(cl, "https://en.wikipedia.org/w/api.php", action="query", prop="extracts", exintro=1, explaintext=1,
                    titles=c, redirects=1, format="json")
            pg = next(iter(d["query"]["pages"].values())); text = pg.get("extract", "") or ""
            name = re.sub(r"\s*\(.*?\)\s*$", "", c).strip(); form = pool[c]["answer_form"].strip()
            sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
            prompts = []
            for s in sents:
                if name.lower() not in s.lower():
                    continue
                cz = re.sub(re.escape(name), "___", s, flags=re.I)
                if form.lower() in cz.lower() or name.lower() in cz.lower() or len(cz.split()) < 6:
                    continue
                prompts.append(cz.strip() + " Which name fills the blank? Answer with just the name.")
                if len(prompts) >= 8:
                    break
            out[c] = {"rank_file": pool[c]["rank_file"], "answer_form": form, "prompts": prompts, "intro_chars": len(text)}
    (ROOT / "results/p1_prompts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    n = [len(v["prompts"]) for v in out.values()]
    print(f"{len(out)} concepts; prompts per concept: 0 in {sum(x == 0 for x in n)}, median {sorted(n)[len(n) // 2]}; "
          f"with >= 3 prompts: {sum(x >= 3 for x in n)}")
    ex = next(v for v in out.values() if v["prompts"]); print("example:", ex["prompts"][0][:220])


if __name__ == "__main__":
    main()
