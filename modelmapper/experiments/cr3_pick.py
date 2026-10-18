"""Pick CR-3 concepts mechanically (PREREG CR-3): Luna proposes, the script keeps the first 4 unused per category.

Writes prompts/concepts_cr3.json. Excludes every concept that has a saved prompt file anywhere in results/ (slug
match) and every name in earlier concept lists. Also asks Luna for 4 realistic made-up concepts written as short
fact sheets (same style as the earlier controls), which must not name any real entity as the answer.

Usage: python src/cr3_pick.py
"""
import glob
import json
import os
from pathlib import Path

from modelmapper.core import oai
from modelmapper.core.concept_pipeline import slug
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
CATS = {"animals": "animals", "food_drink": "foods and drinks", "places": "places, landmarks and cities",
        "science": "scientific ideas and phenomena", "technology": "technologies and inventions",
        "arts": "art forms, artworks and music", "sport_games": "sports and games",
        "abstract": "abstract ideas and emotions", "history": "historical events and periods",
        "people": "famous real people (historical or modern)"}


def used_names():
    slugs = {os.path.basename(f)[:-len("_spec.json")] for f in glob.glob(str(ROOT / "results/*/*_spec.json"))}
    names = set()
    for f in glob.glob(str(ROOT / "prompts/concepts_wave*.json")) + [str(ROOT / "prompts/scale_set.json")]:
        for cs in json.loads(Path(f).read_text(encoding="utf-8"))["concepts"].values():
            names |= set(cs)
    for f in glob.glob(str(ROOT / "results/*/*__clamp_reply.json")):
        x = json.loads(Path(f).read_text(encoding="utf-8"))
        if isinstance(x, dict) and "concept" in x:
            names.add(x["concept"].strip())
    return slugs, sorted(n for n in names if len(n) < 60)


def main():
    slugs, names = used_names()
    print(f"{len(slugs)} used slugs, {len(names)} used names")
    out = {"note": "CR-3 concepts (PREREG CR-3), picked by src/cr3_pick.py before any CR-3 run.", "concepts": {}}
    for cat, desc in CATS.items():
        r = oai.chat_json(
            f"List 12 well-known {desc} that a typical adult would recognise and that a language model could name from a "
            f"description. Use their common English names. Do NOT include any of these already-used concepts or close "
            f"variants of them: {json.dumps(names)}. Reply JSON {{\"items\": [..12 names..]}}.")
        picked = []
        for c in r.get("items", []):
            c = str(c).strip()
            if slug(c) in slugs or slug(c) in {slug(n) for n in names} or any(slug(c) == slug(p) for p in picked):
                continue
            picked.append(c)
            if len(picked) == 4:
                break
        out["concepts"][cat] = picked
        print(f"{cat:<12} {picked}   (Luna order: {r.get('items')})")
    r = oai.chat_json(
        "Invent 4 realistic but entirely fictional concepts (e.g. a mountain, a scientist, a bridge, a festival) that do "
        "NOT exist. For each, write it as a name followed by a parenthesised fact sheet of one sentence with concrete "
        "details (place, dates, what it is known for), like: \"Mount Velkaran (a 6,214 m peak in Peru's Cordillera "
        "Vilcabamba, first climbed in 1971 ...)\". Do not reuse Velkaran, Quorrin or Draxwell. Do not say it is fictional. "
        "Reply JSON {\"items\": [4 strings]}.")
    out["concepts"]["made_up_controls"] = [str(x).strip() for x in r.get("items", [])][:4]
    print("made_up_controls", out["concepts"]["made_up_controls"])
    assert all(len(v) == 4 for v in out["concepts"].values()), "a category has fewer than 4"
    (ROOT / "prompts/concepts_cr3.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
