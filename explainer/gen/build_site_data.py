"""Turn explainer/data/raw/*.json (from make_data.py) into the site's data files.

Writes explainer/site/data/site.json (index, hero, shared parts, timeline) and explainer/site/data/c/<id>.json
(one per concept, loaded on demand). Every number shown on the page comes from these files or from the text of
index.html, which quotes RESULTS.md / MORNING.md.

Usage: .venv/Scripts/python.exe explainer/gen/build_site_data.py
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("EXPLAINER_RAW", ROOT / "explainer" / "data" / "raw"))
OUT = Path(os.environ.get("EXPLAINER_SITE_DATA", ROOT / "explainer" / "site" / "data"))

GROUPS = {
    "People": ["Albert Einstein", "Frida Kahlo", "Napoleon Bonaparte", "William Shakespeare", "Wolfgang Amadeus Mozart",
               "Marie Curie", "Charles Darwin", "Steve Jobs", "Taylor Swift", "Nelson Mandela", "Serena Williams"],
    "Places and landmarks": ["Eiffel Tower", "Golden Gate Bridge", "Mount Everest", "Tokyo", "Venice", "the Great Wall of China",
                             "Egypt's pyramids", "Brazil", "Iceland", "Istanbul", "New York City", "the Colosseum", "Stonehenge",
                             "the Acropolis", "Mount Fuji", "Antarctica", "the Grand Canyon", "Paris", "the Nile"],
    "History": ["the French Revolution", "the Titanic", "the Moon landing", "World War II", "the fall of the Berlin Wall",
                "the Renaissance"],
    "Science and the body": ["photosynthesis", "the Moon", "DNA", "black holes", "evolution by natural selection",
                             "the periodic table", "gravity", "the speed of light", "pi", "electricity", "the heart",
                             "vaccines", "insomnia", "the brain"],
    "Technology and machines": ["the internet", "the steam engine", "the printing press", "Bitcoin", "the bicycle",
                                "the submarine", "the helicopter", "the smartphone", "artificial intelligence",
                                "the light bulb", "the telescope"],
    "Nature and animals": ["the elephant", "the octopus", "the honeybee", "the penguin", "volcanoes", "rainbows", "hurricanes",
                           "the Amazon rainforest", "the oak tree", "bamboo", "the wolf", "the dolphin", "the butterfly",
                           "the shark", "snow", "thunderstorms", "autumn", "the desert heat"],
    "Food and drink": ["coffee", "sushi", "chocolate", "pizza", "wine", "honey", "the tomato", "bread", "cheese",
                       "ice cream", "tea"],
    "Culture, games and sport": ["chess", "the violin", "Harry Potter", "jazz", "the Mona Lisa", "ballet", "haiku",
                                 "Sherlock Holmes", "Star Wars", "Pokémon", "Mickey Mouse", "the guitar", "origami", "opera",
                                 "graffiti", "Zeus", "Thor", "the Buddha", "Medusa", "football (soccer)", "the Olympic Games",
                                 "basketball", "poker", "tennis", "the marathon", "surfing", "chess grandmasters"],
    "Ideas and feelings": ["democracy", "jealousy", "nostalgia", "justice", "infinity", "friendship", "capitalism",
                           "loneliness", "courage"],
    "Everyday things and jobs": ["a doctor", "a firefighter", "a chef", "an astronaut", "the colour red", "the colour blue",
                                 "gold", "shadows", "the umbrella", "the mirror", "the clock", "the key"],
}
GROUP_OF = {c.lower(): g for g, cs in GROUPS.items() for c in cs}

FAILED = [
    {"name": "Mount Velkaran", "why": "made up by us as a control; correctly not recalled"},
    {"name": "The Draxwell Bridge", "why": "made up by us as a control; correctly not recalled"},
    {"name": "Professor Elsabeth Quorrin", "why": "made up by us as a control; correctly not recalled"},
    {"name": "the robot", "why": "not recalled: our questions described specific robots, so the model named those"},
    {"name": "Sahara Desert", "why": "not recalled in this run; recalled later with new questions"},
]
RULES = {"R1": "top neurons in the second half of the model", "R2": "top neurons anywhere",
         "R4": "neurons that most directly push the answer word"}
PROMPT_LABELS = ["Thank-you note", "Sound of rain", "Cat limerick", "Something interesting", "Fairy tale",
                 "Perfect afternoon", "Meet anyone", "Tribute"]
GENERIC = {"the", "mount", "tower", "bridge", "river", "desert", "great", "wall", "games", "table", "engine", "press",
           "colour", "fall", "natural", "selection", "by", "and", "of"}

# Plain labels for the six Eiffel neurons, from their saved blind labels (results/concepts_v3/eiffel_tower__clamp_reply.json)
HERO_LABELS = {(26, 11492): "landmarks and sights", (29, 12094): "buildings and their features",
               (35, 6192): "ordinary words (general-purpose)", (31, 10160): "Paris as a place",
               (35, 6245): "web-page clutter (general-purpose)", (28, 4830): "Paris sights and monuments"}
HERO_FEATURED = [("Autumn poem", "Write a short poem about autumn."), ("Fun fact", "Tell me a fun fact."),
                 ("Misunderstood", "What's something people often misunderstand?"),
                 ("Novel opening", "Write the opening line of a novel."),
                 ("Build anything", "If you could build anything, what would it be?")]

TIMELINE = [
    ["15 Oct, evening", "James reads the paper and writes a plan. First attempt: search for a single \"Golden Gate neuron\". It fails: the search finds neurons for the word \"orange\" and for finishing names."],
    ["15 Oct, late", "Second attempt: trace which neurons push the model's answer. Switching off 5 stops it naming the Golden Gate Bridge; turning up 4 makes it talk about the bridge."],
    ["15 Oct, night", "A neuron atlas: the 40 strongest examples for each of the 442,368 neurons, from 10 million words of web text."],
    ["16 Oct, 00:10", "The census of 1,000 random neurons and the layer test, both written down in advance."],
    ["16 Oct, 01:15", "A 20-concept run with repaired questions. 19 of 20 recalled, all 5 people included."],
    ["16 Oct, 01:30 to 03:40", "We find our measure of \"how many neurons\" was misleading for near-certain facts, and re-measure everything on a better scale."],
    ["16 Oct, 03:00 to 07:46", "Three waves of 40 new concepts each, chosen in advance. Running total: 157 of 158 runs carried by 5 or fewer neurons."],
    ["16 Oct, 04:15", "The interactive dial: pick any concept and steer the model with its neurons."],
    ["16 Oct, 07:49", "Related concepts turn out to share specific neurons about 20 times more than random pairs."],
    ["16 Oct, 08:20", "A literature check: 80 references, each opened and verified, to see what was already known."],
]


def words_for(name, answer):
    ws = [w for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}", f"{name} {answer}") if w.lower() not in GENERIC]
    ws = sorted({w.lower() for w in ws}, key=len, reverse=True)
    return [w[:max(4, len(w) - 2)] for w in ws][:6]         # stems: "coffee" -> "coff", catches "coffees"


def jkey(k):
    """Strength as JavaScript prints it: 1.0 -> "1", 0.25 -> "0.25" (so String(s) finds it)."""
    f = float(k)
    return str(int(f)) if f.is_integer() else str(f)


def jgrid(g):
    return {jkey(k): v for k, v in g.items()}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c").mkdir(exist_ok=True)
    unsteered = json.loads((RAW / "_unsteered.json").read_text(encoding="utf-8"))
    concepts, hero, missing_group = [], None, []
    strengths = None
    for f in sorted(RAW.glob("concepts_*.json")):
        x = json.loads(f.read_text(encoding="utf-8"))
        run = x["run"]
        if x.get("status") != "ok":
            print("skip (not recalled now):", run["concept"]); continue
        name = run["concept"].strip()
        cid = slug(f"{run['dir'].split('_')[1]}-{name}")
        group = GROUP_OF.get(name.lower())
        if not group:
            missing_group.append(name); group = "Other"
        st = x["steer"]
        strengths = [0] + [float(jkey(s)) for s in st["replies"]]
        concepts.append({"id": cid, "name": name, "group": group, "run": run["dir"], "rule": run["rule"], "n": run["n"],
                         "recall_kept": x["recall_kept"], "recall_total": x["recall_total"],
                         "k3": x["knockout"]["k_logodds3"], "highlight": words_for(name, x["answer"])})
        (OUT / "c" / f"{cid}.json").write_text(json.dumps({"steer": {"units": st["units"], "writes": st["writes"],
                                                                     "replies": jgrid(st["replies"])},
                                                           "knockout": x["knockout"]}, ensure_ascii=False), encoding="utf-8")
        if "hero" in x:
            h, ko = x["hero"], x["knockout"]
            feats = [h["prompts"].index(q) for _, q in HERO_FEATURED]
            qplain = [re.search(r"user\n(.*?)<\|im_end\|>", q, re.S).group(1).replace(" Answer with just the name.", "")
                      for q in ko["questions"]]
            hero = {"prompts": h["prompts"], "strengths": h["strengths"], "units": h["units"],
                    "labels": [HERO_LABELS.get(tuple(u), " ".join(w.strip() for w in h["writes"][i][:3])) for i, u in enumerate(h["units"])],
                    "writes": h["writes"], "reply_only": jgrid(h["reply_only"]),
                    "prompt_too": jgrid(h["prompt_too"]),
                    "featured": feats, "featured_labels": [l for l, _ in HERO_FEATURED],
                    "highlight": ["Eiffel", "Eiff", "Tower", "Paris", "Gustave"],
                    "state": ["Normal Qwen.", "Half strength: barely any change.", "The tower starts to slip in.",
                              "The tower turns up in many replies.", "Most replies are about the tower; some start to repeat.",
                              "Repetition takes over.", "Too far: the model can only say \"Eiff Eiff Eiff\"."],
                    "state_prompt": ["Normal Qwen.", "Steering the question too is stronger at every setting.", "", "", "", "", ""],
                    "knock": {**ko, "questions_plain": qplain, "answer": x["answer_form"] or x["answer"]}}
    if missing_group:
        print("WARNING: no menu group for", missing_group)
    pairs = [["the steam engine", "the printing press", 12, 8], ["the Moon", "the Moon landing", 4, 4],
             ["the colour red", "the colour blue", 6, 4], ["the dolphin", "the shark", 3, 3], ["coffee", "tea", 7, 2],
             ["chess", "chess grandmasters", 2, 2], ["snow", "thunderstorms", 2, 1], ["the Golden Gate Bridge", "gold", 1, 1],
             ["Zeus", "Thor", 3, 1], ["the Eiffel Tower", "Paris", 0, 0]]          # results/shared_parts.txt
    site = {"model": "Qwen/Qwen3-8B", "prompts": unsteered["prompts"], "prompt_labels": PROMPT_LABELS,
            "unsteered": unsteered["replies"], "strengths": strengths, "rules": RULES, "default_concept": "coffee",
            "concepts": concepts, "failed": FAILED, "hero": hero, "pairs": pairs,
            "pairs_mean": {"related": 2.6, "random": 0.13}, "timeline": TIMELINE}
    (OUT / "site.json").write_text(json.dumps(site, ensure_ascii=False), encoding="utf-8")
    print(f"{len(concepts)} concepts, hero {'yes' if hero else 'MISSING'}, site.json {(OUT / 'site.json').stat().st_size // 1024} KB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
