"""Layout and lighting data for the neuron map figure (explainer/site/map/index.html).

    python -I -X utf8 explainer/gen/map/build_neuronmap.py [data_dir]

Reads the map data already built by build_map.py (default explainer/site/map/data) and writes neuronmap.json:
  regions   left-to-right subject columns: the Vital Articles section a neuron mostly serves in run 1, plus 'Shared'
            (neurons that serve several sections) and 'Other' (only seen in maps made before run 1)
  neurons   every neuron in some concept's top 30, columnar: layer, index, x (0-10000), y within its layer row
            (0-999), region, role, whether it is in any concept's top 10, number of concepts that use it
  concepts  each mapped concept: name, slug, answer, section, its 30 key neurons (by position in 'neurons') with the
            drop in the answer's log-odds when each is switched off, the logit-lens rank of the answer at each depth
            (run 1 only) and its most important heads
The x/y layout is ours: inside the model a layer's neurons have no order. We group them by subject so the map reads.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "site" / "map" / "data"

REGIONS = ["Shared", "People", "History", "Geography", "Society and social sciences", "Philosophy and religion",
           "Arts", "Everyday life", "Biology and health sciences", "Physical sciences", "Technology", "Mathematics",
           "Other"]
SHORT = {"Society and social sciences": "Society", "Philosophy and religion": "Religion & ideas",
         "Biology and health sciences": "Biology & health", "Physical sciences": "Physical sciences",
         "Other": "Other / tests only"}
ROLE = {None: 0, "say": 1, "attribute": 2, "late-other": 3, "early": 4}
LINKS_SHOWN = 40        # links drawn per concept (the strongest by |drop|); the total passing is shown in words
EARLY_CUT = 20         # layers below this are not grouped by subject: their jobs are much less clear (S2, S4b)
NAMES = Path("D:/already-clean-data/job_names/names.cache.jsonl")   # convenience names (JOB_NAMES.md)


def checked_names():
    """Names that pass the rule fixed in JOB_NAMES.md: Luna confidence high and Jev p >= 0.5 on the held-out concept."""
    if not NAMES.exists():
        return {}
    d = {}
    for line in NAMES.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        d[r["k"]] = r["v"]
    out = {}
    for k, v in d.items():
        if k.startswith("luna|") and v.get("name") and v.get("confidence") == "high":
            j = d.get("jev|" + k[5:])
            if j and j.get("p_held", 0) >= 0.5:
                out[k[5:]] = v["name"]
    return out
PICKS = ["Coffee", "Moon", "Elephant", "Jazz", "Mona Lisa", "Chess", "Gold", "Photosynthesis", "Bicycle",
         "Eiffel Tower"]


def load(p):
    return json.loads((DATA / p).read_text(encoding="utf-8"))


def slug_(s):
    import re
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


LEDGER = Path("D:/already-clean-data/ledger/export")
ERA_AUC, ERA_FORMATS = 0.9, 3     # a 'tested job' if AUC >= 0.9 in at least 3 of the 4 question formats


JOB_TYPES = [   # code, label, how it was found (shown on the page); order = legend order
    ("say", "Writes the answer", "Its output pushes the answer's own word: the answer is among the 10 words it raises most (read from its weights, run 1)."),
    ("attribute", "Pushes the answer up", "Its output raises the answer as one of its words, without it being the top one (weights, run 1)."),
    ("entity", "Recognises a thing", "Switched off only at a thing's name (e.g. the words 'Eiffel Tower'), it breaks that thing's facts and few others (facts tests)."),
    ("relation", "Knows the question type", "Switched off, it breaks the same kind of question (e.g. 'which century') across many different things (facts tests)."),
    ("fact", "Needed for one fact", "Switched off, it breaks one particular fact, but not the thing's other facts or other things' same question (facts tests)."),
    ("era", "Picks out a century", "Its firing separates people, places and works of one century from the rest, however the date is asked for (date test)."),
    ("digit", "Tracks how a number is written", "It fires for a digit in a particular place, e.g. years starting '17', not for the era itself (date and number tests)."),
    ("scale", "A two-ended scale", "Firing one way pushes towards one end of a scale (e.g. north), the other way towards the opposite end (south). Each end's description predicts which concepts use it, and pushing the neuron each way moves the model's answers towards that end (scales test, SCALES.md)."),
    ("shared", "Shared answer machinery", "It matters for answers of every kind: in the top 10 of more than 1% of concepts, or breaking facts generally (maps, facts tests)."),
    ("label", "One clear job in web text", "A blind description of where it fires in web text passed our strict one-job check (census, H-DEPTH)."),
    ("named", "Has a checked name", "Luna named it and the name passed a blind check against a concept Luna hadn't seen (JOB_NAMES.md)."),
]
REL_TEXT = {"city": "which city", "country": "which country", "century": "which century", "material": "what it's made of",
            "designer": "who designed it", "birthplace": "where they were born", "occupation": "what they did",
            "language": "what language", "creator": "who made it", "medium": "what medium", "work": "a famous work"}
FACT_CLASS = {"entity": "entity", "entity+relation": "relation", "relation": "relation", "fact-specific": "fact",
              "generic": "shared"}


def ledger_rows(exp):
    import glob
    import gzip
    for p in sorted(glob.glob(str(LEDGER / "evidence" / exp / "*.jsonl.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def ctx_of(r):
    return dict(x.split("=", 1) for x in (r.get("context") or "").split(";") if "=" in x)


def fact_evidence():
    """Per-unit job classes from the facts tests, and per fact the neurons whose knockout breaks it (grid test)."""
    if not (LEDGER / "manifest.json").exists():
        return {}, {}
    unit_types, facts = defaultdict(set), {}
    for exp in ("facts2_grid", "facts_grid"):                  # facts2 (326 facts) first; facts adds any it lacks
        cls, seen = {}, set(k for k in facts)
        for r in ledger_rows(exp):
            c = ctx_of(r)
            if r["evidence_type"] == "job_class" and r.get("unit"):
                t = FACT_CLASS.get(r.get("value_text"))
                if t:
                    unit_types[r["unit"]].add(t)
                    cls[(r["unit"], r["concept"], c.get("relation"))] = t
        for r in ledger_rows(exp):
            c = ctx_of(r)
            if r["evidence_type"] != "knockout_drop" or c.get("test") != "grid" or not r.get("passes_threshold"):
                continue
            key = (r["concept"], c.get("relation"))
            if key in seen:
                continue
            f = facts.setdefault(key, {"entity": r["concept"], "relation": c.get("relation"), "answer": c.get("answer"),
                                       "units": {}, "exp": exp})
            f["units"][r["unit"]] = (r["value"], cls.get((r["unit"], r["concept"], c.get("relation"))))
    for r in ledger_rows("facts2_entity"):
        c = ctx_of(r)
        if r["evidence_type"] == "specificity" and r.get("unit"):
            unit_types[r["unit"]].add("entity")
        if r["evidence_type"] == "knockout_drop" and c.get("test") == "entity_map" and r.get("passes_threshold"):
            for key, f in facts.items():
                if key[0] == r["concept"] and r["unit"] not in f["units"]:
                    f["units"][r["unit"]] = (r["value"], "entity")
    for r in ledger_rows("numeric_facts"):
        if r["evidence_type"] == "job_class" and r.get("unit") and r.get("passes_threshold") and r.get("measure") == "same_digit_same_position_purity":
            unit_types[r["unit"]].add("digit")
    return unit_types, facts


SCALES = Path("D:/already-clean-data/scales/verdicts.json")


def scale_units():
    """Neurons that pass SCALES.md's strict rule, with each end's description by sign. scales.py showed Luna the two
    ends in a seeded random order (end A = the negative end when flipped); undo that here."""
    import hashlib
    import random
    if not SCALES.exists():
        return {}
    # step 4 (research session, scales_steer): pushing the neuron each way at the answer position (k = 1) raises one
    # end's concept and lowers the other's, with a random same-layer neuron quiet
    steer = {r["unit"]: r for r in ledger_rows("scales_steer") if r["evidence_type"] == "job_class"}
    out = {}
    for x in json.loads(SCALES.read_text(encoding="utf-8")):
        s = steer.get(x["n"])
        if x["luna"] != "opposites" or not s:
            continue
        passed = bool(s.get("passes_threshold"))
        status = ("confirmed" if x.get("verdict") == "scale" and passed else "likely" if passed
                  else "one-sided" if x.get("verdict") == "scale" else None)
        if not status:
            continue
        flip = random.Random(int(hashlib.sha1(x["n"].encode()).hexdigest()[:12], 16)).random() < 0.5
        out[x["n"]] = {"pos": x["b"] if flip else x["a"], "neg": x["a"] if flip else x["b"], "status": status,
                       "ends": [s.get("concept"), s.get("concept2")]}
    return out


def era_units():
    """Era neurons from the ledger's era_formats experiment: unit -> century, AUCs, knockout and the tested-job rule."""
    import glob
    import gzip
    if not (LEDGER / "claims.jsonl").exists():
        return {}
    cent, ko = {}, defaultdict(list)
    for p in sorted(glob.glob(str(LEDGER / "evidence" / "era_formats" / "*.jsonl.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if "control" in (r.get("context") or ""):
                    continue
                if r["evidence_type"] == "activation_auc" and r.get("passes_threshold"):
                    cent[r["unit"]] = r["concept"]
                elif r["evidence_type"] == "knockout_drop" and r.get("measure") == "median_logodds_drop_in_era":
                    ko[r["unit"]].append((r.get("context", "").replace("format=", ""), r["value"], bool(r.get("passes_threshold"))))
    out = {}
    for line in (LEDGER / "claims.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        u = c.get("scope") or ""
        if not str(c.get("claim_id", "")).startswith("era-summary-") or u not in cent:
            continue
        aq = c["metric"].get("auc_questions", {})
        out[u] = {"c": cent[u], "aq": {k: round(v, 2) for k, v in aq.items()},
                  "tested": sum(v >= ERA_AUC for v in aq.values()) >= ERA_FORMATS,
                  "ko": max([v for _, v, ok in ko[u] if ok], default=None),
                  "ko_formats": [f for f, _, ok in ko[u] if ok]}
    return out


def concept_doc(slug, row, cols, shards):
    if "shard" in cols:
        k = row[cols.index("shard")]
        if k not in shards:
            shards[k] = load(f"cs/{k}.json")
        return shards[k][slug]
    return load(f"c/{slug}.json")


def main():
    cj = load("concepts.json"); cols = cj["cols"]; rows = cj["rows"]; I = {k: i for i, k in enumerate(cols)}
    section = [r[I["section"]] if r[I["section"]] in REGIONS else None for r in rows]

    # ---- neurons and their region
    neurons = {}
    for L in range(36):
        for idx, v in load(f"n/L{L}.json").items():
            if not v.get("n_uses"):
                continue
            w = Counter()
            for ci, src, rank, drop in v["uses"]:
                if src == 0 and section[ci]:
                    w[section[ci]] += max(0.05, drop or 0)
            if not w:
                region = "Other"
            else:
                top, share = w.most_common(1)[0][0], w.most_common(1)[0][1] / sum(w.values())
                region = "Shared" if (v.get("hub") or (share < 0.5 and len(w) >= 3)) else top
            first = min(v["uses"], key=lambda u: u[2])           # the concept it ranks highest for
            secs = 0
            for s_ in w:
                secs |= 1 << REGIONS.index(s_)
            neurons[f"L{L}:{idx}"] = {"L": L, "i": int(idx), "region": region, "role": v.get("role"), "secs": secs,
                                      "key": any(u[2] <= 10 for u in v["uses"]), "u": v["n_uses"],
                                      "sort": (rows[first[0]][I["type"]] or "~", rows[first[0]][I["name"]])}
    # neurons from the targeted tests (dates, facts) that no concept map found: add them ('Other' column, or the
    # unsorted early band)
    ERA = era_units()
    UT, FACTS = fact_evidence()
    extra = set(ERA) | set(UT) | {u for f in FACTS.values() for u in f["units"]}
    for u in extra:
        if u not in neurons:
            L_, i_ = int(u[1:].split(":")[0]), int(u.split(":")[1])
            neurons[u] = {"L": L_, "i": i_, "region": "Other", "role": None, "secs": 0, "key": False, "u": 0,
                          "sort": ("~", "~")}
    # ---- columns: width by how many neurons a region holds at its busiest layer
    per = defaultdict(Counter)
    for n in neurons.values():
        if n["L"] >= EARLY_CUT:
            per[n["region"]][n["L"]] += 1
    weight = {r: max(per[r].values()) ** 0.75 if per[r] else 0 for r in REGIONS}
    tot = sum(weight.values()); gap = 60; usable = 10000 - gap * (len(REGIONS) - 1)
    regions, x = [], 0.0
    for r in REGIONS:
        w = usable * weight[r] / tot
        regions.append({"name": r, "label": SHORT.get(r, r), "x0": round(x), "x1": round(x + w),
                        "n": sum(per[r].values())})
        x += w + gap
    rx = {r["name"]: r for r in regions}
    # ---- positions: a jitter-free grid inside each (layer, region) cell, similar subjects next to each other
    cells = defaultdict(list)
    for k, n in neurons.items():
        cells[(n["L"], n["region"] if n["L"] >= EARLY_CUT else None)].append(k)
    for (L, reg), ks in cells.items():
        if reg is None:                                       # early layers: the neurons' real order, not by subject
            ks.sort(key=lambda k: neurons[k]["i"])
            x0, x1 = 0, 10000
        else:
            ks.sort(key=lambda k: neurons[k]["sort"])
            x0, x1 = rx[reg]["x0"], rx[reg]["x1"]
        w = max(1, x1 - x0)
        nrows = max(1, min(6, round(math.sqrt(len(ks) * 1000 * 0.35 / w))))   # rows inside a layer band
        ncols = math.ceil(len(ks) / nrows)
        for j, k in enumerate(ks):
            c, r = j // nrows, j % nrows
            neurons[k]["x"] = round(x0 + (c + 0.5) / ncols * w)
            neurons[k]["y"] = round(150 + 700 * ((r + 0.5) / nrows))
    order = sorted(neurons, key=lambda k: (neurons[k]["L"], neurons[k]["x"]))
    pos = {k: j for j, k in enumerate(order)}
    # every kind of job each neuron has shown (bit j = JOB_TYPES[j])
    hubs_ = set()
    for L in range(36):
        for idx, v in load(f"n/L{L}.json").items():
            if v.get("hub"):
                hubs_.add(f"L{L}:{idx}")
            if any(f.get("kind") == "label" and (f.get("metric") or {}).get("clean") for f in v.get("findings", [])):
                UT[f"L{L}:{idx}"].add("label")
    for k, n in neurons.items():
        if n["role"] in ("say", "attribute"):
            UT[k].add(n["role"])
        if k in hubs_:
            UT[k].add("shared")
    for u, e in ERA.items():
        UT[u].add("era" if e["tested"] else "digit" if min(e["aq"].values()) < 0.6 else "")
    names_ = checked_names()
    for u in names_:
        UT[u].add("named")
    SC = scale_units()
    for u, v in SC.items():
        if v["status"] == "confirmed":                 # the job type counts only scales that passed both tests
            UT[u].add("scale")
    bit = {c: 1 << j for j, (c, _, _) in enumerate(JOB_TYPES)}
    out_n = {f: [] for f in ("L", "i", "x", "y", "r", "role", "key", "u", "s", "nm", "era", "jt")}
    rindex = {r: j for j, r in enumerate(REGIONS)}
    names = checked_names(); name_list = sorted(set(names.values())); name_ix = {s: j for j, s in enumerate(name_list)}
    era_list = sorted(ERA, key=lambda u: (int(ERA[u]["c"].split("th")[0].split("st")[0].split("nd")[0].split("rd")[0]), u))
    era_ix = {u: j for j, u in enumerate(era_list)}
    for k in order:
        n = neurons[k]
        for f, v in (("L", n["L"]), ("i", n["i"]), ("x", n["x"]), ("y", n["y"]), ("r", rindex[n["region"]]),
                     ("role", ROLE.get(n["role"], 0)), ("key", 1 if n["key"] else 0), ("u", n["u"]),
                     ("s", n["secs"]), ("nm", name_ix[names[k]] if k in names else -1),
                     ("era", era_ix.get(k, -1)), ("jt", sum(bit.get(t, 0) for t in UT.get(k, ())))):
            out_n[f].append(v)

    # ---- links between a concept's key neurons (LINKS.md): only pairs that pass the rule are drawn
    sys.path.insert(0, str(HERE))
    import links as links_mod
    LK, link_src = links_mod.load()
    lk_by = {k.strip().casefold().removeprefix("the "): v for k, v in LK.items()}

    # ---- concepts: key neurons, logit lens, heads
    out_c, shards, links_total = [], {}, {}
    for r in rows:
        if r[I["status"]] != "mapped":
            continue
        d = concept_doc(r[I["slug"]], r, cols, shards)
        m = next((m for m in d["maps"] if m.get("status") == "mapped"), None)
        if not m or not m.get("top"):
            continue
        # [neuron, drop x100, sign of its activation at the answer position: which way it fires for this concept]
        top = [[pos[t[0]], round((t[1] or 0) * 100), (1 if t[3] >= 0 else -1) if t[3] is not None else 0]
               for t in m["top"] if t[0] in pos]
        lens = (m.get("lens") or {}).get("rank")
        heads = [[h, round(dv * 100)] for h, dv in (m.get("heads") or [])[:12] if dv > 0]
        tier = str(r[I["tier"]] or "")
        fame = int(tier.split("-")[0].rstrip("+")) if tier[:1].isdigit() else 0
        mine = {t[0] for t in m["top"]}
        lks = [[pos[x["from"]], pos[x["to"]], round(x["drop"] * 100), round(x["null"] * 100)]
               for x in lk_by.get(r[I["name"]].strip().casefold().removeprefix("the "), [])
               if x["passes"] and x["from"] in mine and x["to"] in mine and x["from"] in pos and x["to"] in pos]
        if lks:
            links_total[r[I["slug"]]] = len(lks)
            lks = sorted(lks, key=lambda x: -abs(x[2]))[:LINKS_SHOWN]        # draw the strongest; the total is kept
        out_c.append([r[I["slug"]], r[I["name"]], d.get("answer"), r[I["section"]] if r[I["section"]] in REGIONS else None,
                      top, lens, heads, m.get("k_mean"), m.get("source"), fame, lks,
                      m.get("baseline"), m.get("drops"), m.get("ks")])
    # centuries as things to pick: their era neurons light up (size: the in-era knockout drop where it passed)
    centuries = []
    for cname_ in sorted({e["c"] for e in ERA.values()}, key=lambda s: int("".join(ch for ch in s if ch.isdigit()))):
        units = [u for u in era_list if ERA[u]["c"] == cname_]
        top = [[pos[u], round((ERA[u]["ko"] or 0.15) * 100)] for u in units if u in pos]
        centuries.append(["era-" + cname_.replace(" ", "-"), cname_, None, None, top, None, [], None, "era_formats", 0, [],
                          None, None, None, "era"])
    # facts as things to pick: the neurons whose knockout breaks the fact, coloured by their job class
    TI = {c: j for j, (c, _, _) in enumerate(JOB_TYPES)}
    PRIO = ["entity", "relation", "era", "digit", "fact", "shared", "say", "attribute"]

    def fact_class(u, c):
        """The neuron's class on this fact if it was classified there, else its main job from any test."""
        if c:
            return TI[c]
        return next((TI[t] for t in PRIO if t in UT.get(u, ())), -1)
    facts_out = []
    for (ent, rel), f in sorted(FACTS.items(), key=lambda kv: (kv[0][0].casefold(), kv[0][1] or "")):
        top = sorted(([pos[u], round(v * 100), fact_class(u, c)] for u, (v, c) in f["units"].items() if u in pos),
                     key=lambda t: -t[1])[:40]
        if top:
            facts_out.append(["fact-" + slug_(ent) + "-" + (rel or ""), ent, f["answer"], None, top, None, [], None,
                              f["exp"], 0, [], None, None, None, "fact", rel, REL_TEXT.get(rel, (rel or "").replace("_", " "))])
    job_counts = {c: sum(1 for k in order if UT.get(k) and c in UT[k]) for c, _, _ in JOB_TYPES}
    by_name = {c[1]: c[0] for c in out_c}
    out = {"regions": regions, "neurons": out_n, "concepts": out_c, "names": name_list, "early_cut": EARLY_CUT,
           "eras": [dict(ERA[u], unit=u) for u in era_list], "centuries": centuries, "facts": facts_out,
           "scales": {str(pos[u]): v for u, v in SC.items() if u in pos},
           "job_types": [{"code": c, "label": l, "how": h, "n": job_counts[c]} for c, l, h in JOB_TYPES],
           "links_source": link_src, "n_with_links": sum(1 for c in out_c if c[10]), "links_total": links_total,
           "links_shown": LINKS_SHOWN,
           "picks": [by_name[p] for p in PICKS if p in by_name], "roles": list(ROLE)}
    p = DATA / "neuronmap.json"
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{p}: {p.stat().st_size / 1e6:.1f} MB; {len(order)} neurons, {len(out_c)} concepts; regions:",
          {r['label']: r['n'] for r in regions})


if __name__ == "__main__":
    main()
