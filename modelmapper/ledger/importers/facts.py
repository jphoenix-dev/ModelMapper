"""Facts grid (landmarks, facts/) and the scaled grid + direct entity neurons + chain (facts2/) -> ledger packs.

A fact = (entity, relation) -> answer. Evidence about a fact uses concept = the entity (thing:...), context
'relation=<r>;answer=<a>', concept2 = the answer value (era:century-N for centuries, else value:<slug>).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, concept_id, era_century, read_jsonl, relation, thing  # noqa: E402

DOCS = {"facts": "RESULTS.md#18 Oct 2026: job-finding for knowledge neurons - an entity x relation causal grid + position test (research branch)",
        "facts2": "RESULTS.md#18 Oct 2026: scaled facts grid (landmarks, people, works) + the fact route, causally (research branch)"}


def answer_concept(pw, rel, ans):
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)( century)?", str(ans).strip().lower())
    if rel == "century" and m:
        return era_century(pw, int(m.group(1)))
    return pw.concept(concept_id("value", ans), str(ans), "value")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def import_dir(name, force):
    D = DATA_ROOT / name
    files = [D / f for f in ("facts.json", "maps.jsonl", "grid.jsonl", "grid_units.json", "grid_classes.json", "position.jsonl")]
    ent_files = [D / f for f in ("entity_maps.jsonl", "entity_spec.jsonl", "entity_units.json", "entity_sharing.json")]
    chain_files = [D / f for f in ("chain.jsonl", "chain_k5.jsonl")]
    inputs = [f for f in files + ent_files + chain_files + [D / "entities.json"] if f.exists()]
    with PackWriter(f"{name}_grid", inputs=inputs, code=[__file__], force=force) as pw:
        if pw.skip:
            print(f"{name}: unchanged, skipped"); return
        scaled = name == "facts2"
        G = f"{name}_grid"
        pw.experiment(G, title=("Facts grid, scaled: 86 entities x 5 relations (landmarks, people, works)" if scaled
                                else "Facts grid: 30 landmarks x 5 relations"),
                      method=("Each fact ('In which city is X?') is mapped like a concept: attribution of the answer's log-odds, top "
                              "candidates each zeroed alone; the top neurons of every fact are then zeroed on EVERY fact (the grid) "
                              "and classed by where they matter: entity (all of one entity's facts), relation (one question type "
                              "across entities), answer-value (facts sharing the answer), fact-specific or generic. A position test "
                              "zeroes each neuron only at the entity's name tokens or only at the last position."),
                      status="exploratory (stated predictions in RESULTS.md)", date="2026-10-18",
                      script=f"src/loop_v2/{'facts2' if scaled else 'facts'}.py",
                      reproduce=f"python src/loop_v2/{'facts2' if scaled else 'facts'}.py entities|build|map|grid|position|summary",
                      data_files=[f"data:{name}/{f.name}" for f in files if f.exists()],
                      caveats=["No per-fact random-neuron null; class rules use a 0.2 log-odds threshold set by us.",
                               "3 phrasings per fact; answer must be recalled (P >= 0.2) on >= 2."],
                      docs=[DOCS[name], "LAYERS_AND_JOBS.md"], params={"class_threshold": "max(0.2, 25% of own drop)"})
        F = load(D / "facts.json"); fact = {(f["entity"], f["relation"]): f for f in F}
        ent = {}
        for f in F:
            e = ent.setdefault(f["entity"], thing(pw, f["entity"], entity_kind=f.get("kind", "landmark")))
            relation(pw, f["relation"])
        ctx = lambda e, r: f"relation={r};answer={fact[(e, r)]['answer']}" if (e, r) in fact else f"relation={r}"
        ans2 = lambda e, r: answer_concept(pw, r, fact[(e, r)]["answer"]) if (e, r) in fact else None
        for i, m in enumerate(read_jsonl(D / "maps.jsonl")):
            e, r = m["entity"], m["relation"]
            for k, x in enumerate(m["top"]):
                pw.evidence(experiment=G, unit=x["neuron"], concept=ent[e], concept2=ans2(e, r), context=ctx(e, r) + ";test=own_map",
                            evidence_type="knockout_drop", metric="mean_logodds_drop", value=x["drop"], uom="logodds", n=m.get("n_prompts"),
                            threshold=0.2, passes=x["drop"] > 0.2, rank=k + 1,
                            details={k2: v for k2, v in x.items() if k2 not in ("neuron", "drop")}, source=f"{name}/maps.jsonl#{i}")
        for i, g in enumerate(read_jsonl(D / "grid.jsonl")):
            e, r = g["entity"], g["relation"]
            for u, d in g["drops"].items():
                pw.evidence(experiment=G, unit=u, concept=ent[e], concept2=ans2(e, r), context=ctx(e, r) + ";test=grid", evidence_type="knockout_drop",
                            metric="mean_logodds_drop", value=d, uom="logodds", threshold=0.2, passes=d > 0.2, source=f"{name}/grid.jsonl#{i}")
        for i, c in enumerate(load(D / "grid_classes.json")):
            e, r = c["fact"].split("|")
            pw.evidence(experiment=G, unit=c["unit"], concept=ent[e], concept2=ans2(e, r), context=ctx(e, r), evidence_type="job_class",
                        metric="grid_class", value=c["own"], value_text=c["class"], uom="logodds",
                        details={k: v for k, v in c.items() if k not in ("unit", "fact", "class")}, source=f"{name}/grid_classes.json#{i}")
        for i, p in enumerate(read_jsonl(D / "position.jsonl")):
            e, r = p["entity"], p["relation"]
            last = p.get("drop_last_only", p.get("drop_answer_only"))
            for pos, v in (("all", p["drop_all"]), ("name_tokens", p["drop_subject_only"]), ("last_position", last)):
                pw.evidence(experiment=G, unit=p["neuron"], concept=ent[e], context=ctx(e, r) + f";positions={pos}", evidence_type="position_split",
                            metric="mean_logodds_drop", value=v, uom="logodds", source=f"{name}/position.jsonl#{i}")
        if (D / "entity_maps.jsonl").exists():
            E = f"{name}_entity"
            pw.experiment(E, title="Direct entity neurons: attribution at the entity's name tokens, zeroed only there",
                          method=("For each entity, neurons in ALL layers are ranked by attribution summed over only the entity's "
                                  "name-token positions across its facts, then each zeroed only at the name tokens (mean drop over the "
                                  "entity's facts). Specificity: the top-3 (drop > 0.2) zeroed at the name tokens of every fact. Input "
                                  "side: each one's 20 strongest web-text firings, and whether they mention the entity."),
                          status="exploratory", date="2026-10-18", script="src/loop_v2/facts2.py entity; src/loop_v2/entity_sharing.py",
                          reproduce="python src/loop_v2/facts2.py entity; python src/loop_v2/entity_sharing.py",
                          data_files=[f"data:{name}/{f.name}" for f in ent_files if f.exists()],
                          caveats=["Specificity threshold max(0.2, 25% of own) set by us; input-side 'about' = a word of the name within +-8 tokens."],
                          docs=["RESULTS.md#18 Oct 2026: direct entity neurons, scaled to three kinds - and how shared / polysemantic they are (research branch)", "LAYERS_AND_JOBS.md#strongly-supported"],
                          params={"min_drop": 0.2, "share_rule": "max(0.2, 25% of own)"})
            for i, m in enumerate(read_jsonl(D / "entity_maps.jsonl")):
                ce = ent.get(m["entity"]) or thing(pw, m["entity"], entity_kind=m.get("kind"))
                for k, x in enumerate(m["top"]):
                    pw.evidence(experiment=E, unit=x["neuron"], concept=ce, context="positions=name_tokens;test=entity_map", evidence_type="knockout_drop",
                                metric="mean_logodds_drop_over_facts", value=x["mean_drop"], uom="logodds", n=m.get("n_facts"), threshold=0.2,
                                passes=x["mean_drop"] > 0.2, rank=k + 1, details={"per_fact": x.get("per_fact")}, source=f"{name}/entity_maps.jsonl#{i}")
            owner = {}
            for u, e in load(D / "entity_units.json"):
                owner.setdefault(u, []).append(e)
            for i, s in enumerate(read_jsonl(D / "entity_spec.jsonl")):
                e, r = s["entity"], s["relation"]; ce = ent.get(e) or thing(pw, e)
                for u, d in s["drops"].items():
                    for oe in owner.get(u, [None]):
                        pw.evidence(experiment=E, unit=u, concept=ent.get(oe) or (oe and thing(pw, oe)), concept2=ce, context=f"positions=name_tokens;relation={r};test=specificity",
                                    evidence_type="sharing", metric="drop_on_concept2_facts", value=d, uom="logodds", threshold=0.2, passes=d > 0.2,
                                    key=f"spec|{u}|{oe}|{e}|{r}", source=f"{name}/entity_spec.jsonl#{i}")
            if (D / "entity_sharing.json").exists():
                for i, x in enumerate(load(D / "entity_sharing.json")):
                    ce = ent.get(x["entity"]) or thing(pw, x["entity"])
                    pw.evidence(experiment=E, unit=x["unit"], concept=ce, evidence_type="specificity", metric="own_entity_drop", value=x["own"], uom="logodds",
                                details={"shared_with": x["shared_with"]}, source=f"{name}/entity_sharing.json#{i}")
                    pw.evidence(experiment=E, unit=x["unit"], concept=ce, evidence_type="web_firing_share", metric="about", value=x.get("input_about_entity"),
                                n=len(x.get("input_snippets") or []), context="set=atlas_top20;rule=name_word_within_8_tokens",
                                details={"snippets": x.get("input_snippets")}, source=f"{name}/entity_sharing.json#{i}")
        for cf in chain_files:
            if not cf.exists():
                continue
            C = f"{name}_chain" + ("" if cf.stem == "chain" else "_" + cf.stem.split("_")[1])
            pw.experiment(C, title=f"Fact route: entity neurons -> shared answer-value neuron ({'top-1' if cf.stem == 'chain' else 'top-5'} entity neurons)",
                          method=("For each shared answer-value neuron V and every fact with V's answer, the entity's top entity neuron(s) "
                                  "are zeroed only at its name tokens; V's activation at the last prompt position (sign-aligned) and the "
                                  "answer's log-odds are measured, against a random same-layer neuron and another entity's neuron."),
                          status="pre-registered prediction (median V drop >= 20%): failed", date="2026-10-18", script="src/loop_v2/chain.py",
                          reproduce=f"{'CHAIN_K=5 ' if cf.stem != 'chain' else ''}python src/loop_v2/chain.py", data_files=[f"data:{name}/{cf.name}"],
                          caveats=["Each entity neuron supplies only a slice of the value neuron's drive (distributed route)."],
                          docs=[DOCS["facts2"], "LAYERS_AND_JOBS.md#moderately-supported"], params={"prediction": "median relative drop >= 0.2"})
            for i, x in enumerate(read_jsonl(cf)):
                e, r = x["entity"], x["relation"]; ce = ent.get(e) or thing(pw, e); a2 = answer_concept(pw, r, x["answer"])
                ctrl = max(x["v_rel_drop"]["null"], x["v_rel_drop"]["cross"])     # pass rule (RESULTS.md): entity beats both controls
                for cond, unit in (("entity", x["entity_neuron"]), ("null", x["null_neuron"]), ("cross", x["cross_neuron"])):
                    is_e = cond == "entity"
                    pw.evidence(experiment=C, unit=unit, unit2=x["value_neuron"], concept=ce, concept2=a2,
                                context=f"relation={r};condition={cond};positions=name_tokens" + ("" if is_e else ";control=" + ("random" if cond == "null" else "other_entity")),
                                evidence_type="causal_route", metric="relative_activation_drop", value=x["v_rel_drop"][cond],
                                noise_floor=ctrl if is_e else None, threshold=ctrl if is_e else None, passes=(x["v_rel_drop"][cond] > ctrl) if is_e else None,
                                details={"answer_drop": x["answer_drop"][cond], "v_base": x["v_base"], "k": x.get("k", 1)},
                                key=f"{cf.stem}|{i}|{cond}", source=f"{name}/{cf.name}#{i}")


def main(force=False):
    for name in ("facts", "facts2"):
        import_dir(name, force)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
