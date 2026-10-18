"""Job-finding for knowledge neurons, scaled + direct entity neurons. (Luna; GPU)

James (18 Oct): both - (1) scale the entity x relation grid to more kinds of entity and relation; (2) find entity
neurons DIRECTLY at the entity's name tokens, so early knowledge neurons are not crowded out by late answer-sayers.
Entities: 3 kinds x ~30 (landmarks, people, works), each with 5 relations, 3 FIXED phrasings per relation; answers from
Luna; facts kept where Qwen answers >= 2 of 3 phrasings (as facts.py; helpers imported from it).
Stages:
  entities  Luna's short answers                                                              -> entities.json
  build     recall filter + name-token positions                                               -> facts.json
  entity    DIRECT entity neurons: per entity, attribution (activation x gradient of each fact's answer log-prob) summed
            over ONLY its name-token positions and all its facts -> top-100 -> each zeroed ONLY at the name tokens ->
            mean drop over the entity's facts -> entity map (top-20). Specificity: each entity's top-3 zeroed at the name
            tokens of EVERY fact -> own entity vs other entities of the same kind vs other kinds.
  map / grid / position   as facts.py, on all facts (scaled grid)
  summary
Predictions (stated before running): direct entity neurons sit mostly early (median layer < 20); they are entity-specific
(own-entity drop >= 5x other entities'); in the scaled grid early fact neurons are entity neurons for all three kinds,
and relation neurons exist for people and works too.
Output: D:/already-clean-data/facts2/.  Usage: python src/loop_v2/facts2.py entities | build | entity | map | grid | position | summary
"""
import json
import os
import statistics as stx
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "loop_v2"))
import facts as F1  # noqa: E402

OUT = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "facts2"
F1.OUT = OUT                                          # reuse facts.py stages with this folder
KINDS = {
    "landmark": (F1.ENTITIES, F1.TEMPLATES),
    "person": (["Albert Einstein", "Marie Curie", "Isaac Newton", "Charles Darwin", "Leonardo da Vinci", "Pablo Picasso",
                "Wolfgang Amadeus Mozart", "Ludwig van Beethoven", "William Shakespeare", "Napoleon", "Abraham Lincoln", "Mahatma Gandhi",
                "Nelson Mandela", "Cleopatra", "Julius Caesar", "Queen Victoria", "Galileo Galilei", "Nikola Tesla", "Frida Kahlo",
                "Vincent van Gogh", "Jane Austen", "Leo Tolstoy", "Karl Marx", "Confucius", "Genghis Khan", "Johann Sebastian Bach",
                "Sigmund Freud", "Marco Polo", "Christopher Columbus", "Michelangelo"],
               {"country": ["Which country was {X} from?", "What was the nationality of {X}?", "{X} came from which country?"],
                "century": ["In which century was {X} born?", "{X} was born in which century?", "During which century was {X} born?"],
                "occupation": ["What was the profession of {X}?", "What was {X} best known as?", "{X} worked as what?"],
                "work": ["What is the most famous work of {X}?", "Name the best-known work by {X}.", "{X} is most famous for which work?"],
                "birthplace": ["In which city was {X} born?", "Where was {X} born?", "{X} was born in which city?"]}),
    "work": (["Mona Lisa", "The Starry Night", "Hamlet", "Don Quixote", "War and Peace", "Pride and Prejudice", "Divine Comedy",
              "Romeo and Juliet", "Nineteen Eighty-Four", "Odyssey", "Moby-Dick", "The Great Gatsby", "Guernica", "The Last Supper",
              "The Scream", "Les Misérables", "One Hundred Years of Solitude", "The Little Prince", "Faust", "The Odyssey",
              "Crime and Punishment", "The Brothers Karamazov", "Ulysses", "The Divine Comedy", "The Birth of Venus", "The Night Watch",
              "Swan Lake", "The Magic Flute", "Symphony No. 9 (Beethoven)", "The Tale of Genji"],
             {"creator": ["Who created {X}?", "Who is the author or artist of {X}?", "{X} was made by whom?"],
              "country": ["From which country does {X} come?", "{X} originated in which country?", "Which country is {X} from?"],
              "century": ["In which century was {X} created?", "{X} was made in which century?", "During which century was {X} produced?"],
              "language": ["In which language was {X} originally made or written?", "What is the original language of {X}?",
                           "{X} was originally in which language?"],
              "medium": ["What kind of work is {X}?", "Is {X} a painting, a novel, a play, a poem or music? Which kind?",
                         "{X} is what type of artwork?"]})}


def entities():
    import oai
    from concurrent.futures import ThreadPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    SYS = "You are a precise encyclopedia. Reply in JSON only."
    jobs = [(k, e, list(t)) for k, (es, t) in KINDS.items() for e in dict.fromkeys(es)]

    def ask(job):
        k, e, rels = job
        hint = {"century": "<ordinal only, e.g. 19th>", "medium": "<one word: painting, novel, play, poem, opera, ballet, symphony...>",
                "material": "<one or two words>", "designer": "<the main architect or designer as usually cited>"}
        q = f'For "{e}" give short standard answers as JSON with exactly these keys: ' + ", ".join(f'"{r}": "{hint.get(r, "...")}"' for r in rels) + \
            ". Use the most widely cited answer; one to three words each."
        for _ in range(3):
            try:
                return (k, e), oai.chat_json(q, SYS)
            except Exception:  # noqa: BLE001
                pass
        return (k, e), None
    with ThreadPoolExecutor(16) as ex:
        res = {f"{k}|{e}": v for (k, e), v in ex.map(ask, jobs)}
    (OUT / "entities.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(res)} entities; Luna {oai.usage}")


def build():
    import concept_pipeline as cp
    E = json.loads((OUT / "entities.json").read_text(encoding="utf-8")); model, tok = F1.setup()
    facts = []
    for key, ans in E.items():
        if not ans:
            continue
        kind, e = key.split("|", 1); tpls_all = KINDS[kind][1]
        for rel, tpls in tpls_all.items():
            a = str(ans.get(rel, "")).strip()
            if not a or "unknown" in a.lower():
                continue
            forms = list(dict.fromkeys([a, a[:1].upper() + a[1:], a + " century" if rel == "century" else a]))
            enc = [F1.encode_with_subject(tok, t.format(X=e), e) for t in tpls]
            best = None
            for fm in forms:
                ids = tok.encode(fm)
                lo = F1.logodds_groups(model, tok, [[(p, ids, None) for p, _ in enc]])[0]
                ok = [i for i, l in enumerate(lo) if 1 / (1 + 2.718281828 ** (-l)) >= cp.MIN_P]
                if best is None or len(ok) > len(best[1]):
                    best = (fm, ok, ids, lo)
            fm, ok, ids, lo = best
            if len(ok) >= 2:
                facts.append({"kind": kind, "entity": e, "relation": rel, "answer": a, "form": fm, "answer_ids": ids,
                              "prompts": [enc[i][0] for i in ok], "subject_pos": [enc[i][1] for i in ok], "recalled": len(ok)})
    (OUT / "facts.json").write_text(json.dumps(facts, ensure_ascii=False), encoding="utf-8")
    print(f"{len(facts)} facts kept; by kind x relation {dict(Counter((f['kind'], f['relation']) for f in facts))}")


def entity_stage():
    import torch
    from neurons import DEVICE, mlps
    Fs = json.loads((OUT / "facts.json").read_text(encoding="utf-8")); model, tok = F1.setup()
    L, W = model.config.num_hidden_layers, model.config.intermediate_size
    by_e = defaultdict(list)
    for fa in Fs:
        by_e[(fa["kind"], fa["entity"])].append(fa)
    ents = {k: v for k, v in by_e.items() if len(v) >= 2}
    f = open(OUT / "entity_maps.jsonl", "w", encoding="utf-8"); emap = {}
    for (kind, e), fl in ents.items():
        attr = torch.zeros(L, W, device=DEVICE); n_p = sum(len(fa["prompts"]) for fa in fl)
        for fa in fl:
            for p, subj in zip(fa["prompts"], fa["subject_pos"]):
                if not subj:
                    continue
                acts = {}

                def keep(l):
                    def h(mod, a):
                        a0 = a[0]
                        if not a0.requires_grad:
                            a0 = a0.detach().requires_grad_(True); acts[l] = a0; return (a0,)
                        a0.retain_grad(); acts[l] = a0
                    return h
                hks = [mlps(model)[l].down_proj.register_forward_pre_hook(keep(l)) for l in range(L)]
                ids = torch.tensor([p + fa["answer_ids"][:-1]], device=DEVICE); n = len(p)
                try:
                    with torch.enable_grad():
                        lp = model(input_ids=ids).logits[0, n - 1:].float().log_softmax(-1)
                        lp[torch.arange(len(fa["answer_ids"]), device=DEVICE), torch.tensor(fa["answer_ids"], device=DEVICE)].sum().backward()
                finally:
                    for h in hks:
                        h.remove()
                for l in range(L):
                    attr[l] += (acts[l][0, subj].float() * acts[l].grad[0, subj].float()).sum(0) / n_p
        top = [divmod(int(i), W) for i in attr.flatten().topk(100).indices]
        drops = defaultdict(list)
        for fa in fl:
            P = len(fa["prompts"])
            for c0 in range(0, len(top), 32):
                g = [(p, fa["answer_ids"], None) for p in fa["prompts"]]
                for (l, nn) in top[c0:c0 + 32]:
                    g += [(p, fa["answer_ids"], [(l, nn, sp)] if sp else []) for p, sp in zip(fa["prompts"], fa["subject_pos"])]
                v = F1.logodds_groups(model, tok, [g])[0]; base = v[:P]
                for j, (l, nn) in enumerate(top[c0:c0 + 32]):
                    drops[f"L{l}:{nn}"].append(stx.mean(b - x for b, x in zip(base, v[P * (j + 1):P * (j + 2)])))
        mean_d = {u: stx.mean(v) for u, v in drops.items()}
        ranked = sorted(mean_d, key=lambda u: -mean_d[u])[:20]
        rec = {"kind": kind, "entity": e, "n_facts": len(fl), "top": [{"neuron": u, "mean_drop": round(mean_d[u], 4),
               "per_fact": [round(x, 3) for x in drops[u]]} for u in ranked]}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); emap[(kind, e)] = rec
        print(f"entity {kind}/{e}: {[(x['neuron'], x['mean_drop']) for x in rec['top'][:3]]}", flush=True)
    # specificity: each entity's top-3 zeroed at the name tokens of EVERY fact
    units = sorted({(x["neuron"], k[1]) for k, r in emap.items() for x in r["top"][:3] if x["mean_drop"] > 0.2})
    uniq = sorted({u for u, _ in units})
    g_out = open(OUT / "entity_spec.jsonl", "w", encoding="utf-8")
    for fa in Fs:
        P = len(fa["prompts"]); res = {}
        for c0 in range(0, len(uniq), 48):
            chunk = uniq[c0:c0 + 48]
            g = [(p, fa["answer_ids"], None) for p in fa["prompts"]]
            for u in chunk:
                l, nn = int(u[1:].split(":")[0]), int(u.split(":")[1])
                g += [(p, fa["answer_ids"], [(l, nn, sp)] if sp else []) for p, sp in zip(fa["prompts"], fa["subject_pos"])]
            v = F1.logodds_groups(model, tok, [g])[0]; base = v[:P]
            for j, u in enumerate(chunk):
                res[u] = round(stx.mean(b - x for b, x in zip(base, v[P * (j + 1):P * (j + 2)])), 4)
        g_out.write(json.dumps({"kind": fa["kind"], "entity": fa["entity"], "relation": fa["relation"], "drops": res}, ensure_ascii=False) + "\n")
    (OUT / "entity_units.json").write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")


def summary():
    md = stx.median; lay = lambda u: int(u[1:].split(":")[0])
    L = []
    EM = [json.loads(l) for l in (OUT / "entity_maps.jsonl").read_text(encoding="utf-8").splitlines()]
    top3 = [(r["kind"], r["entity"], x) for r in EM for x in r["top"][:3] if x["mean_drop"] > 0.2]
    L.append(f"DIRECT entity neurons ({len(EM)} entities; top-3 with mean drop > 0.2, zeroed only at the name tokens): n={len(top3)}, "
             f"median layer {md(lay(x['neuron']) for _, _, x in top3)}; share < 20: {sum(lay(x['neuron']) < 20 for _, _, x in top3) / max(1, len(top3)):.0%}; "
             f"median drop {md(x['mean_drop'] for _, _, x in top3):.2f}")
    for k in KINDS:
        t = [x for kk, _, x in top3 if kk == k]
        if t:
            L.append(f"  {k:<9} n={len(t)} median layer {md(lay(x['neuron']) for x in t)}, median drop {md(x['mean_drop'] for x in t):.2f}")
    S = [json.loads(l) for l in (OUT / "entity_spec.jsonl").read_text(encoding="utf-8").splitlines()]
    units = json.loads((OUT / "entity_units.json").read_text(encoding="utf-8"))
    kind_of = {r["entity"]: r["kind"] for r in EM}
    ratios, rows = [], []
    for u, e in units:
        own = [s["drops"][u] for s in S if s["entity"] == e and u in s["drops"]]
        same_k = [s["drops"][u] for s in S if s["entity"] != e and s["kind"] == kind_of.get(e) and u in s["drops"]]
        other_k = [s["drops"][u] for s in S if s["kind"] != kind_of.get(e) and u in s["drops"]]
        if own:
            mo, ms, mk = stx.mean(own), stx.mean(same_k) if same_k else 0, stx.mean(other_k) if other_k else 0
            rows.append((u, e, mo, ms, mk)); ratios.append(mo / max(1e-3, max(ms, mk, 1e-3)))
    L.append(f"entity specificity (top-3 zeroed at the name tokens of every fact; n={len(rows)}): own-entity drop median {md(r[2] for r in rows):.2f}, "
             f"other entities same kind {md(r[3] for r in rows):.3f}, other kinds {md(r[4] for r in rows):.3f}; own >= 5x others in "
             f"{sum(r >= 5 for r in ratios)}/{len(ratios)}")
    L.append("  examples: " + "; ".join(f"{u} ({e}) own {mo:.2f} / same-kind {ms:.2f} / other {mk:.2f}" for u, e, mo, ms, mk in sorted(rows, key=lambda r: -r[2])[:10]))
    txt = "\n".join(L)
    (OUT / "summary_entity.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)
    F1.summary()                                       # scaled grid / position summary (facts.py, this folder)


if __name__ == "__main__":
    {"entities": entities, "build": build, "entity": entity_stage, "map": F1.fact_map, "grid": F1.grid, "position": F1.position,
     "summary": summary}[sys.argv[1]]()
