"""Loop v3 step D (pilot): the subject pass - neurons that act at a concept's NAME when Qwen answers questions ABOUT it.
(Luna for questions; GPU for the map.)

Why: loop v2 asks questions whose ANSWER is the concept, so it finds the late neurons that write it; facts2 showed the
knowledge neurons act early at the entity's name (median layer 8) and are invisible to answer-side maps.
Per concept (pilot: the 12 loop-v2 gate concepts):
  questions  Luna writes 6 short factual questions ABOUT the concept, each containing its name verbatim, with 1-3 word
             answers; Qwen must give the answer P >= MIN_P (best of a few casings; " Answer in as few words as possible.")
  map        attribution of each recalled fact's exact log-odds, summed over the name-token positions only, all layers
             (hooks registered once per concept); top 100 each zeroed ONLY at the name tokens -> mean drop over facts;
             6 layer-matched random neurons zeroed the same way (null); top 30 kept with per-fact drops
  summary    predictions below; overlap with the concept's answer-side v2 top 30 (gate_repro records)
Predictions (stated before running):
  S1  >= 9 of 12 concepts get >= 3 recalled facts.
  S2  the subject pass finds different, earlier neurons: median layer of its top-10 <= 15, and <= 10% of its top-30 are in
      the concept's answer-side v2 top-30.
  S3  they matter at the name: the median top-5 drop >= 5x the median |null| drop.
Output: D:/already-clean-data/loop_v3/subject/.  Usage: python src/loop_v2/subject_pass.py questions | map | summary
(questions needs Luna - run from the main checkout; map needs MODEL_MAPPER_MODELS when run from the research worktree)
"""
import json
import os
import random
import statistics as stx
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "loop_v2"))
DATA = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data"))
OUT = DATA / "loop_v3" / "subject"
GATE = DATA / "loop_v2" / "gate_repro" / "records.jsonl"
SUFFIX = " Answer in as few words as possible."


def gate():
    return [json.loads(l) for l in GATE.read_text(encoding="utf-8").splitlines() if l.strip()]


def questions():
    import oai
    from concurrent.futures import ThreadPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    SYS = "You write precise quiz questions. Reply in JSON only."

    def ask(r):
        name = (r["row"].get("answer_form") or r["concept"]).strip()
        q = (f'Write 6 short factual questions ABOUT "{r["concept"]}" that a well-read person could answer in 1-3 words. Each question '
             f'must contain the exact text "{name}" verbatim and must not contain its answer. No yes/no questions; unambiguous answers; '
             f'prefer well-known facts (place, date or century, creator / founder, nationality, field, type of thing). Reply JSON: '
             f'{{"facts": [{{"question": "...", "answer": "..."}}]}}')
        for _ in range(3):
            try:
                return r["concept"], {"name": name, "facts": oai.chat_json(q, SYS)["facts"]}
            except Exception:  # noqa: BLE001
                pass
        return r["concept"], None
    with ThreadPoolExecutor(12) as ex:
        res = dict(ex.map(ask, gate()))
    (OUT / "questions.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{sum(1 for v in res.values() if v)} concepts with questions; Luna {oai.usage}")


def map_():
    import torch
    import concept_pipeline as cp
    import facts as F1
    from config import DEFAULT_MODEL, model_path
    from neurons import DEVICE, mlps
    if not os.path.isdir(model_path(DEFAULT_MODEL)):
        raise SystemExit("set MODEL_MAPPER_MODELS (no local model)")
    Q = json.loads((OUT / "questions.json").read_text(encoding="utf-8"))
    model, tok = F1.setup(); F1.SUFFIX = SUFFIX
    L, W = model.config.num_hidden_layers, model.config.intermediate_size
    f = open(OUT / "subject_maps.jsonl", "w", encoding="utf-8")
    acts = {}

    def keep(l):
        def h(mod, a):
            a0 = a[0]
            if not a0.requires_grad:
                a0 = a0.detach().requires_grad_(True); acts[l] = a0; return (a0,)
            a0.retain_grad(); acts[l] = a0
        return h
    for concept, q in Q.items():
        if not q:
            continue
        rng = random.Random(f"subject|{concept}")
        facts = []
        for x in q["facts"]:
            if q["name"] not in x["question"]:
                continue
            p, subj = F1.encode_with_subject(tok, x["question"], q["name"])
            a = str(x["answer"]).strip(); best = None
            for form in dict.fromkeys([a, a[:1].upper() + a[1:], a.lower(), " " + a]):
                ids = tok.encode(form)
                lo = F1.logodds_groups(model, tok, [[(p, ids, None)]])[0][0]
                if best is None or lo > best[1]:
                    best = (ids, lo, form)
            prob = 1 / (1 + 2.718281828 ** (-best[1]))
            facts.append({"question": x["question"], "answer": a, "form": best[2], "answer_ids": best[0], "prompt": p, "subject_pos": subj,
                          "p": round(prob, 4), "recalled": prob >= cp.MIN_P and bool(subj)})
        rec = [x for x in facts if x["recalled"]]
        out = {"concept": concept, "name": q["name"], "facts": [{k: v for k, v in x.items() if k not in ("prompt", "answer_ids")} for x in facts],
               "n_recalled": len(rec)}
        if len(rec) >= 3:
            attr = torch.zeros(L, W, device=DEVICE)
            hks = [mlps(model)[l].down_proj.register_forward_pre_hook(keep(l)) for l in range(L)]   # once per concept
            try:
                for x in rec:
                    ids = torch.tensor([x["prompt"] + x["answer_ids"][:-1]], device=DEVICE); n = len(x["prompt"])
                    with torch.enable_grad():
                        lp = model(input_ids=ids).logits[0, n - 1:].double().log_softmax(-1)
                        c = lp[torch.arange(len(x["answer_ids"]), device=DEVICE), torch.tensor(x["answer_ids"], device=DEVICE)].sum()
                        (c - torch.log(-torch.expm1(c))).backward()               # exact log-odds (v3 ranking)
                    for l in range(L):
                        attr[l] += (acts[l][0, x["subject_pos"]].float() * acts[l].grad[0, x["subject_pos"]].float()).sum(0)
                    acts.clear()
            finally:
                for h in hks:
                    h.remove()
            top = [divmod(int(i), W) for i in attr.flatten().topk(100).indices]
            nulls = [(l, rng.randrange(W)) for l, _ in top[:6]]
            units = top + nulls; drops = {u: [] for u in units}
            for x in rec:
                for c0 in range(0, len(units), 32):
                    chunk = units[c0:c0 + 32]
                    g = [(x["prompt"], x["answer_ids"], None)] + [(x["prompt"], x["answer_ids"], [(l, n_, x["subject_pos"])]) for l, n_ in chunk]
                    v = F1.logodds_groups(model, tok, [g])[0]
                    for j, u in enumerate(chunk):
                        drops[u].append(v[0] - v[j + 1])
            md = {u: stx.mean(v) for u, v in drops.items()}
            ranked = sorted(top, key=lambda u: -md[u])[:30]
            out["top"] = [{"neuron": f"L{l}:{n_}", "mean_drop": round(md[(l, n_)], 4), "per_fact": [round(x, 4) for x in drops[(l, n_)]],
                           "attr_rank": top.index((l, n_)) + 1} for l, n_ in ranked]
            out["nulls"] = [{"neuron": f"L{l}:{n_}", "mean_drop": round(md[(l, n_)], 4), "per_fact": [round(x, 4) for x in drops[(l, n_)]]} for l, n_ in nulls]
        f.write(json.dumps(out, ensure_ascii=False) + "\n"); f.flush()
        print(concept, out["n_recalled"], [(x["neuron"], x["mean_drop"]) for x in out.get("top", [])[:3]], flush=True)


def summary():
    M = [json.loads(l) for l in (OUT / "subject_maps.jsonl").read_text(encoding="utf-8").splitlines()]
    G = {r["concept"]: [x["neuron"] for x in (r.get("rank") or {}).get("top", [])[:30]] for r in gate()}
    lay = lambda u: int(u[1:].split(":")[0]); md = stx.median
    mapped = [m for m in M if m.get("top")]
    L = [f"Subject pass pilot: {len(M)} gate concepts; >= 3 recalled facts: {len(mapped)}  (S1 needs >= 9: {'HOLDS' if len(mapped) >= 9 else 'FAILS'})"]
    if mapped:
        l10 = [lay(x["neuron"]) for m in mapped for x in m["top"][:10]]
        ov = [len({x["neuron"] for x in m["top"][:30]} & set(G.get(m["concept"], []))) / 30 for m in mapped]
        L.append(f"top-10 layers median {md(l10)}; share of top-30 also in the answer-side v2 top-30: median {md(ov):.0%}  "
                 f"(S2 needs <= 15 and <= 10%: {'HOLDS' if md(l10) <= 15 and md(ov) <= 0.1 else 'FAILS'})")
        t5 = [x["mean_drop"] for m in mapped for x in m["top"][:5]]; nl = [abs(x["mean_drop"]) for m in mapped for x in m["nulls"]]
        L.append(f"top-5 drop median {md(t5):.3f} vs |null| median {md(nl):.3f}  (S3 needs >= 5x: {'HOLDS' if md(t5) >= 5 * max(md(nl), 1e-9) else 'FAILS'})")
        for m in mapped:
            L.append(f"  {m['concept']:<28} facts {m['n_recalled']}: " + ", ".join(f"{x['neuron']} {x['mean_drop']:.2f}" for x in m["top"][:5]))
    txt = "\n".join(L); (OUT / "summary.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    {"questions": questions, "map": map_, "summary": summary}[sys.argv[1]]()
