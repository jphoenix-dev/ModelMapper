"""Are attribute jobs reusable parts? Can a neuron's job, read from the WEIGHTS, pick out which concepts USE it
(from the knockout maps)? Two independent sources: if they agree, the jobs are legible, reusable parts.
(GPU for the logit lens; Luna as a blind judge; ~$0.05)

James (17 Oct): after job_overlap.py showed attribute neurons ('Georgia', '1970s', 'Bay Area') alongside 'say it' ones.
1. Candidates: late neurons (layer >= 20) in the top-30 maps of >= 3 of the 914 mapped concepts and of < 15% of them
   (more = generic machinery). Job = the 30 tokens its direct push raises most (logit lens; push sign = the majority
   sign of its activation at the answer position across the concepts that use it).
2. Sample 200 candidates at random (seed 21). For each: k = min(5, users) users at random + (10 - k) distractors at
   random from concepts that don't have it in their top-30, shuffled. Luna sees ONLY the job tokens and the 10 names,
   is told exactly k use it, and picks k.
3. Control (same lists): the job tokens of a different random candidate. Chance = k/10.
Criterion (stated before running): hit rate >= 2x chance and >= 2x the control, sign test real > control p < 0.05,
and >= 1.5x chance on 'non-lexical' users (users whose name and answer don't appear among the job tokens).
Output: results/attribute_sharing.json (every candidate, job tokens, users, prompts, Luna replies), .txt
Usage: python src/attribute_sharing.py
"""
import glob
import json
import math
import random
import re
import statistics as stx
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import oai
from config import DEFAULT_MODEL
from neurons import load, mlps

ROOT = Path(__file__).resolve().parents[1]
LATE, GENERIC, MIN_USERS, N_SAMPLE, K, LIST = 20, 0.15, 3, 200, 5, 10
OUT = ROOT / "results/attribute_sharing.json"
SYSTEM = "You are a careful analyst of language-model internals. Reply in JSON only."


def maps():
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json"))):
        if "rank_save_test" in f or f.endswith("_exact.json"):
            continue
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" in v and c not in out:
                out[c] = {"answer": v["answer_token"], "top": [(e["neuron"], float(e["act_answer_pos"])) for e in v["top"][:30]]}
    return out


def build():
    M = maps(); n = len(M)
    users = defaultdict(list)
    for c, v in M.items():
        for u, a in v["top"]:
            users[u].append((c, a))
    cand = {u: us for u, us in users.items() if int(u.split(":")[0][1:]) >= LATE and MIN_USERS <= len(us) < GENERIC * n}
    model, tok = load(DEFAULT_MODEL)
    WU = model.lm_head.weight.detach().float(); g = model.model.norm.weight.detach().float(); downs = [m.down_proj.weight.detach() for m in mlps(model)]
    jobs = {}
    for u, us in cand.items():
        L, i = (int(x) for x in u[1:].split(":"))
        sign = 1.0 if sum(a > 0 for _, a in us) >= len(us) / 2 else -1.0
        lg = (sign * downs[L][:, i].float() * g) @ WU.T
        jobs[u] = {"sign": sign, "tokens": [tok.decode([t]) for t in lg.topk(30).indices.tolist()], "users": [c for c, _ in us]}
    return M, jobs


def prompt(tokens, names, k):
    return ("A component inside a language model, when active, pushes the model towards these output tokens "
            f"(most strongly first):\n{json.dumps(tokens, ensure_ascii=False)}\n\n"
            f"Below are {len(names)} concepts the model can name. Exactly {k} of them make use of this component when the model "
            f"recalls them. Which {k}? Judge by what the tokens have in common (a place, era, role, kind of name, spelling, "
            "topic...), not just exact word matches.\n"
            + "\n".join(f"{j + 1}. {nm}" for j, nm in enumerate(names))
            + f'\n\nReply as {{"picks": [{k} numbers], "job": "<a few words naming what the component does>"}}')


def lexical(name, answer, tokens):
    """Surface match: a job token equals a word of the name/answer, starts one (spelling jobs: 'El' -> Elvis), or (>= 4
    letters) sits inside one."""
    toks = {t.strip(" \t_-.:/=*{(").lower() for t in tokens}
    toks = {t for t in toks if len(t) >= 2 and t.isalpha()}
    words = {w.lower() for w in re.findall(r"\w{2,}", name)} | {answer.strip().lower()}
    return any(w in toks or any(w.startswith(t) for t in toks) or any(t in w for t in toks if len(t) >= 4) for w in words)


def main():
    M, jobs = build()
    rng = random.Random(21); names = list(M)
    sample = rng.sample(sorted(jobs), min(N_SAMPLE, len(jobs)))
    trials = []
    for u in sample:
        us = jobs[u]["users"]; k = min(K, len(us))
        chosen = rng.sample(us, k); others = [c for c in names if c not in set(us)]
        lst = chosen + rng.sample(others, LIST - k); rng.shuffle(lst)
        ctrl = rng.choice([x for x in sorted(jobs) if x != u])
        trials.append({"neuron": u, "k": k, "list": lst, "members": chosen, "control_neuron": ctrl,
                       "lexical_members": [c for c in chosen if lexical(c, M[c]["answer"], jobs[u]["tokens"])]})

    def ask(tr, which):
        toks = jobs[tr["neuron"] if which == "real" else tr["control_neuron"]]["tokens"]
        p = prompt(toks, tr["list"], tr["k"])
        for _ in range(3):
            try:
                r = oai.chat_json(p, SYSTEM)
                picks = [int(x) for x in r.get("picks", [])][: tr["k"]]
                return {"prompt": p, "reply": r, "picked": [tr["list"][j - 1] for j in picks if 1 <= j <= len(tr["list"])]}
            except Exception as e:                       # recorded, retried; a failure after 3 tries is kept as such
                err = repr(e)
        return {"prompt": p, "error": err, "picked": []}

    with ThreadPoolExecutor(8) as ex:
        real = list(ex.map(lambda t: ask(t, "real"), trials)); ctrl = list(ex.map(lambda t: ask(t, "control"), trials))
    for t, r, c in zip(trials, real, ctrl):
        t["real"] = r; t["control"] = c
        t["hits_real"] = len(set(r["picked"]) & set(t["members"])); t["hits_ctrl"] = len(set(c["picked"]) & set(t["members"]))
        nl = [m for m in t["members"] if m not in t["lexical_members"]]
        t["nonlex_n"] = len(nl); t["nonlex_hits"] = len(set(r["picked"]) & set(nl))
    ok = [t for t in trials if "error" not in t["real"] and "error" not in t["control"]]
    hr = sum(t["hits_real"] for t in ok) / sum(t["k"] for t in ok); hc = sum(t["hits_ctrl"] for t in ok) / sum(t["k"] for t in ok)
    chance = sum(t["k"] * t["k"] / LIST for t in ok) / sum(t["k"] for t in ok)
    w = sum(t["hits_real"] > t["hits_ctrl"] for t in ok); l_ = sum(t["hits_real"] < t["hits_ctrl"] for t in ok); kk = w + l_
    p = sum(math.comb(kk, i) for i in range(w, kk + 1)) / 2 ** kk if kk else 1.0
    nln = sum(t["nonlex_n"] for t in ok); nlh = sum(t["nonlex_hits"] for t in ok)
    # chance for a non-lexical member = k/10 for its trial (picks are k of 10)
    nl_chance = sum(t["nonlex_n"] * t["k"] / LIST for t in ok) / max(1, nln)
    users_n = Counter(min(len(j["users"]), 20) for j in jobs.values())
    L = [f"Attribute sharing: {len(jobs)} candidate neurons (layer >= {LATE}, in the top-30 of {MIN_USERS} to <{GENERIC:.0%} of {len(M)} concepts); "
         f"users per neuron (capped display at 20): {dict(sorted(users_n.items()))}",
         f"trials: {len(trials)} sampled, {len(ok)} scored (errors {len(trials) - len(ok)}); Luna {oai.usage}",
         f"hit rate (picked members / members): real {hr:.1%} | control (another neuron's job) {hc:.1%} | chance {chance:.1%}",
         f"per neuron: real beats control {w}, control beats real {l_}, ties {len(ok) - kk}; sign test p = {p:.2g}",
         f"non-lexical members (name/answer not among the job tokens): {nlh}/{nln} = {nlh / max(1, nln):.1%} vs chance {nl_chance:.1%}; "
         f"lexical members {sum(len(t['lexical_members']) for t in ok)}"]
    crit = hr >= 2 * chance and hr >= 2 * hc and p < 0.05 and nlh / max(1, nln) >= 1.5 * nl_chance
    L.append(f"CRITERION {'MET' if crit else 'NOT MET'}")
    L.append("\nexamples (Luna's job name | job tokens | users picked correctly / members):")
    for t in random.Random(4).sample(ok, min(15, len(ok))):
        L.append(f"  {t['neuron']:<10} {t['real']['reply'].get('job', '')!r} | {jobs[t['neuron']]['tokens'][:8]} | "
                 f"{sorted(set(t['real']['picked']) & set(t['members']))} / {t['members']}")
    txt = "\n".join(L)
    OUT.write_text(json.dumps({"jobs": jobs, "trials": trials, "usage": dict(oai.usage)}, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results/attribute_sharing.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
