"""Run 1: neurons that appear in several concepts' maps - same job each time? related users? (no model, no API)

James (18 Oct). 'Shared' = in the top-10 of >= 3 recalled concepts' maps (top-10: the neurons that matter most;
>= 3 users so consistency can be measured). For each shared neuron:
  sign      its activation sign at the answer position across its users (act_answer_pos): same sign = it pushes the
            SAME output direction every time (its job, mechanically, is fixed by its weights x this sign)
  role      per user, from the saved weights-based roles (answer's rank / z in its push): 'say' for one user and
            'attribute' / 'late-other' for others = a broad job that raises several related answers
  push      its 30 most-pushed tokens (weights; identical for every user with the same sign)
  coherence do its users share a Wikidata type (first P31) / Vital Articles section more often than random sets of
            the same size? (chance from 2,000 random draws per set size)
Output: <run>/shared_neurons.json (every shared neuron with users, signs, roles, push tokens), shared_neurons.txt
Usage: python src/loop_v2/shared_neurons.py <run_dir>
"""
import glob
import json
import random
import statistics as stx
import sys
from collections import Counter, defaultdict
from pathlib import Path
from modelmapper.core.settings import DATA_DIR

run = Path(sys.argv[1]); FR = (DATA_DIR / "loop_v2/frame")
TOPK, MIN_USERS = 10, 3


def main():
    F = {x["concept"]: x for x in json.loads((FR / "frame.json").read_text(encoding="utf-8"))["draw"]}
    V = json.loads((FR / "vital.json").read_text(encoding="utf-8"))
    recs = {}
    for b in glob.glob(str(run / "results/**/map/*/records.jsonl"), recursive=True):
        for line in Path(b).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["row"]["status"] == "ok":
                recs[r["concept"]] = r
    p31 = {c: (V["wd"].get(V["qid"].get(c) or "", {}).get("p31") or [None])[0] for c in recs}
    sec = {c: F[c]["section"] for c in recs}
    users = defaultdict(list)
    for c, r in recs.items():
        roles = {x["neuron"]: x for x in r["v2"]["roles_top30"]}
        for e in r["rank"]["top"][:TOPK]:
            u = e["neuron"]
            users[u].append({"concept": c, "act": e["act_answer_pos"], "single_drop": e["single_drop"],
                             "role": roles.get(u, {}).get("role"), "answer_z": roles.get(u, {}).get("answer_z"),
                             "job_tokens": roles.get(u, {}).get("job_tokens")})
    shared = {u: us for u, us in users.items() if len(us) >= MIN_USERS}
    concepts = list(recs); rng = random.Random(101)

    def frac_same(cs, key):
        vals = [key[c] for c in cs if key[c]]
        if len(vals) < 2:
            return None
        return Counter(vals).most_common(1)[0][1] / len(vals)
    chance = {}
    for n in sorted({len(us) for us in shared.values()}):
        chance[n] = (stx.mean(frac_same(rng.sample(concepts, n), p31) or 0 for _ in range(2000)),
                     stx.mean(frac_same(rng.sample(concepts, n), sec) or 0 for _ in range(2000)))
    out = {}; L = []
    for u, us in shared.items():
        signs = Counter(1 if x["act"] > 0 else -1 for x in us)
        roles = Counter(x["role"] for x in us)
        cs = [x["concept"] for x in us]
        out[u] = {"layer": int(u[1:].split(":")[0]), "n_users": len(us), "sign_agreement": signs.most_common(1)[0][1] / len(us),
                  "roles": dict(roles), "p31_same": frac_same(cs, p31), "section_same": frac_same(cs, sec),
                  "chance_p31": chance[len(us)][0], "chance_section": chance[len(us)][1],
                  "push_tokens": next((x["job_tokens"] for x in us if (x["act"] > 0) == (signs.most_common(1)[0][0] > 0) and x["job_tokens"]), None),
                  "users": us}
    n = len(out); md = stx.median
    L.append(f"Shared neurons (top-{TOPK} of >= {MIN_USERS} of {len(recs)} maps): {n}; users per neuron median {md(o['n_users'] for o in out.values())}, "
             f"max {max(o['n_users'] for o in out.values())}; layers {dict(sorted(Counter(o['layer'] // 6 * 6 for o in out.values()).items()))} (bins of 6)")
    sa = [o["sign_agreement"] for o in out.values()]
    L.append(f"SAME PUSH DIRECTION across users: sign agreement median {md(sa):.0%}; all users same sign in {sum(s == 1 for s in sa)}/{n} ({sum(s == 1 for s in sa) / n:.0%})")
    rc = Counter()
    for o in out.values():
        k = "say for all" if set(o["roles"]) == {"say"} else "say for some" if "say" in o["roles"] else \
            "attribute/other only" if set(o["roles"]) <= {"attribute", "late-other"} else "early" if set(o["roles"]) == {"early"} else "mixed"
        rc[k] += 1
    L.append(f"ROLE across users: {dict(rc)}  ('say' = the user's answer is in its top-10 pushed tokens)")
    pv = [o for o in out.values() if o["p31_same"] is not None]
    L.append(f"RELATED USERS: share of users with the most common Wikidata type median {md(o['p31_same'] for o in pv):.0%} vs chance "
             f"{md(o['chance_p31'] for o in pv):.0%}; above chance in {sum(o['p31_same'] > o['chance_p31'] for o in pv)}/{len(pv)}; "
             f"most common section median {md(o['section_same'] for o in out.values()):.0%} vs chance {md(o['chance_section'] for o in out.values()):.0%}")
    big = sorted(out.items(), key=lambda kv: -kv[1]["n_users"])[:25]
    L.append("\nmost-shared neurons (users; sign agreement; roles; pushed tokens; example users):")
    for u, o in big:
        L.append(f"  {u:<10} {o['n_users']:>4} users, sign {o['sign_agreement']:.0%}, {o['roles']} | push {(o['push_tokens'] or [])[:8]} | "
                 f"{[x['concept'] for x in o['users'][:8]]}")
    rng2 = random.Random(7)
    L.append("\nrandom shared neurons with 3-10 users:")
    for u, o in rng2.sample([kv for kv in out.items() if kv[1]["n_users"] <= 10], 25):
        L.append(f"  {u:<10} {o['n_users']:>2} users, sign {o['sign_agreement']:.0%}, {o['roles']} | push {(o['push_tokens'] or [])[:8]} | {[x['concept'] for x in o['users']]}")
    txt = "\n".join(L)
    (run / "shared_neurons.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (run / "shared_neurons.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
