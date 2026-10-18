"""Run 1 shared neurons (shared_neurons.py) + their feeders (feeders.py) -> ledger pack.

shared_neurons: every neuron in the top-10 of >= 3 run-1 maps (3,010). Per neuron, unit-level rows (concept None) of
evidence_type 'shared_use': n_users (with every user, its activation sign, single drop, role and the push tokens in
details), sign_agreement, section_same / p31_same against the script's chance level (noise_floor; passes = above chance),
and the role pattern across users (value_text, the script's classes).
feeders: per (target neuron N, user concept c) the 50 strongest earlier-layer feeders by activation x gradient of N's
activation (unit = feeder, unit2 = N, concept = c, value = signed score, rank by |score|); plus 'feeder_overlap' rows:
per N the mean Jaccard of its users' top-20 feeder sets (noise floor = chance median, recomputed exactly as feeders.py
did: same seeds, same sampling), and per concept the Jaccard between two different shared neurons' top-20 sets.
"""
import json
import random
import statistics as stx
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, read_jsonl, thing  # noqa: E402

RUN = DATA_ROOT / "loop_v2" / "run1"
SHARED, SHARED_TXT = RUN / "shared_neurons.json", RUN / "shared_neurons.txt"
FEED, FEED_TXT = RUN / "feeders.jsonl", RUN / "feeders.txt"
EXP_S, EXP_F = "loop_v2_shared_neurons", "loop_v2_feeders"
DOC = "RESULTS.md#18 Oct 2026: shared neurons across 4,194 maps - same job? same inputs? (run 1; research branch)"
TOPK, MIN_USERS = 10, 3


def role_pattern(roles):
    """The script's role classes (shared_neurons.py, ROLE across users)."""
    r = set(roles)
    return ("say for all" if r == {"say"} else "say for some" if "say" in r else
            "attribute/other only" if r <= {"attribute", "late-other"} else "early" if r == {"early"} else "mixed")


def jac(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


def replay_feeders_sampling(S, feeders):
    """Re-run feeders.py's sampling (seed 111) to recover the topic / wide groups and the chance distribution exactly."""
    rng = random.Random(111)
    topic = sorted(u for u, o in S.items() if o["n_users"] <= 10 and o["layer"] >= 24)
    wide = sorted(u for u, o in S.items() if o["n_users"] >= 100)
    pick = [(u, [x["concept"] for x in S[u]["users"]]) for u in rng.sample(topic, min(80, len(topic)))]
    pick += [(u, rng.sample([x["concept"] for x in S[u]["users"]], 6)) for u in rng.sample(wide, min(20, len(wide)))]
    order = [(u, c) for u, us in pick for c in us]
    keys = list(feeders); chance = []
    while len(chance) < 2000:
        a, b = rng.sample(keys, 2)
        if a[0] != b[0] and a[1] != b[1]:
            chance.append(jac(feeders[a], feeders[b]))
    return set(topic), set(wide), order, chance


def main(force=False):
    with PackWriter("shared_neurons", inputs=[SHARED, SHARED_TXT, FEED, FEED_TXT], code=[__file__], force=force) as pw:
        if pw.skip:
            print("shared_neurons: unchanged, skipped"); return
        S = json.loads(SHARED.read_text(encoding="utf-8"))
        F = list(read_jsonl(FEED))
        feeders = {(r["neuron"], r["concept"]): [a for a, _ in r["feeders_top50"][:20]] for r in F}
        topic, wide, order, chance = replay_feeders_sampling(S, feeders)
        replay_ok = order == [(r["neuron"], r["concept"]) for r in F]
        ch_med, ch_mean = stx.median(chance), stx.mean(chance)
        print(f"shared_neurons: feeders sampling replay matches file order: {replay_ok}; chance Jaccard median {ch_med:.3f} mean {ch_mean:.3f}")

        pw.experiment(EXP_S, title="Run 1 shared neurons: same job, related users? (3,010 neurons in the top-10 of >= 3 maps)",
                      method=("Every neuron in the top-10 (by single-knockout drop) of at least 3 of the 4,194 run-1 maps is a shared "
                              "neuron. For each, the sign of its answer-position activation across its users (same sign = same push "
                              "direction, since the push is fixed by its weights), its weights-based role per user, its 30 most-pushed "
                              "tokens, and whether its users share a Vital Articles section / first Wikidata P31 type more often than "
                              "2,000 random concept sets of the same size. CPU only, from saved run-1 records."),
                      status="exploratory", date="2026-10-18", script="src/loop_v2/shared_neurons.py",
                      reproduce="python src/loop_v2/shared_neurons.py D:/already-clean-data/loop_v2/run1",
                      data_files=["data:loop_v2/run1/shared_neurons.json", "data:loop_v2/run1/shared_neurons.txt"],
                      caveats=["'Topic' vs 'machinery' readings in RESULTS.md are informal readings of push tokens + users, not labels.",
                               "Roles are the weights-based roles saved by run 1 (top-30 only); users outside a neuron's top-30 role list have role None.",
                               "First Wikidata type is too fine-grained to show relatedness (33% vs 33% chance); sections are broad."],
                      docs=[DOC, "LOOP_V2.md"],
                      params={"topk": TOPK, "min_users": MIN_USERS, "chance_draws": 2000, "chance_seed": 101,
                              "n_shared": len(S), "summary_text": SHARED_TXT.read_text(encoding="utf-8")})
        pw.experiment(EXP_F, title="Run 1 shared neurons: do they have the same earlier feeders in each concept that uses them?",
                      method=("For 100 shared neurons (80 'topic' neurons: top-10 of 3-10 maps, layer >= 24; 20 widely shared: >= 100 "
                              "maps, 6 random users each; seed 111) and each user concept, every earlier-layer MLP neuron is scored by "
                              "activation x gradient of the target's answer-position activation (mean over the concept's recall prompts), "
                              "summed over positions; the 50 strongest by |score| are saved. Top-20 feeder sets are compared by Jaccard: "
                              "same neuron across users, same concept across shared neurons, and random pairs (chance)."),
                      status="pre-registered", date="2026-10-18", script="src/loop_v2/feeders.py",
                      reproduce="python src/loop_v2/feeders.py D:/already-clean-data/loop_v2/run1",
                      data_files=["data:loop_v2/run1/feeders.jsonl", "data:loop_v2/run1/feeders.txt"],
                      caveats=["Feeders are ranked by first-order attribution, not knockouts.",
                               "The same-concept baseline is small (27 pairs).",
                               "'topic' vs 'wide' is split by user count, not a test of meaning.",
                               "Prediction half wrong: widely shared neurons were predicted to vary feeders by concept; they share most (median Jaccard 0.445)."],
                      docs=[DOC],
                      params={"n_feeders_saved": 50, "jaccard_top": 20, "seed": 111, "groups": {"topic": "n_users <= 10 and layer >= 24 (80 sampled)",
                                                                                          "wide": "n_users >= 100 (20 sampled, 6 users each)"},
                              "prediction": ("topic neurons share feeders across their users well above chance (and above the same-concept "
                                             "baseline); widely shared machinery neurons share fewer, with feeders varying by concept"),
                              "prediction_outcome": "topic part held (80/80 above chance median); wide part failed (0.445, the highest overlap)",
                              "noise_floor": {"rule": "Jaccard of 2,000 random (different neuron, different concept) pairs, replayed with feeders.py's seed",
                                              "median": ch_med, "mean": ch_mean, "replay_matches_file_order": replay_ok},
                              "summary_text": FEED_TXT.read_text(encoding="utf-8")})

        # ---- shared neurons: unit-level stats
        for u, o in S.items():
            src = f"loop_v2/run1/shared_neurons.json#{u}"
            ctx = f"topk={TOPK};maps=loop_v2_run1"
            pw.evidence(experiment=EXP_S, unit=u, evidence_type="shared_use", metric="n_users", value=o["n_users"], n=o["n_users"],
                        threshold=MIN_USERS, passes=o["n_users"] >= MIN_USERS, context=ctx, source=src,
                        details={"users": o["users"], "roles": o["roles"], "push_tokens": o["push_tokens"],
                                 "users_concept_ids": [thing(pw, x["concept"]) for x in o["users"]]})
            pw.evidence(experiment=EXP_S, unit=u, evidence_type="shared_use", metric="sign_agreement", value=o["sign_agreement"],
                        uom="fraction", n=o["n_users"], threshold=1.0, passes=o["sign_agreement"] >= 1.0, context=ctx, source=src,
                        details={"signs": [1 if x["act"] > 0 else -1 for x in o["users"]]})
            pw.evidence(experiment=EXP_S, unit=u, evidence_type="shared_use", metric="role_pattern", value_text=role_pattern(o["roles"]),
                        n=o["n_users"], context=ctx, source=src, details={"roles": o["roles"]})
            for m_, ch in (("section_same", "chance_section"), ("p31_same", "chance_p31")):
                v = o[m_]
                pw.evidence(experiment=EXP_S, unit=u, evidence_type="shared_use", metric=m_, value=v, uom="fraction", n=o["n_users"],
                            noise_floor=o[ch], threshold=o[ch], passes=None if v is None else v > o[ch], context=ctx, source=src)

        # ---- feeders: every saved feeder
        for i, r in enumerate(F):
            N, c = r["neuron"], r["concept"]; cid = thing(pw, c)
            grp = "topic" if N in topic else "wide" if N in wide else "unknown"
            for k, (fu, sc) in enumerate(r["feeders_top50"]):
                pw.evidence(experiment=EXP_F, unit=fu, unit2=N, concept=cid, evidence_type="feeder", metric="act_x_grad_of_unit2_act",
                            value=sc, rank=k + 1, context=f"sample={grp};positions=all;target_pos=answer", source=f"loop_v2/run1/feeders.jsonl#{i}",
                            details={"target_act_mean": r["target_act_mean"], "in_top20": k < 20})
        # ---- feeder overlap: same neuron across its users
        users_of = defaultdict(list); neurons_of = defaultdict(list)
        for (N, c) in feeders:
            users_of[N].append(c); neurons_of[c].append(N)
        for N, cs in users_of.items():
            pairs = {f"{a}|{b}": jac(feeders[(N, a)], feeders[(N, b)]) for a, b in combinations(cs, 2)}
            v = stx.mean(pairs.values())
            grp = "topic" if N in topic else "wide" if N in wide else "unknown"
            pw.evidence(experiment=EXP_F, unit=N, evidence_type="feeder_overlap", metric="mean_jaccard_top20_across_users", value=v,
                        uom="fraction", n=len(pairs), noise_floor=ch_med, threshold=ch_med, passes=v > ch_med, context=f"sample={grp}",
                        details={"pairs": pairs, "users": cs, "chance_mean": ch_mean}, source=f"loop_v2/run1/feeders.jsonl#neuron={N}")
        # ---- feeder overlap: same concept, two different shared neurons
        for c, ns in neurons_of.items():
            cid = thing(pw, c)
            for a, b in combinations(ns, 2):
                v = jac(feeders[(a, c)], feeders[(b, c)])
                pw.evidence(experiment=EXP_F, unit=a, unit2=b, concept=cid, evidence_type="feeder_overlap", metric="jaccard_top20_same_concept",
                            value=v, uom="fraction", noise_floor=ch_med, threshold=ch_med, passes=v > ch_med,
                            source=f"loop_v2/run1/feeders.jsonl#concept={c}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
