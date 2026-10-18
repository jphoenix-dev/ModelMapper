"""Concept maps from before loop v2 (15-17 Oct 2026) -> ledger pack. Ported from the map visualiser's
explainer/gen/map/sources/earlier_maps.py.

One experiment per batch (earlier_cr2, earlier_w2, ... earlier_fame5). Every concept in every batch keeps its own rows
(a concept mapped in several batches is a re-measurement on new questions; nothing is de-duplicated). Per concept:
recall, concept_stat rows (baseline, k_mean, k_median, k_flip, k_curve), every candidate in the rank file
('candidates_all200' if present, else the saved 'top' 30) as knockout_drop, and head knockouts where the batch has them.
Map-loop batches also list concepts that were not recalled (recall row only). No per-concept random null exists for
these runs, so noise_floor / threshold / passes are left empty on knockouts.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import REPO_RESULTS, PackWriter, concept_id  # noqa: E402

KS = [1, 2, 3, 5, 10, 20, 50]
# (experiment id, rank_save file, map_loop rows file or None, batch label, status)
BATCHES = [
    ("earlier_cr2", "concepts_v3/rank_save_cr2_exact.json", None, "CR-2", "exploratory"),
    ("earlier_w2", "concepts_w2/rank_save_w2_exact.json", None, "wave 2", "exploratory"),
    ("earlier_w3", "concepts_w3/rank_save_w3_exact.json", None, "wave 3", "exploratory"),
    ("earlier_w4", "concepts_w4/rank_save_w4_exact.json", None, "wave 4", "exploratory"),
    ("earlier_cr3", "concepts_cr3/rank_save_cr3_exact.json", None, "CR-3 (pre-registered)", "pre-registered (PREREG.md#cr-3)"),
    ("earlier_fame1", "concepts_fame/rank_save_fame_exact.json", None, "fame sample 1", "exploratory"),
    ("earlier_fame2", "concepts_fame2/rank_save_fame2_exact.json", None, "fame sample 2", "exploratory"),
    ("earlier_fame3", "concepts_fame3/rank_save_fame3_exact.json", None, "fame sample 3", "exploratory"),
    ("earlier_fame3m", "concepts_fame3/rank_save_fame3m.json", "concepts_fame3/map_loop_fame3m.json", "coverage-3 (map loop)", "exploratory"),
    ("earlier_fame5", "concepts_fame5/rank_save_fame5.json", "concepts_fame5/map_loop_fame5.json", "coverage-4 (map loop)", "exploratory"),
]
# ISSUES #35 (hidden log-odds cap) was fixed in commit 251140b, 2026-10-17 00:22 +0100. The '_exact' rank_save files were
# re-ranked with exact float64 log-odds; map_loop.py uses pipe.batch_logodds (exact) throughout, so the two map-loop
# batches (written 17 Oct 11:57 and 16:08, after the fix) are exact although their names lack '_exact'.
EXACT_NOTE = {True: "exact float64 log-odds ('_exact' re-rank, ISSUES #35 fix)",
              False: "exact float64 log-odds (map_loop.py uses pipe.batch_logodds; written after the ISSUES #35 fix)"}

METHOD = ("The same core measurement as loop v2 (recall questions -> attribution top 200 -> each candidate zeroed alone "
          "-> ranked by the drop in the answer's log-odds; top 30 kept), run as rank_save / map_loop before the v2 "
          "additions. No per-concept random-neuron null, no weights-based roles, and head knockouts only for the "
          "later batches. Where an '_exact' file exists its log-odds are exact float64 (ISSUES #35).")
CAVEATS = ["Files without '_exact' predate the log-odds cap fix (ISSUES #35): drops near saturation may be understated. "
           "(Checked at import: the two map-loop batches used here, rank_save_fame3m / rank_save_fame5, lack '_exact' but were "
           "written by map_loop.py after the fix with exact log-odds.)",
           "Prompts were written independently of run 1's, so a concept mapped in both is a genuine re-measurement on new questions.",
           "No per-concept random-neuron null in these runs: knockout rows have no noise_floor / threshold / passes. The k curve's "
           "random_drops (k random neurons zeroed together, map-loop batches only) is the only random control.",
           "Batches before fame sample 3 saved only the top 30 candidates (by drop) and no per-prompt drops or prompts; later "
           "batches keep all 200 attributed candidates with per-prompt drops.",
           "attr_rank in details is as in the source files: 0-based (0 = strongest attribution)."]


def main(force=False):
    files = []
    for _, rel, rows_rel, _, _ in BATCHES:
        files.append(REPO_RESULTS / rel)
        if rows_rel:
            files.append(REPO_RESULTS / rows_rel)
    with PackWriter("earlier_maps", inputs=files, code=[__file__], force=force) as pw:
        if pw.skip:
            print("earlier_maps: unchanged, skipped"); return
        for exp, rel, rows_rel, label, status in BATCHES:
            p = REPO_RESULTS / rel
            ranks = json.loads(p.read_text(encoding="utf-8"))
            rows = json.loads((REPO_RESULTS / rows_rel).read_text(encoding="utf-8")) if rows_rel else {}
            is_map_loop = rows_rel is not None
            short = exp.removeprefix("earlier_")
            pw.experiment(exp, title=f"Earlier concept map: {label}", method=METHOD, status=status,
                          date=time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime)),
                          script="src/map_loop.py" if is_map_loop else "src/rank_save.py",
                          reproduce=("RECALL_VARIANT=V3 CONCEPT_OUT_DIR=<dir> python src/map_loop.py <wave json> <tag>" if is_map_loop
                                     else "LOGODDS=exact python src/rank_save.py ...   (see RESULTS.md for the batch)"),
                          data_files=[f"repo_results:{rel}"] + ([f"repo_results:{rows_rel}"] if rows_rel else []),
                          caveats=CAVEATS, docs=["RESULTS.md", "PREREG.md#cr-3", "ISSUES.md"],
                          params={"batch": label, "n_concepts_rank_file": len(ranks), "n_concepts_rows_file": len(rows) or None,
                                  "logodds": EXACT_NOTE["_exact" in rel], "ks": KS if is_map_loop else None,
                                  "candidates_saved": "all 200" if any("candidates_all200" in (r or {}) for r in ranks.values()) else "top 30",
                                  "noise_floor": None, "attr_rank_base": 0,
                                  "k_criteria": {"k_mean": "smallest k (top-k by single drop, zeroed together) with mean drop >= 3 log-odds",
                                                 "k_median": "same with median drop >= 3", "k_flip": "same with >= 50% of correct prompts flipped (P < 0.5)",
                                                 "ks_tested": KS}},
                          description=f"Batch '{label}' of the pre-loop-v2 concept maps ({len(ranks)} mapped concepts).",
                          code_commit=None)
            ctx = f"batch={short};positions=all;prompts=recall"
            for c in list(dict.fromkeys(list(ranks) + list(rows))):
                rank = ranks.get(c) or {}; row = rows.get(c) or {}
                cid = pw.concept(concept_id("thing", c), c, "thing", aliases=[row.get("answer_form"), row.get("answer")],
                                 attributes={"category": row.get("category")})
                src = f"repo_results:{rel}#{c}" if rank else f"repo_results:{rows_rel}#{c}"
                rec = rank.get("recall", row.get("recall"))
                rec = int(rec) if rec is not None else None
                prompts = row.get("recall_prompts") or rank.get("prompts")
                afs = row.get("answer_form_scores") or {}
                n_written = len(next(iter(afs.values()))) if afs else None
                pw.evidence(experiment=exp, concept=cid, evidence_type="recall", metric="n_recall_prompts", value=rec, n=n_written,
                            value_text=row.get("status") or ("ok" if rank.get("top") else None), context=f"batch={short}", source=src,
                            details={"answer_token": rank.get("answer_token"), "answer": row.get("answer"), "answer_form": row.get("answer_form"),
                                     "answer_form_scores": afs or None, "discovered": row.get("discovered"), "category": row.get("category"),
                                     "matched_original": rank.get("matched_original"), "mismatch": rank.get("mismatch"),
                                     "prompts": prompts, "timing_s": row.get("timing_s"), "luna": row.get("luna"),
                                     "rows_source": f"repo_results:{rows_rel}#{c}" if row else None})
                if not rank or not rank.get("top"):
                    continue
                n = rec
                for m_ in ("baseline_mean_logodds", "k_mean", "k_median", "k_flip"):
                    if m_ not in rank and m_ not in row:
                        continue
                    v = rank.get(m_) if m_ in rank else row.get(m_)
                    det = None
                    if m_ == "baseline_mean_logodds":
                        det = {"base_per_prompt": rank.get("base_per_prompt") or row.get("base_per_prompt"),
                               "rows_file_value": row.get("baseline_mean_logodds")}
                    elif v is None:
                        det = {"note": f"null in source: no k in {KS} reached the {m_} criterion"}
                    pw.evidence(experiment=exp, concept=cid, evidence_type="concept_stat", metric=m_, value=v, n=n, context=f"batch={short}",
                                source=src, details=det)
                if row.get("drops") is not None:
                    pw.evidence(experiment=exp, concept=cid, evidence_type="concept_stat", metric="k_curve", value=None, n=n,
                                context=f"batch={short}", source=f"repo_results:{rows_rel}#{c}",
                                details={"ks": KS, "drops": row.get("drops"), "random_drops": row.get("random_drops")})
                ppd = rank.get("per_prompt_drop_all200") or {}
                all200 = rank.get("candidates_all200")
                cands = sorted(all200 or rank["top"], key=lambda x: -(x.get("single_drop") if x.get("single_drop") is not None else -1e9))
                for k, x in enumerate(cands):
                    pw.evidence(experiment=exp, unit=x["neuron"], concept=cid, evidence_type="knockout_drop", metric="mean_logodds_drop",
                                value=x.get("single_drop"), uom="logodds", n=n, rank=k + 1, context=ctx,
                                details={"attr_rank": x.get("attr_rank"), "act_answer_pos": x.get("act_answer_pos"),
                                         "per_prompt": ppd.get(x["neuron"]), "list": "candidates_all200" if all200 else "top30"},
                                source=src)
                hk = rank.get("head_knockout_mean") or {}; hpp = rank.get("head_knockout_per_prompt") or {}
                hattr = {h: j + 1 for j, h in enumerate(rank.get("heads_top50_by_attr") or [])}
                for k, (h, d) in enumerate(sorted(hk.items(), key=lambda kv: -kv[1])):
                    pw.evidence(experiment=exp, unit=h, concept=cid, evidence_type="knockout_drop", metric="mean_logodds_drop", value=d,
                                uom="logodds", n=n, rank=k + 1, context=ctx,
                                details={"attr_rank_1based": hattr.get(h), "per_prompt": hpp.get(h)}, source=src)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
