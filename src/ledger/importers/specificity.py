"""Question-side test + specificity gradient (specificity.py) -> ledger pack.

One 'specificity_gradient' row per (question set, ablation): concept = the landmark whose neurons were zeroed (the
source; for the 'other5' control, the NEXT source whose top-5 was zeroed), concept2 = the concept the questions are about
(the source itself for its naming / fact sets, else the neighbour), value = mean log-odds drop over the recalled
questions, context = ablation, question kind and neighbour level. noise_floor = max |drop| of the two random-5 controls
on the same question set; threshold = that noise floor and passes = drop > it (importer's rule: the script defines the
random-5 null but no cut). The random-5 rows themselves carry no noise floor. details = zeroed units, n questions,
level and its definition, the source's Wikidata qid / type / section, and Luna's candidate questions for the set.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, concept_id, read_jsonl, thing  # noqa: E402

SP = DATA_ROOT / "specificity"
RES, NEI, QS, SUM, LOG = SP / "results.jsonl", SP / "neighbours.json", SP / "questions.json", SP / "summary.txt", SP / "run.log"
EXP = "specificity_gradient"
DOC = "RESULTS.md#18 Oct 2026: question-side test + specificity gradient - 'say it' vs meaning, and how specific (research branch)"
LEVELS = {"L0": "the source itself", "L1": "same place: notable physical place within 3 km (Wikidata coordinates)",
          "L2": "same kind: same 'instance of' (P31) as the source, > 500 km away", "L3": "same Vital Articles section, different P31",
          "L4": "unrelated: a different Vital Articles section"}
ABL = {"src_k": "the source's top-k_mean neurons (k capped 1..5)", "src5": "the source's top-5", "src5_late": "its top-5 in layers >= 20",
       "src5_early": "its top-5 in layers < 20", "rand5_a": "5 random neurons from the same layers (draw a)",
       "rand5_b": "5 random neurons from the same layers (draw b)", "other5": "the next source's top-5 (specificity control)"}


def main(force=False):
    with PackWriter("specificity", inputs=[RES, NEI, QS, SUM, LOG], code=[__file__], force=force) as pw:
        if pw.skip:
            print("specificity: unchanged, skipped"); return
        N = json.loads(NEI.read_text(encoding="utf-8")); Q = json.loads(QS.read_text(encoding="utf-8"))
        srcs = list(N)
        pw.experiment(EXP, title="Question-side test + specificity gradient: do a landmark's naming neurons carry facts about it, and its neighbours?",
                      method=("15 mapped landmarks; each source's top-5 / top-k_mean neurons (from its map) are zeroed at every position, "
                              "against an as-is copy in the same pass, and the mean log-odds drop is measured on its own naming questions, "
                              "Luna-written FACT questions about it (name in the question, short answer), and naming questions for 3 "
                              "Wikidata neighbours at each of 4 distances (same place / same kind / same section / unrelated). Only "
                              "questions Qwen answers (P >= 0.2) are used. Controls: 2 random same-layer 5-sets, another landmark's top-5."),
                      status="pre-registered", date="2026-10-18", script="src/loop_v2/specificity.py",
                      reproduce="python src/loop_v2/specificity.py neighbours && ... questions && ... run && ... summary",
                      data_files=["data:specificity/results.jsonl", "data:specificity/neighbours.json", "data:specificity/questions.json",
                                  "data:specificity/summary.txt", "data:specificity/run.log"],
                      caveats=["Small n per level (many neighbours had < 2 recalled naming questions; sets with < 2 are absent).",
                               "Fact answers are Luna-written; one forced 'answer in as few words as possible' format for facts.",
                               "15 sources, all landmarks.",
                               "Which candidate questions passed the recall filter was not saved (only n_questions); details list all Luna candidates.",
                               "Exceptions with facts damaged too (Burj Khalifa, St. Peter's, Machu Picchu) not yet explained."],
                      docs=[DOC, "RESEARCH_STATE.md"],
                      params={"levels": LEVELS, "ablations": ABL, "neighbours_per_level": 3, "min_questions": 2, "recall_min_p": 0.2, "seed_run": 152,
                              "seed_neighbours": 151, "naming_suffix": "Answer with just the name.", "fact_suffix": "Answer in as few words as possible.",
                              "noise_floor": "max |mean drop| of rand5_a / rand5_b on the same question set (2 draws)",
                              "threshold_rule": "threshold = noise floor; passes = drop > threshold (importer's choice; the script states no cut)",
                              "predictions": ["late 'say' neurons leave facts intact while naming drops; early neurons may not",
                                              "damage falls with distance L1 -> L4; 'say' neurons stay specific",
                                              "attribute neurons partly damage neighbours that share the attribute"],
                              "prediction_outcome": "held: naming 3.62 vs facts 0.03 (k); late 6.44 naming / 0.30 facts, early 1.44 / 0.97; L1 0.17 -> L2 0.10 -> L3 0.00 -> L4 -0.02 with shared-attribute exceptions",
                              "neighbours": N, "summary_text": SUM.read_text(encoding="utf-8")})
        for i, r in enumerate(read_jsonl(RES)):
            s = r["source"]; sset = r["set"]
            if sset == "src_name":
                lvl, qkind, about, qkey = "L0", "naming", s, f"name|{s}"
            elif sset == "src_fact":
                lvl, qkind, about, qkey = "L0", "facts", s, f"fact|{s}"
            else:
                lvl, about = sset.split("|", 1); qkind, qkey = "naming", f"name|{about}"
            src_c = pw.concept(concept_id("thing", s), s, "thing", qid=N[s].get("qid"),
                               attributes={"p31": N[s].get("p31"), "section": N[s].get("section")})
            about_c = thing(pw, about)
            other_src = srcs[(srcs.index(s) + 1) % len(srcs)]
            drops = r["drops"]
            nf = max(abs(drops[a]) for a in ("rand5_a", "rand5_b") if a in drops) if any(a in drops for a in ("rand5_a", "rand5_b")) else None
            for a, v in drops.items():
                zeroed = thing(pw, other_src) if a == "other5" else src_c
                ctl = a.startswith("rand5")
                pw.evidence(experiment=EXP, concept=zeroed, concept2=about_c, evidence_type="specificity_gradient", metric="mean_logodds_drop",
                            value=v, uom="logodds", n=r["n_questions"], noise_floor=None if ctl else nf, threshold=None if ctl else nf,
                            passes=None if (ctl or nf is None) else v > nf,
                            context=f"ablation={a};questions={qkind};level={lvl};positions=all",
                            source=f"specificity/results.jsonl#{i}",
                            details={"zeroed_units": r["units"].get(a), "ablation_desc": ABL.get(a), "question_set": sset, "question_set_source": s,
                                     "level": lvl, "level_desc": LEVELS[lvl], "neighbour": None if lvl == "L0" else about,
                                     "zeroed_neurons_belong_to": other_src if a == "other5" else s,
                                     "all_drops_this_set": drops, "candidate_questions_luna": Q.get(qkey),
                                     "question_source": "luna (questions.json)" if qkey in Q else "the source's map recall prompts (first 12)"})


if __name__ == "__main__":
    main(force="--force" in sys.argv)
