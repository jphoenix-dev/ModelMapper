"""Luna firing judge (concept neurons' web-text firings: about / related / unrelated) -> ledger pack. Ported from the map
visualiser's explainer/gen/map/sources/web_firings.py.

Source: results/luna_firing_judge.json, jobs keyed '<shown neuron>|concept|<named neuron>' (a concept neuron judged
against its own concept) and '<random neuron>|control|<concept neuron>' (its paired random same-layer neuron's firings
judged against the concept neuron's concept name) - see src/luna_firing_judge.py. Each judged job -> three
web_firing_share rows (metric about / related / unrelated, value = share of the judged firings, n = number judged).
Controls: unit = the random neuron, unit2 = the concept neuron it controls for, context control=random. Skipped jobs
(too few firings) -> one row, metric 'skipped', value_text = the reason.
A concept neuron found for several concepts was judged against all their names at once ('A / B / C'): such a job gets
its rows once per named concept, with context named=<k> and details.named_concepts (the share is for the joint name).
"""
import json
import statistics as stx
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import REPO_RESULTS, PackWriter, concept_id  # noqa: E402

SRC = REPO_RESULTS / "luna_firing_judge.json"
EXP = "luna_firing_judge"
GRADES = {"A": "about", "B": "related", "C": "unrelated"}


def main(force=False):
    with PackWriter("web_firings", inputs=[SRC], code=[__file__], force=force) as pw:
        if pw.skip:
            print("web_firings: unchanged, skipped"); return
        d = json.loads(SRC.read_text(encoding="utf-8"))
        jobs = d["jobs"]
        concept_of = {k.split("|")[0]: j.get("concept") for k, j in jobs.items() if k.split("|")[1] == "concept"}
        ctrl_shares = {g: [] for g in GRADES.values()}
        for k, j in jobs.items():
            if j.get("control") and j.get("labels"):
                c = Counter(j["labels"]); t = sum(c.values())
                for a, g in GRADES.items():
                    ctrl_shares[g].append(c.get(a, 0) / t)
        pw.experiment(EXP, title="Concept neurons in ordinary web text (Luna firing judge)",
                      method=("For concept neurons found by the maps, 12 of their firings in FineWeb are sampled across the "
                              "activation range and shown to GPT-6 Luna with the concept named. Each snippet is judged A (about "
                              "the concept), B (related: same topic, a closely related thing or a broader category) or C "
                              "(unrelated). Random neurons judged against the same concepts are the control (95% unrelated)."),
                      status="exploratory", date="2026-10-17", script="src/luna_firing_judge.py",
                      reproduce="python src/luna_firing_judge.py   (research/attention branch)",
                      data_files=["repo_results:luna_firing_judge.json"],
                      caveats=["Concept neurons are mostly polysemantic in the wild (12% dedicated, 22% ingredient, 57% mostly "
                               "unrelated) but specific in use: off-topic firings are harmless (interference test).",
                               "No per-row noise floor: the control rows (context control=random) are the null; their pooled mean "
                               "shares are in params.control_mean_share.",
                               "Neurons found for several concepts were judged against all the names at once ('A / B / C'); their "
                               "rows repeat once per named concept with context named=<k>, and 'about' means about any of them."],
                      docs=["RESULTS.md#17-oct-2026-night-luna-reads-the-concept-neurons-web-text-firings"],
                      params={"grades": GRADES, "snippets_per_job": 12,
                              "control_mean_share": {g: (stx.mean(v) if v else None) for g, v in ctrl_shares.items()},
                              "n_control_jobs_judged": len(ctrl_shares["about"]), "n_jobs": len(jobs),
                              "n_jobs_skipped": sum(1 for j in jobs.values() if not j.get("labels")),
                              **{k: v for k, v in d.items() if k != "jobs"}},
                      description="Share of each concept neuron's sampled web firings that Luna judges about / related to / unrelated to its concept.",
                      code_commit=None)
        for k, j in jobs.items():
            shown, kind, named = k.split("|")         # key = <neuron whose firings were shown>|kind|<concept neuron that gave the name>
            control = kind == "control"
            cname = j.get("concept") or concept_of.get(named)
            names = cname.split(" / ") if cname else [None]
            unit, unit2 = shown, (named if control else None)
            src = f"repo_results:luna_firing_judge.json#jobs.{k}"
            c = Counter(j.get("labels") or []); t = sum(c.values())
            for ci, name in enumerate(names):
                cid = pw.concept(concept_id("thing", name), name, "thing") if name else None
                ctx = ";".join(x for x in ("control=random" if control else None, f"named={len(names)}" if len(names) > 1 else None) if x) or None
                first = ci == 0                      # full job record (snippets, raw) once per job, on its first row
                if not j.get("labels"):
                    pw.evidence(experiment=EXP, unit=unit, unit2=unit2, concept=cid, context=ctx, evidence_type="web_firing_share",
                                metric="skipped", value_text=j.get("skipped") or "no labels", n=0,
                                details={**j, "named_concepts": names if len(names) > 1 else None}, source=src)
                    continue
                for g_letter, g in GRADES.items():
                    det = {"count": c.get(g_letter, 0), "labels": j["labels"], "valid": j.get("valid"),
                           "theme_of_unrelated": j.get("theme_of_unrelated"), "layer": j.get("layer"), "control_flag": j.get("control"),
                           "named_concepts": names if len(names) > 1 else None}
                    if first and g == "about":       # snippets + raw once per job, to avoid copying them onto every row
                        det.update(snippets=j.get("snippets"), raw=j.get("raw"))
                    else:
                        det["snippets_on"] = "the job's first 'about' row (same source)"
                    pw.evidence(experiment=EXP, unit=unit, unit2=unit2, concept=cid, context=ctx, evidence_type="web_firing_share",
                                metric=g, value=c.get(g_letter, 0) / t, uom="fraction", n=t, details=det, source=src)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
