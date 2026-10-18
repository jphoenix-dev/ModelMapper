"""Blind input-side neuron labels -> ledger pack. Ported from the map visualiser's explainer/gen/map/sources/labels.py.

Two sets: the calibrated census of 1,000 random neurons (results/census_regrade_rubric.json, rubric soft re-grade) and the
H-DEPTH labels of 415 neurons (results/hdepth_labels.json). One `label` row per (neuron, pole) with a label; no concept.
value_text = Luna's explanation, value = its coverage on held-out examples, passes = 'clean'.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import REPO_RESULTS, PackWriter  # noqa: E402

CENSUS = REPO_RESULTS / "census_regrade_rubric.json"
HDEPTH = REPO_RESULTS / "hdepth_labels.json"
BAR = {"pos": 0.6, "neg": 0.5}           # census clean bars (calibrated on spliced two-job dossiers, ISSUES #25)
RANDOM_CAP = 0.25
HD_BAR, HD_RANDOM_CAP = 0.8, 0.2           # H-DEPTH's own clean rule (src/label_batch.py): coverage >= 0.8, random fit <= 0.2

METHOD = ("GPT-6 Luna reads a neuron's 20 strongest firings on 10M tokens of FineWeb (each pole separately, blind: the "
          "dossier carries only a random code) and writes a label with a verdict (ONE / TWO_RELATED / MIXTURE / NONE). "
          "Jev grades the label on the held-out examples (coverage) and on 20 random web tokens (random fit).")
CAVEATS = ["Labels describe the INPUT side (what makes a neuron fire in web text), not what the neuron does to the output.",
           "The clean rate is closer to a floor than an estimate (REPORT §3)."]
DOCS = ["LABELLING.md", "PREREG.md", "RESULTS.md#16-oct-2026"]
REPRO = ("python src/label_batch.py Qwen/Qwen3-8B random:1000 ... ; python src/regrade.py ... ; "
         "python src/calibrate_clean.py ...   (README, 'Main pipelines' step 2)")


def main(force=False):
    with PackWriter("labels", inputs=[CENSUS, HDEPTH], code=[__file__], force=force) as pw:
        if pw.skip:
            print("labels: unchanged, skipped"); return
        cen = json.loads(CENSUS.read_text(encoding="utf-8"))
        hd = json.loads(HDEPTH.read_text(encoding="utf-8"))
        pw.experiment("labels_census", title="Blind labels: census of 1,000 random neurons (rubric soft re-grade)",
                      method=METHOD + (" A pole is 'clean' if Luna says ONE, soft coverage >= 0.6 (positive pole) or >= 0.5 "
                                       "(negative), and soft random fit < 0.25: bars calibrated so that only 8% / 4% of known "
                                       "two-job (spliced) dossiers pass."),
                      status="pre-registered (CEN)", date="2026-10-16", script="src/regrade.py", reproduce=REPRO,
                      data_files=["repo_results:census_regrade_rubric.json"], caveats=CAVEATS, docs=DOCS,
                      params={"clean_rule": {"verdict": "ONE", "coverage_soft_min": BAR, "random_soft_max_exclusive": RANDOM_CAP},
                              **{k: v for k, v in cen.items() if k != "units"}, "n_units": len(cen["units"])},
                      description="Census labels, both poles, with the soft rubric grades (coverage and random fit).", code_commit=None)
        pw.experiment("labels_hdepth", title="Blind labels: H-DEPTH (415 neurons)",
                      method=METHOD + (" H-DEPTH grading: coverage = share of held-out examples graded >= 0.5, random fit = "
                                       "share of random snippets graded >= 0.5; clean = verdict ONE, coverage >= 0.8, random "
                                       "fit <= 0.2 (src/label_batch.py)."),
                      status="pre-registered (H-DEPTH)", date="2026-10-16", script="src/label_batch.py", reproduce=REPRO,
                      data_files=["repo_results:hdepth_labels.json"], caveats=CAVEATS + [
                          "7 poles were excluded for API failures (details.excluded); they keep Luna's label but have no grade."],
                      docs=DOCS + ["PREREG.md#hypothesis-h-depth-stages-by-depth-registered-16-oct-0035-before-the-test-was-run"],
                      params={"clean_rule": {"verdict": "ONE", "coverage_min": HD_BAR, "random_fit_max": HD_RANDOM_CAP},
                              **{k: v for k, v in hd.items() if k != "units"}},
                      description="H-DEPTH labels, both poles, with hit-rate grades.", code_commit=None)
        for key, u in cen["units"].items():
            unit = (u["layer"], u["neuron"])
            for pole in ("pos", "neg"):
                x = u.get(pole)
                if not x:
                    continue
                clean = bool(x.get("verdict") == "ONE" and x.get("coverage_soft", 0) >= BAR[pole] and x.get("random_soft", 1) < RANDOM_CAP)
                pw.evidence(experiment="labels_census", unit=unit, evidence_type="label", metric="coverage_soft",
                            value=x.get("coverage_soft"), value_text=x.get("explanation"), uom="fraction", n=x.get("n_own"),
                            threshold=BAR[pole], passes=clean, context=f"pole={pole};set=census",
                            details={"verdict": x.get("verdict"), "coverage": x.get("coverage_soft"), "coverage_hits": x.get("coverage_hits"),
                                     "random_fit": x.get("random_soft"), "random_hits": x.get("random_hits"), "n_own": x.get("n_own"),
                                     "random_cap": RANDOM_CAP, "clean": clean},
                            source=f"repo_results:census_regrade_rubric.json#units.{key}.{pole}")
        for key, u in hd["units"].items():
            unit = (u["layer"], u["neuron"])
            for pole in ("pos", "neg"):
                x = u.get(pole)
                if not x:
                    continue
                excl = x.get("excluded")
                pw.evidence(experiment="labels_hdepth", unit=unit, evidence_type="label", metric="coverage",
                            value=x.get("coverage"), value_text=x.get("explanation"), uom="fraction",
                            threshold=HD_BAR, passes=None if excl else bool(x.get("clean")), context=f"pole={pole};set=hdepth",
                            details={"verdict": x.get("verdict"), "coverage": x.get("coverage"), "random_fit": x.get("random_fit"),
                                     "random_cap": HD_RANDOM_CAP, "clean": None if excl else bool(x.get("clean")), "excluded": excl},
                            source=f"repo_results:hdepth_labels.json#units.{key}.{pole}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
