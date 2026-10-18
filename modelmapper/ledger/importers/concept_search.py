"""Concept-search feasibility (concept_search_feasibility.py + concept_search_size_check.py) -> ledger pack.

Experiment-level results only (CV AUC, out-of-sample percentiles, enrichment, feature weights, the size check) in the
experiment's params; the dense per-neuron outputs (30 features and the concept-likeness score / within-layer percentile
for all 442,368 neurons) are registered as artifacts, not expanded into 442k evidence rows (they would make every
neuron of the model a 'unit with evidence'). The scores of the neurons that search_verify tested are copied into that
pack's details.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter  # noqa: E402

CS = DATA_ROOT / "concept_search"
SUMJ, SUMT, SIZE, FEAT, SCORES = CS / "summary.json", CS / "summary.txt", CS / "size_check.txt", CS / "features.npz", CS / "scores.npz"
EXP = "concept_search_feasibility"
DOC = "RESULTS.md#17 Oct 2026 (late evening): can we SEARCH for concept neurons instead of grinding? Feasibility (EXPLORATORY; research branch)"


def main(force=False):
    with PackWriter("concept_search", inputs=[SUMJ, SUMT, SIZE, FEAT, SCORES], code=[__file__], force=force) as pw:
        if pw.skip:
            print("concept_search: unchanged, skipped"); return
        S = json.loads(SUMJ.read_text(encoding="utf-8"))
        pw.experiment(EXP, title="Concept-search feasibility: can cheap model-internal profiles pick out concept neurons?",
                      method=("Every one of the 442,368 MLP neurons gets 30 cheap features, both signs: from the web-text atlas (peak, "
                              "spikiness, decay, distinct tokens / documents, word-start / capitalised-word / gate-open shares, corpus max "
                              "and min) and from the weights (max z of its push over the vocabulary, capitalised / word-start / Latin share "
                              "of its 50 most-pushed tokens, gate / up / down norms, cos(gate, up)). Logistic regression: positives = every "
                              "pre-coverage-4 mapped top-5 neuron (1,986), negatives = 5 same-layer neurons per positive in no top-30. "
                              "Size check: re-run with size-matched negatives, and without size features."),
                      status="pre-registered", date="2026-10-17", script="src/concept_search_feasibility.py (+ src/concept_search_size_check.py)",
                      reproduce="python src/concept_search_feasibility.py && python src/concept_search_size_check.py",
                      data_files=["data:concept_search/summary.json", "data:concept_search/summary.txt", "data:concept_search/size_check.txt",
                                  "data:concept_search/features.npz", "data:concept_search/scores.npz"],
                      caveats=["Criteria NOT MET, narrowly: CV AUC 0.842 (needed 0.85); new neurons at the 90.1th percentile (needed 90).",
                               "About a third of the signal is neuron size (gate norm, corpus peak); attribution, which found the positives, favours big neurons.",
                               "Negatives may include concept neurons for concepts never mapped (lowers scores).",
                               "Capitalised-push feature reflects the sample: we mapped named things.",
                               "Per-neuron scores are an artifact (scores.npz), not evidence rows."],
                      docs=[DOC],
                      params={"criteria": {"cv_auc_min": 0.85, "new_median_pct_min": 90},
                              "criteria_outcome": "NOT MET (AUC 0.842; median percentile 90.1)",
                              "results": S, "summary_text": SUMT.read_text(encoding="utf-8"),
                              "size_check_text": SIZE.read_text(encoding="utf-8"),
                              "size_check_reading": "a real ~34-percentile-point signal survives size matching and removing size features (82.9th vs 48.8th)"})
        pw.artifact(FEAT, experiment=EXP, fmt="npz", sha=True, description="30 cheap features for all 442,368 neurons (row = layer*12288 + index)",
                    keys={"X": "[442368, 30] float32 features", "names": "[30] feature names", "layer": "[442368] layer"})
        pw.artifact(SCORES, experiment=EXP, fmt="npz", sha=True, description="concept-likeness score and within-layer percentile for every neuron",
                    keys={"score": "[442368] logistic score", "pct": "[442368] percentile within its layer", "w": "[30] weights", "b": "bias",
                          "names": "[30] feature names"})


if __name__ == "__main__":
    main(force="--force" in sys.argv)
