"""Blind neuron labels: the calibrated census of 1,000 random neurons, and the H-DEPTH labels (415 neurons)."""
from __future__ import annotations

from common import RESULTS, finding, nid, read_json

SOURCE = {
    "id": "labels",
    "title": "Blind labels: census (1,000 random neurons) and H-DEPTH (415 neurons)",
    "kind": "neuron labels",
    "status": "pre-registered (CEN, H-DEPTH)",
    "date": "2026-10-16",
    "method": (
        "GPT-6 Luna reads a neuron's 20 strongest firings on 10M tokens of FineWeb (each pole separately, blind: the "
        "dossier carries only a random code) and writes a label with a verdict (ONE / TWO_RELATED / MIXTURE / NONE). "
        "Jev grades the label on the held-out examples (coverage) and on 20 random web tokens (random fit). A pole is "
        "'clean' if Luna says ONE, coverage >= 0.6 (positive pole) or >= 0.5 (negative), and random fit < 0.25: bars "
        "calibrated so that only 8% / 4% of known two-job (spliced) dossiers pass."),
    "reproduce": "python src/label_batch.py Qwen/Qwen3-8B random:1000 ... ; python src/regrade.py ... ; "
                 "python src/calibrate_clean.py ...   (README, 'Main pipelines' step 2)",
    "docs": ["LABELLING.md", "PREREG.md", "RESULTS.md#16-oct-2026"],
    "caveats": ["Labels describe the INPUT side (what makes a neuron fire in web text), not what the neuron does to "
                "the output.",
                "The clean rate is closer to a floor than an estimate (REPORT §3)."],
}

BAR = {"pos": 0.6, "neg": 0.5}
RANDOM_CAP = 0.25


def load(ctx):
    p = RESULTS / "census_regrade_rubric.json"
    if p.exists():
        for key, u in read_json(p)["units"].items():
            n = nid(key)
            for pole in ("pos", "neg"):
                x = u.get(pole)
                if not x or not x.get("explanation"):
                    continue
                clean = bool(x.get("verdict") == "ONE" and x.get("coverage_soft", 0) >= BAR[pole]
                             and x.get("random_soft", 1) < RANDOM_CAP)
                ctx.add_finding(finding(
                    n, "label", x["explanation"], "labels", status="pre-registered (CEN)",
                    detail=f"{'positive' if pole == 'pos' else 'negative'} pole · census",
                    metric={"pole": pole, "verdict": x.get("verdict"), "coverage": x.get("coverage_soft"),
                            "random_fit": x.get("random_soft"), "clean": clean}))
    p = RESULTS / "hdepth_labels.json"
    if p.exists():
        for key, u in read_json(p)["units"].items():
            n = f"L{u['layer']}:{u['neuron']}"
            for pole in ("pos", "neg"):
                x = u.get(pole)
                if not x or not x.get("explanation"):
                    continue
                ctx.add_finding(finding(
                    n, "label", x["explanation"], "labels", status="pre-registered (H-DEPTH)",
                    detail=f"{'positive' if pole == 'pos' else 'negative'} pole · H-DEPTH",
                    metric={"pole": pole, "verdict": x.get("verdict"), "coverage": x.get("coverage"),
                            "random_fit": x.get("random_fit"), "clean": bool(x.get("clean"))}))
