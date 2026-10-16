"""Marker concepts: a few concept -> neuron associations that a correct install should rediscover.

An answer key (answer_keys/<model>.json) lists, for a handful of concepts the model reliably recalls, one "marker"
neuron each: a neuron that came near the top of the concept's attribution in two independently written prompt sets,
knocked out alone it cut P(answer) clearly both times, and its label (written blind from its top firings on web text)
names the concept. scripts/make_answer_key.py builds keys; scripts/verify_answer_key.py checks an install against one.

The measurement is the concept pipeline's own: V3 question-format recall prompts, activation x gradient attribution
over every MLP neuron, then each of the top attributed neurons knocked out alone (set to 0 at every position) and the
mean P(answer) over the recall prompts re-measured. No steering, so a concept takes seconds on a small model.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def setup_env(out_dir, discovery: bool):
    """Must run before concept_pipeline / pipeline_fast are imported: they read these at import time."""
    os.environ["RECALL_VARIANT"] = "V3"
    os.environ["ANSWER_DISCOVERY"] = "1" if discovery else "0"
    os.environ["CONCEPT_OUT_DIR"] = str(out_dir)


def set_out_dir(out_dir):
    """Point already-imported pipeline modules at another spec folder (each prompt set gets its own)."""
    import concept_pipeline as cp
    import pipeline_fast
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cp.OUT = out_dir
    pipeline_fast.OUT = out_dir


def run_concept(pipe, concept: str, top_k: int = 20) -> dict:
    """Recall -> attribution -> single-neuron knockouts for the top_k attributed neurons (same measures as CR-2)."""
    spec, tid, recall, _ = pipe.make_prompts(concept)
    row = {"concept": concept, "answer": spec["answer"], "answer_form": getattr(pipe, "answer_form", None),
           "recall": len(recall), "recall_total": len(spec.get("recall_v3") or spec["recall"])}
    if len(recall) < 5:
        return {**row, "status": "not_recalled"}
    rp = [q for q, _ in recall]
    base = sum(pipe.probs(rp, tid)) / len(rp)
    A = pipe.attribute(rp, tid)
    la = A["last_act"]
    neurons = []
    for i, (l, n) in enumerate(A["top"][:top_k]):
        p = sum(pipe.probs(rp, tid, [(l, n, 0.0)])) / len(rp)
        neurons.append({"rank": i + 1, "layer": l, "neuron": n, "drop": round(base - p, 4),
                        "sign": "neg" if la[l, n] < 0 else "pos"})
    del A
    return {**row, "status": "ok", "baseline_p": round(base, 4), "neurons": neurons, "spec": spec}


def label_units(model_name: str, units, out_dir, use_jev: bool = False) -> dict:
    """Blind labels for (layer, neuron) units from this install's atlas (its own corpus): {"L.._N..": {"pos","neg"}}."""
    from concurrent.futures import ThreadPoolExecutor

    import label as lab
    from atlas import Atlas
    at = Atlas(model_name)
    key = at.write_dossiers(list(units), out_dir)
    with ThreadPoolExecutor(max(1, len(key))) as ex:
        res = dict(zip(key, ex.map(lambda c: lab.label_unit(Path(out_dir) / f"{c}.md", use_jev=use_jev), key)))
    return {k["unit"]: {"pos": (res[c]["luna"].get("pos") or {}), "neg": (res[c]["luna"].get("neg") or {})}
            for c, k in key.items()}


def keyword_hit(text: str, keywords) -> list[str]:
    """Which of the marker's job keywords (lower-case stems) the label text contains."""
    t = (text or "").lower()
    return [k for k in keywords if re.search(r"\b" + re.escape(k.lower()), t)]


def uid(layer, neuron) -> str:
    return f"L{layer}_N{neuron}"
