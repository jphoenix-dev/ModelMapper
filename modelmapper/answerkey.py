"""Answer keys: a few marker neurons per model that a correct install must find again (modelmapper/answer_keys/).

Tier 1 (`tier1`) re-runs each marker's concept on the key's own pinned questions and answer wording, ranks the
model's neurons by attribution, switches the top 20 off one at a time, and checks the marker comes back near the key's
rank with a similar drop in P(answer). It is the measurement the keys were built and calibrated with (16 Oct:
markers.run_concept's pinned path and verify_answer_key.py's tier-1 rule, unchanged), so a pass means this install
measures what ours did. The key file also carries each marker's full question set (`spec`), which `check` uses to run
the whole `map_concept` offline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

KEYS = Path(__file__).resolve().parent / "answer_keys"


def load_key(model="Qwen/Qwen3-0.6B") -> dict:
    p = KEYS / f"{model.split('/')[-1]}.json"
    if not p.exists():
        raise FileNotFoundError(f"no answer key for {model}; keys: {', '.join(x.stem for x in KEYS.glob('*.json'))}")
    return json.loads(p.read_text(encoding="utf-8"))


def uid(layer, neuron) -> str:
    return f"L{layer}_N{neuron}"


def chat_prompt(tok, question: str) -> str:
    """The V3 recall format, exactly as pipeline_fast builds it (through measure.as_prompt since 0.1.1)."""
    from modelmapper.measure import as_prompt
    return as_prompt(tok, question + " Answer with just the name.")


def run_pinned(pipe, concept: str, pinned: dict, top_k: int = 20) -> dict:
    """markers.run_concept with pinned questions: recall -> attribution -> single-neuron knockouts of the top_k."""
    rp = [chat_prompt(pipe.tok, q) for q in pinned["questions"]]
    tid = pipe.tok.encode(pinned["answer_form"])
    row = {"concept": concept, "answer": pinned["answer_form"], "answer_form": pinned["answer_form"],
           "recall": len(rp), "recall_total": len(rp), "kept": list(pinned["questions"])}
    base = sum(pipe.probs(rp, tid)) / len(rp)
    A = pipe.attribute(rp, tid)
    la = A["last_act"]
    neurons = []
    for i, (l, n) in enumerate(A["top"][:top_k]):
        p = sum(pipe.probs(rp, tid, [(l, n, 0.0)])) / len(rp)
        neurons.append({"rank": i + 1, "layer": l, "neuron": n, "drop": round(base - p, 4),
                        "sign": "neg" if la[l, n] < 0 else "pos"})
    del A
    return {**row, "status": "ok", "baseline_p": round(base, 4), "neurons": neurons}


def tier1(pipe, key=None, log=print) -> dict:
    """Every marker of the key on its pinned questions; the key's tier-1 pass rule. Returns the result (as
    verify_answer_key.py wrote it): passed, found, markers, pass_fraction, rows."""
    key = key or load_key(pipe.model_name)
    rules = key["pass_rules"]["tier1"]
    rows = []
    for m in key["markers"]:
        t0 = time.time()
        try:
            r = run_pinned(pipe, m["concept"], m["pinned"])
        except Exception as e:  # noqa: BLE001  (this marker's result; the other markers still run)
            r = {"status": "error", "detail": repr(e)[:300], "recall": 0, "recall_total": 0}
        got = next((n for n in r.get("neurons", []) if (n["layer"], n["neuron"]) == (m["layer"], m["neuron"])), None)
        ref = m["evidence"][0]
        max_rank = rules["rank_factor"] * ref["rank"] + rules["rank_slack"]
        min_drop = rules["drop_factor"] * ref["drop"]
        ok = bool(got and got["rank"] <= max_rank and got["drop"] >= min_drop)
        why = (f"concept {r['status']}: {r['detail']}" if r.get("detail") else
               f"not in the top {len(r['neurons'])}" if not got else
               f"rank {got['rank']} (needs <= {max_rank:g}), drop {got['drop']:.3f} (needs >= {min_drop:.3f})")
        rows.append({"concept": m["concept"], "unit": m["unit"], "ok": ok, "detail": why, "key_rank": ref["rank"],
                     "key_drop": ref["drop"], "rank": got and got["rank"], "drop": got and got["drop"],
                     "top5": [uid(n["layer"], n["neuron"]) for n in r.get("neurons", [])[:5]],
                     "seconds": round(time.time() - t0, 1)})
        log(f"[{'ok' if ok else '--'}] {m['concept']:<24} {m['unit']:<11} {why}")
    need = rules["pass_fraction"]
    n_ok = sum(r["ok"] for r in rows)
    return {"model": key["model"], "tier": 1, "passed": n_ok >= need * len(rows), "found": n_ok, "markers": len(rows),
            "pass_fraction": need, "rows": rows}
