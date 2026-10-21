"""The label runner: chunking, resume (finished steps are skipped), the spend cap and the scorecard at the end. No API
calls and no model: every step's output is written beforehand, as a stopped run would have left them."""
import json

import pytest

from modelmapper import label

NEURONS = ["L0:1", "L0:2", "L1:3"]


def write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def prepared(tmp_path, cost=0.5, jev=0.2):
    """A run whose exams, words and generative checks are all done (chunk=2 -> chunks c0, c1)."""
    recs = {"L0:1": {"bin": "intelligible", "list_pass": True, "auc_own": 0.9, "auc_control": 0.5, "description": "dogs"},
            "L0:2": {"bin": "stable-not-named", "list_pass": True, "auc_own": 0.6, "auc_control": 0.5, "description": "?"},
            "L1:3": {"bin": "intelligible", "list_pass": True, "auc_own": 0.8, "auc_control": 0.4, "description": "cats"}}
    for k, part in enumerate(label.pools(NEURONS, 2)):             # interleaved: [L0:1, L1:3], [L0:2]
        for b in label.BOARDS:
            tag = f"t_c{k}_{b}"
            write(tmp_path / f"intelligibility_{tag}.json", {"_summary": {"board": b, "cost_usd": cost, "jev_usd": jev},
                                                             **{u: recs[u] for u in part}})
            write(tmp_path / f"gen_words_{tag}_surf.json", {"words": {}, "luna_usd": 0.01})
            write(tmp_path / f"gen_check_{tag}_surf.json", {"_summary": {"luna_usd": 0.01},
                                                            **{u: {"pass": u == "L0:1"} for u in part if recs[u]["bin"] == "intelligible"}})
    return tmp_path


def test_resumes_from_finished_steps_and_scores(tmp_path):
    out = prepared(tmp_path)
    logs = []
    res = label.label_run(NEURONS, out, "t", out, "v2", chunk=2, model=object(), tok=object(), log=logs.append)
    assert res["summary"]["triage"] == {"clear": 2, "one-sided": 0, "pattern-not-named": 1, "unexplained": 0}
    assert res["neurons"]["L0:1"]["gate"]["gen_pass"] is True and res["neurons"]["L1:3"]["gate"]["gen_pass"] is False
    assert label.pools(list(range(10)), 4) == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert sum("done, skipped" in m for m in logs) == 6                        # 2 chunks x 3 boards, nothing re-run
    assert res["spend_usd"] == {"labeller": round(6 * 0.5 + 6 * 0.01, 4), "judge": round(6 * 0.2, 4)}
    assert (out / "t_results.json").exists()


def test_spend_cap_stops_before_new_work(tmp_path):
    out = prepared(tmp_path, cost=30.0)
    with pytest.raises(label.SpendCap):
        label.label_run(NEURONS, out, "t", out, "v2", chunk=2, model=object(), tok=object(), luna_cap=45, log=lambda *_: None)
