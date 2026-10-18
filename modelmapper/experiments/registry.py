"""Every study in modelmapper/experiments, with its stages. `modelmapper experiment list` prints this table.

`modelmapper experiment <name> [<stage>] [args ...]` runs the module exactly as `python src/.../<name>.py [<stage>]
[args ...]` did: same argv, same environment, same outputs. A study with stages takes one of them as its first
argument (the old `sys.argv[1]` switch); the CLI checks it is one of these before anything runs. Nothing here imports
the modules (many of them do their work at import), so listing and --help stay instant.

Fields: module, group, stages (None = the script takes its own arguments, see its docstring), and env (environment
variables the study reads, settable with --env NAME=VALUE).
"""
from __future__ import annotations

G_LOOP = "loop v2/v3 studies (17-18 Oct): facts, numbers, eras, lenses, graphs"
G_ATLAS = "concept search and the concept atlas (17 Oct)"
G_RECALL = "recall and log-odds (REPORT v1)"
G_STEER = "steering (REPORT v1)"
G_PEOPLE = "people (REPORT v1)"
G_CR3 = "CR-3 pre-registered replication (REPORT v1)"
G_LABELS = "labels and graders (REPORT v1)"
G_GG = "Golden Gate experiment (REPORT v1)"
G_SUM = "summaries (REPORT v1)"

_RV = ("RECALL_VARIANT", "CONCEPT_OUT_DIR")


def _e(group, stages=None, env=()):
    return {"group": group, "stages": stages, "env": tuple(env)}


EXPERIMENTS = {
    # loop v2 / v3 studies (were src/loop_v2/<name>.py)
    "facts": _e(G_LOOP, ["entities", "build", "map", "grid", "position", "summary"]),
    "facts2": _e(G_LOOP, ["entities", "build", "entity", "map", "grid", "position", "summary"]),
    "chain": _e(G_LOOP, env=["CHAIN_K"]),
    "entity_sharing": _e(G_LOOP),
    "numeric": _e(G_LOOP, ["entities", "build", "map", "share", "magnitude", "position", "summary"]),
    "numeric_mag": _e(G_LOOP, ["uncertain", "dose", "summary"]),
    "era": _e(G_LOOP, ["entities", "build", "acts", "text", "readout", "causal", "summary"]),
    "jlens": _e(G_LOOP, ["estimate", "merge", "validate", "neurons", "tests"], env=["JL_ROWS", "JL_TOKENS", "JL_MAX_DIMS"]),
    "scales_steer": _e(G_LOOP),
    "attrib_graph": _e(G_LOOP, ["graphs", "ablate"], env=["AG_ROWS"]),
    "subject_pass": _e(G_LOOP, ["questions", "map", "summary"]),
    "ranking_check": _e(G_LOOP, ["check", "disputed"]),
    "shared_neurons": _e(G_LOOP),
    "feeders": _e(G_LOOP),
    "fanout": _e(G_LOOP, env=["FANOUT_TAG"]),
    "fast_readout": _e(G_LOOP, ["answers", "layers", "list"], env=["NB", "MULT"]),
    "neuron_readout": _e(G_LOOP),
    "specificity": _e(G_LOOP, ["neighbours", "questions", "run", "summary"]),
    # concept search / atlas (were src/<name>.py)
    "concept_search_feasibility": _e(G_ATLAS),
    "concept_atlas": _e(G_ATLAS, ["fetch", "build", "propose", "verify", "verify_fast", "verify_gate"], env=["NB", "VF_TAG", "VF_LIMIT"]),
    "fame_summary": _e(G_ATLAS),
    # REPORT v1 studies (were src/<name>.py)
    "logodds_check": _e(G_RECALL, env=_RV),
    "logodds_compare": _e(G_RECALL),
    "logodds_summary": _e(G_RECALL),
    "k50_check": _e(G_RECALL, env=_RV),
    "backup_routes": _e(G_RECALL, env=_RV),
    "answer_discovery": _e(G_RECALL, env=["CONCEPT_OUT_DIR"]),
    "lexical_check": _e(G_RECALL),
    "replicate_compare": _e(G_RECALL),
    "shared_parts": _e(G_RECALL, env=_RV + ("ANSWER_DISCOVERY",)),
    "shared_parts_lexical": _e(G_RECALL),
    "steer_select": _e(G_STEER, env=_RV + ("REPLY_ONLY",)),
    "steer_retest": _e(G_STEER, env=_RV),
    "steer_retest_summary": _e(G_STEER),
    "auto_steer_scale": _e(G_STEER, env=_RV + ("AUTO_DIRS", "AUTO_OUT")),
    "auto_steer_jev_check": _e(G_STEER),
    "single_neuron_steer": _e(G_STEER, env=_RV),
    "label_rule": _e(G_STEER, env=_RV + ("LABEL_MATCH_FILE", "ONLY", "OUT_NAME", "SKIP_R1")),
    "label_match_luna": _e(G_STEER),
    "fluency_proxy": _e(G_STEER),
    "default_heldout_regen": _e(G_STEER, env=_RV),
    "shuffled_judge": _e(G_STEER),
    "people_rules": _e(G_PEOPLE, env=_RV + ("STEER_MODE",)),
    "people_prompts": _e(G_PEOPLE, env=_RV),
    "people_fluency": _e(G_PEOPLE),
    "says_name": _e(G_PEOPLE),
    "cr3_pick": _e(G_CR3),
    "cr3_jev": _e(G_CR3, env=_RV),
    "cr3_score": _e(G_CR3),
    "cr3_people_baseline": _e(G_CR3, env=_RV),
    "grader2": _e(G_LABELS),
    "ctx_compare": _e(G_LABELS),
    "effort_compare": _e(G_LABELS),
    "h_depth": _e(G_LABELS, ["label", "classify"]),
    "spotcheck_build": _e(G_LABELS),
    "spotcheck_score": _e(G_LABELS),
    "gg_search": _e(G_GG),
    "make_units": _e(G_GG),
    "gg_recognition": _e(G_GG),
    "attribute": _e(G_GG),
    "attribution_diag": _e(G_GG),
    "summarise_gg": _e(G_GG),
    "summarise_concepts": _e(G_SUM),
    "wave_summary": _e(G_SUM),
}

for _name, _ent in EXPERIMENTS.items():
    _ent["module"] = f"modelmapper.experiments.{_name}"
