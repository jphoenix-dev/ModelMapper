# Scripts: old paths, new modules, commands

On 18 Oct 2026 the code moved from flat scripts in `src/` into the `modelmapper` package, with one command to run
everything. RESULTS.md, the ledger's experiment rows and old logs still name scripts by their old path, such as
`python src/loop_v2/era.py build`. The table below gives each old path's new file and the command that runs it now.

**How to rerun an old command.** Find its old path below, and use the command in that row with the **same
arguments**. For example:

| before | now |
|---|---|
| `python src/loop_v2/era.py build` | `modelmapper experiment era build` |
| `python src/interference.py ...` | `modelmapper archive interference ...` |
| `python src/ledger/import_all.py --no-export` | `modelmapper ledger import --no-export` |
| `RECALL_VARIANT=V3 python src/pipeline_fast.py Qwen/Qwen3-8B "coffee"` | `RECALL_VARIANT=V3 modelmapper concept Qwen/Qwen3-8B "coffee"` |

You can also skip the lookup: `modelmapper run src/<old path> <same arguments>` runs any old script as written.
`modelmapper scripts era` searches this table. `modelmapper <command> --help` prints the script's documentation; it
never starts a run.

The commands come from `pip install -e .` in the repo root. Without installing, `python -m modelmapper ...` from the
repo root works the same.

## What changed, and what didn't

- **The maths, thresholds, prompts, seeds and output formats did not change.** Every command hands its script the
  same arguments (`sys.argv`) and the same environment variables as the old `python src/...` command, and writes
  the same files in the same places. REFACTOR_CHECKS.md has the old-versus-new comparisons.
- **core, loop, experiments, dist** (`modelmapper/core`, `loop`, `experiments`, `dist`): moved, and three kinds of
  edit were made. No other lines changed.
  - Imports now name the package: `from config import env` became `from modelmapper.core.config import env`. The
    `sys.path.insert(...)` lines that made the old bare imports work were removed.
  - The repo root comes from `modelmapper.core.settings.REPO_ROOT`, not `Path(__file__).parents[...]`. Its value
    is unchanged.
  - Data paths come from `modelmapper.core.settings` (`DATA_DIR`, `ATLAS_DOCS`, `STATS`), with the same defaults as
    before.
- **One behaviour change, on purpose.** Some scripts had hard-coded `D:/already-clean-data/...` paths that ignored
  `AC_DATA_DIR`. They now follow it like every other script. With `AC_DATA_DIR` unset the paths are the same as
  before. The scripts affected:
  - loop: `p1_check`, `report`, `run1_checks`, `tag_check`;
  - experiments: `chain`, `entity_sharing`, `era`, `facts`, `fanout`, `fast_readout`, `jlens`, `shared_neurons`,
    `specificity`;
  - dist: `make_run`'s `--ag-dir` and `--jl-docs` defaults.
- **Provenance hashes.** `concept_pipeline`'s saved provenance hashes its five source files from
  `modelmapper/core/`, where they live now. The values differ from older results because those files' import lines
  changed.
- **Moved byte-for-byte, no edits at all:**
  - `ledger/core.py`, `ledger/pack.py` and `ledger/importers/`. Every pack id hashes its importer's source and
    `core.py`, so editing them would re-mint every pack. They keep their bare `ledger.*` / `config` imports, which
    `modelmapper.ledger` resolves.
  - The ledger's data files (`vocab.yaml`, `salience.sql`, `contrib/`).
  - All of `archive/`: the 16-17 Oct one-off analyses, kept for the record. Run them with `modelmapper archive
    <name>`. That makes their bare imports work and gives each script its old `src/<name>.py` location, so its
    `ROOT` and its output paths are what they were.
- **Ledger rows keep their old paths.** The ledger's `experiments` table still names scripts by old path, since
  changing the packs would change their ids. `modelmapper ledger export` adds three fields to each exported
  experiment: `script_files`, `command` (the new command for `script`) and `reproduce_command` (`reproduce` with
  every old `python src/...` call replaced). It also adds `scripts.jsonl`, the table below.
- **Left as written.** These are records of what was run, so they keep the old paths:
  - RESULTS.md (with a pointer to this file), REPORT.md, PREREG*.md, ISSUES.md and LOOP_V2.md;
  - the old job queues `scripts/overnight_chain*.sh`, `after_atlas.sh` and `run_replication.sh`;
  - the `reproduce` strings in `explainer/gen/map/sources/`.

## Layout

| folder | what | kind in the table |
|---|---|---|
| `modelmapper/core/` | model loading, neuron reads and steering, labeller (`oai`) and judge (`jev`) clients, the main pipelines (stats, atlas, scan, labels, concept pipeline, judge-free wave, dial), and `settings.py`: every data path and env var | core |
| `modelmapper/loop/` | the mapping loop: `map_loop` (v1/v2 mapping), `prepare`, `map_core`, `enrich` (v3), `frame`, `report`, and the run checks | loop |
| `modelmapper/experiments/` | one module per study; `registry.py` lists each study's stages (`modelmapper experiment list`) | experiments |
| `modelmapper/ledger/` | the evidence ledger (LEDGER.md) | ledger |
| `modelmapper/dist/` | the rented-GPU runner (RUNBOOK.md) | dist |
| `modelmapper/archive/` | the 16-17 Oct one-off analyses (the 79 flat scripts added after the 16 Oct release, less the 6 that other modules import, which moved to core / loop / experiments) | archive |

"(library)" in the command column means other modules import the file; it has nothing to run on its own.

## Every old path

Generated by `modelmapper scripts --markdown`. The **stages** column lists the first-argument choices the command
accepts; the CLI checks a stage is valid before anything runs.

| old path | new file | kind | command | stages |
|---|---|---|---|---|
| `src/answer_discovery.py` | `modelmapper/experiments/answer_discovery.py` | experiments | `modelmapper experiment answer_discovery` |  |
| `src/answer_readout.py` | `modelmapper/archive/answer_readout.py` | archive | `modelmapper archive answer_readout` |  |
| `src/answer_readout_summary.py` | `modelmapper/archive/answer_readout_summary.py` | archive | `modelmapper archive answer_readout_summary` |  |
| `src/atlas.py` | `modelmapper/core/atlas.py` | core | (library: imported by other modules) |  |
| `src/attribute.py` | `modelmapper/experiments/attribute.py` | experiments | `modelmapper experiment attribute` |  |
| `src/attribute_sharing.py` | `modelmapper/archive/attribute_sharing.py` | archive | `modelmapper archive attribute_sharing` |  |
| `src/attribution_diag.py` | `modelmapper/experiments/attribution_diag.py` | experiments | `modelmapper experiment attribution_diag` |  |
| `src/auto_steer_jev_check.py` | `modelmapper/experiments/auto_steer_jev_check.py` | experiments | `modelmapper experiment auto_steer_jev_check` |  |
| `src/auto_steer_scale.py` | `modelmapper/experiments/auto_steer_scale.py` | experiments | `modelmapper experiment auto_steer_scale` |  |
| `src/backup_routes.py` | `modelmapper/experiments/backup_routes.py` | experiments | `modelmapper experiment backup_routes` |  |
| `src/bowtie.py` | `modelmapper/archive/bowtie.py` | archive | `modelmapper archive bowtie` |  |
| `src/bowtie_summary.py` | `modelmapper/archive/bowtie_summary.py` | archive | `modelmapper archive bowtie_summary` |  |
| `src/build_atlas.py` | `modelmapper/core/build_atlas.py` | core | `modelmapper atlas` |  |
| `src/cache.py` | `modelmapper/core/cache.py` | core | (library: imported by other modules) |  |
| `src/calibrate_clean.py` | `modelmapper/core/calibrate_clean.py` | core | `modelmapper core calibrate_clean` |  |
| `src/cap_fix_compare.py` | `modelmapper/archive/cap_fix_compare.py` | archive | `modelmapper archive cap_fix_compare` |  |
| `src/causal_sweep_args.py` | `modelmapper/archive/causal_sweep_args.py` | archive | `modelmapper archive causal_sweep_args` |  |
| `src/causal_sweep_summary.py` | `modelmapper/archive/causal_sweep_summary.py` | archive | `modelmapper archive causal_sweep_summary` |  |
| `src/chain_cut.py` | `modelmapper/archive/chain_cut.py` | archive | `modelmapper archive chain_cut` |  |
| `src/circuit_maps.py` | `modelmapper/archive/circuit_maps.py` | archive | `modelmapper archive circuit_maps` |  |
| `src/concept_atlas.py` | `modelmapper/experiments/concept_atlas.py` | experiments | `modelmapper experiment concept_atlas` | fetch &#124; build &#124; propose &#124; verify &#124; verify_fast &#124; verify_gate |
| `src/concept_pipeline.py` | `modelmapper/core/concept_pipeline.py` | core | `modelmapper core concept_pipeline` |  |
| `src/concept_search_feasibility.py` | `modelmapper/experiments/concept_search_feasibility.py` | experiments | `modelmapper experiment concept_search_feasibility` |  |
| `src/concept_search_size_check.py` | `modelmapper/archive/concept_search_size_check.py` | archive | `modelmapper archive concept_search_size_check` |  |
| `src/config.py` | `modelmapper/core/config.py` | core | (library: imported by other modules) |  |
| `src/corpus_profile.py` | `modelmapper/archive/corpus_profile.py` | archive | `modelmapper archive corpus_profile` |  |
| `src/corpus_profile_summary.py` | `modelmapper/archive/corpus_profile_summary.py` | archive | `modelmapper archive corpus_profile_summary` |  |
| `src/cr3_jev.py` | `modelmapper/experiments/cr3_jev.py` | experiments | `modelmapper experiment cr3_jev` |  |
| `src/cr3_people_baseline.py` | `modelmapper/experiments/cr3_people_baseline.py` | experiments | `modelmapper experiment cr3_people_baseline` |  |
| `src/cr3_pick.py` | `modelmapper/experiments/cr3_pick.py` | experiments | `modelmapper experiment cr3_pick` |  |
| `src/cr3_score.py` | `modelmapper/experiments/cr3_score.py` | experiments | `modelmapper experiment cr3_score` |  |
| `src/cross_activation.py` | `modelmapper/archive/cross_activation.py` | archive | `modelmapper archive cross_activation` |  |
| `src/cross_activation_summary.py` | `modelmapper/archive/cross_activation_summary.py` | archive | `modelmapper archive cross_activation_summary` |  |
| `src/cross_position.py` | `modelmapper/archive/cross_position.py` | archive | `modelmapper archive cross_position` |  |
| `src/cross_position_summary.py` | `modelmapper/archive/cross_position_summary.py` | archive | `modelmapper archive cross_position_summary` |  |
| `src/ctx_compare.py` | `modelmapper/experiments/ctx_compare.py` | experiments | `modelmapper experiment ctx_compare` |  |
| `src/dedication_curve.py` | `modelmapper/archive/dedication_curve.py` | archive | `modelmapper archive dedication_curve` |  |
| `src/default_heldout_regen.py` | `modelmapper/experiments/default_heldout_regen.py` | experiments | `modelmapper experiment default_heldout_regen` |  |
| `src/dial.py` | `modelmapper/core/dial.py` | core | `modelmapper dial` |  |
| `src/dist/__init__.py` | `modelmapper/dist/__init__.py` | dist | (library: imported by other modules) |  |
| `src/dist/attrib_task.py` | `modelmapper/dist/attrib_task.py` | dist | (library: imported by other modules) |  |
| `src/dist/compare_runs.py` | `modelmapper/dist/compare_runs.py` | dist | `modelmapper dist compare` |  |
| `src/dist/jlens_task.py` | `modelmapper/dist/jlens_task.py` | dist | (library: imported by other modules) |  |
| `src/dist/launcher.py` | `modelmapper/dist/launcher.py` | dist | `modelmapper dist launch` | run &#124; status &#124; terminate &#124; billing |
| `src/dist/links.py` | `modelmapper/dist/links.py` | dist | (library: imported by other modules) |  |
| `src/dist/make_run.py` | `modelmapper/dist/make_run.py` | dist | `modelmapper dist make-run` |  |
| `src/dist/manifest.py` | `modelmapper/dist/manifest.py` | dist | (library: imported by other modules) |  |
| `src/dist/noise_floor.py` | `modelmapper/dist/noise_floor.py` | dist | `modelmapper dist noise-floor` |  |
| `src/dist/quota.py` | `modelmapper/dist/quota.py` | dist | (library: imported by other modules) |  |
| `src/dist/runpod_api.py` | `modelmapper/dist/runpod_api.py` | dist | (library: imported by other modules) |  |
| `src/dist/runtime.py` | `modelmapper/dist/runtime.py` | dist | (library: imported by other modules) |  |
| `src/dist/storage.py` | `modelmapper/dist/storage.py` | dist | (library: imported by other modules) |  |
| `src/dist/tasks.py` | `modelmapper/dist/tasks.py` | dist | (library: imported by other modules) |  |
| `src/dist/timeline.py` | `modelmapper/dist/timeline.py` | dist | `modelmapper dist timeline` |  |
| `src/dist/window_bench.py` | `modelmapper/dist/window_bench.py` | dist | `modelmapper dist window-bench` |  |
| `src/dist/worker.py` | `modelmapper/dist/worker.py` | dist | `modelmapper dist worker` |  |
| `src/effort_compare.py` | `modelmapper/experiments/effort_compare.py` | experiments | `modelmapper experiment effort_compare` |  |
| `src/fame_coding.py` | `modelmapper/archive/fame_coding.py` | archive | `modelmapper archive fame_coding` |  |
| `src/fame_summary.py` | `modelmapper/experiments/fame_summary.py` | experiments | `modelmapper experiment fame_summary` |  |
| `src/families.py` | `modelmapper/archive/families.py` | archive | `modelmapper archive families` |  |
| `src/feeders.py` | `modelmapper/archive/feeders.py` | archive | `modelmapper archive feeders` |  |
| `src/feeders_general.py` | `modelmapper/archive/feeders_general.py` | archive | `modelmapper archive feeders_general` |  |
| `src/feeders_summary.py` | `modelmapper/archive/feeders_summary.py` | archive | `modelmapper archive feeders_summary` |  |
| `src/fluency_proxy.py` | `modelmapper/experiments/fluency_proxy.py` | experiments | `modelmapper experiment fluency_proxy` |  |
| `src/gain_trio.py` | `modelmapper/archive/gain_trio.py` | archive | `modelmapper archive gain_trio` |  |
| `src/gg_recognition.py` | `modelmapper/experiments/gg_recognition.py` | experiments | `modelmapper experiment gg_recognition` |  |
| `src/gg_search.py` | `modelmapper/experiments/gg_search.py` | experiments | `modelmapper experiment gg_search` |  |
| `src/grader2.py` | `modelmapper/experiments/grader2.py` | experiments | `modelmapper experiment grader2` |  |
| `src/grader_validation.py` | `modelmapper/core/grader_validation.py` | core | `modelmapper core grader_validation` |  |
| `src/h_depth.py` | `modelmapper/experiments/h_depth.py` | experiments | `modelmapper experiment h_depth` | label &#124; classify |
| `src/head_arms.py` | `modelmapper/archive/head_arms.py` | archive | `modelmapper archive head_arms` |  |
| `src/head_logodds.py` | `modelmapper/archive/head_logodds.py` | archive | `modelmapper archive head_logodds` |  |
| `src/head_looks2.py` | `modelmapper/archive/head_looks2.py` | archive | `modelmapper archive head_looks2` |  |
| `src/head_looks_summary.py` | `modelmapper/archive/head_looks_summary.py` | archive | `modelmapper archive head_looks_summary` |  |
| `src/head_summary.py` | `modelmapper/archive/head_summary.py` | archive | `modelmapper archive head_summary` |  |
| `src/heads_to_neurons.py` | `modelmapper/archive/heads_to_neurons.py` | archive | `modelmapper archive heads_to_neurons` |  |
| `src/indirect_path.py` | `modelmapper/archive/indirect_path.py` | archive | `modelmapper archive indirect_path` |  |
| `src/indirect_path_summary.py` | `modelmapper/archive/indirect_path_summary.py` | archive | `modelmapper archive indirect_path_summary` |  |
| `src/interference.py` | `modelmapper/archive/interference.py` | archive | `modelmapper archive interference` |  |
| `src/interference_summary.py` | `modelmapper/archive/interference_summary.py` | archive | `modelmapper archive interference_summary` |  |
| `src/intruder_test.py` | `modelmapper/core/intruder_test.py` | core | `modelmapper core intruder_test` |  |
| `src/jev.py` | `modelmapper/core/jev.py` | core | (library: imported by other modules) |  |
| `src/jev_test.py` | `modelmapper/core/jev_test.py` | core | (library: imported by other modules) |  |
| `src/job_overlap.py` | `modelmapper/archive/job_overlap.py` | archive | `modelmapper archive job_overlap` |  |
| `src/job_overlap_z.py` | `modelmapper/archive/job_overlap_z.py` | archive | `modelmapper archive job_overlap_z` |  |
| `src/k50_check.py` | `modelmapper/experiments/k50_check.py` | experiments | `modelmapper experiment k50_check` |  |
| `src/label.py` | `modelmapper/core/label.py` | core | `modelmapper core label` |  |
| `src/label_batch.py` | `modelmapper/core/label_batch.py` | core | `modelmapper label` |  |
| `src/label_match_luna.py` | `modelmapper/experiments/label_match_luna.py` | experiments | `modelmapper experiment label_match_luna` |  |
| `src/label_rule.py` | `modelmapper/experiments/label_rule.py` | experiments | `modelmapper experiment label_rule` |  |
| `src/ledger/__init__.py` | `modelmapper/ledger/__init__.py` | ledger | (library: imported by other modules) |  |
| `src/ledger/build.py` | `modelmapper/ledger/build.py` | ledger | `modelmapper ledger build` |  |
| `src/ledger/core.py` | `modelmapper/ledger/core.py` | ledger | (library: imported by other modules) |  |
| `src/ledger/export.py` | `modelmapper/ledger/export.py` | ledger | `modelmapper ledger export` |  |
| `src/ledger/import_all.py` | `modelmapper/ledger/import_all.py` | ledger | `modelmapper ledger import` |  |
| `src/ledger/importers/__init__.py` | `modelmapper/ledger/importers/__init__.py` | ledger | (library: imported by other modules) |  |
| `src/ledger/importers/attrib_graphs.py` | `modelmapper/ledger/importers/attrib_graphs.py` | ledger | `modelmapper ledger importer attrib_graphs` |  |
| `src/ledger/importers/concept_atlas.py` | `modelmapper/ledger/importers/concept_atlas.py` | ledger | `modelmapper ledger importer concept_atlas` |  |
| `src/ledger/importers/concept_search.py` | `modelmapper/ledger/importers/concept_search.py` | ledger | `modelmapper ledger importer concept_search` |  |
| `src/ledger/importers/curated_claims.py` | `modelmapper/ledger/importers/curated_claims.py` | ledger | `modelmapper ledger importer curated_claims` |  |
| `src/ledger/importers/earlier_maps.py` | `modelmapper/ledger/importers/earlier_maps.py` | ledger | `modelmapper ledger importer earlier_maps` |  |
| `src/ledger/importers/era.py` | `modelmapper/ledger/importers/era.py` | ledger | `modelmapper ledger importer era` |  |
| `src/ledger/importers/facts.py` | `modelmapper/ledger/importers/facts.py` | ledger | `modelmapper ledger importer facts` |  |
| `src/ledger/importers/fanout.py` | `modelmapper/ledger/importers/fanout.py` | ledger | `modelmapper ledger importer fanout` |  |
| `src/ledger/importers/forced_readout.py` | `modelmapper/ledger/importers/forced_readout.py` | ledger | `modelmapper ledger importer forced_readout` |  |
| `src/ledger/importers/jlens.py` | `modelmapper/ledger/importers/jlens.py` | ledger | `modelmapper ledger importer jlens` |  |
| `src/ledger/importers/job_names.py` | `modelmapper/ledger/importers/job_names.py` | ledger | `modelmapper ledger importer job_names` |  |
| `src/ledger/importers/labels.py` | `modelmapper/ledger/importers/labels.py` | ledger | `modelmapper ledger importer labels` |  |
| `src/ledger/importers/links.py` | `modelmapper/ledger/importers/links.py` | ledger | `modelmapper ledger importer links` |  |
| `src/ledger/importers/loop_v3.py` | `modelmapper/ledger/importers/loop_v3.py` | ledger | `modelmapper ledger importer loop_v3` |  |
| `src/ledger/importers/numeric.py` | `modelmapper/ledger/importers/numeric.py` | ledger | `modelmapper ledger importer numeric` |  |
| `src/ledger/importers/run1.py` | `modelmapper/ledger/importers/run1.py` | ledger | `modelmapper ledger importer run1` |  |
| `src/ledger/importers/scales.py` | `modelmapper/ledger/importers/scales.py` | ledger | `modelmapper ledger importer scales` |  |
| `src/ledger/importers/search_verify.py` | `modelmapper/ledger/importers/search_verify.py` | ledger | `modelmapper ledger importer search_verify` |  |
| `src/ledger/importers/shared_neurons.py` | `modelmapper/ledger/importers/shared_neurons.py` | ledger | `modelmapper ledger importer shared_neurons` |  |
| `src/ledger/importers/specificity.py` | `modelmapper/ledger/importers/specificity.py` | ledger | `modelmapper ledger importer specificity` |  |
| `src/ledger/importers/web_firings.py` | `modelmapper/ledger/importers/web_firings.py` | ledger | `modelmapper ledger importer web_firings` |  |
| `src/ledger/pack.py` | `modelmapper/ledger/pack.py` | ledger | `modelmapper ledger pack` | validate &#124; from-jsonl |
| `src/ledger/query.py` | `modelmapper/ledger/query.py` | ledger | `modelmapper ledger query` |  |
| `src/ledger/verify.py` | `modelmapper/ledger/verify.py` | ledger | `modelmapper ledger verify` |  |
| `src/lexical_check.py` | `modelmapper/experiments/lexical_check.py` | experiments | `modelmapper experiment lexical_check` |  |
| `src/logodds_cap_audit.py` | `modelmapper/archive/logodds_cap_audit.py` | archive | `modelmapper archive logodds_cap_audit` |  |
| `src/logodds_check.py` | `modelmapper/experiments/logodds_check.py` | experiments | `modelmapper experiment logodds_check` |  |
| `src/logodds_compare.py` | `modelmapper/experiments/logodds_compare.py` | experiments | `modelmapper experiment logodds_compare` |  |
| `src/logodds_exact_check.py` | `modelmapper/archive/logodds_exact_check.py` | archive | `modelmapper archive logodds_exact_check` |  |
| `src/logodds_summary.py` | `modelmapper/experiments/logodds_summary.py` | experiments | `modelmapper experiment logodds_summary` |  |
| `src/loop_v2/__init__.py` | `modelmapper/loop/__init__.py` | loop | (library: imported by other modules) |  |
| `src/loop_v2/attrib_graph.py` | `modelmapper/experiments/attrib_graph.py` | experiments | `modelmapper experiment attrib_graph` | graphs &#124; ablate |
| `src/loop_v2/chain.py` | `modelmapper/experiments/chain.py` | experiments | `modelmapper experiment chain` |  |
| `src/loop_v2/enrich.py` | `modelmapper/loop/enrich.py` | loop | `modelmapper enrich` | sample &#124; questions &#124; run &#124; summary |
| `src/loop_v2/entity_sharing.py` | `modelmapper/experiments/entity_sharing.py` | experiments | `modelmapper experiment entity_sharing` |  |
| `src/loop_v2/era.py` | `modelmapper/experiments/era.py` | experiments | `modelmapper experiment era` | entities &#124; build &#124; acts &#124; text &#124; readout &#124; causal &#124; summary |
| `src/loop_v2/facts.py` | `modelmapper/experiments/facts.py` | experiments | `modelmapper experiment facts` | entities &#124; build &#124; map &#124; grid &#124; position &#124; summary |
| `src/loop_v2/facts2.py` | `modelmapper/experiments/facts2.py` | experiments | `modelmapper experiment facts2` | entities &#124; build &#124; entity &#124; map &#124; grid &#124; position &#124; summary |
| `src/loop_v2/fanout.py` | `modelmapper/experiments/fanout.py` | experiments | `modelmapper experiment fanout` |  |
| `src/loop_v2/fast_readout.py` | `modelmapper/experiments/fast_readout.py` | experiments | `modelmapper experiment fast_readout` | answers &#124; layers &#124; list |
| `src/loop_v2/feeders.py` | `modelmapper/experiments/feeders.py` | experiments | `modelmapper experiment feeders` |  |
| `src/loop_v2/frame.py` | `modelmapper/loop/frame.py` | loop | `modelmapper frame` | types &#124; vital &#124; draw &#124; tag |
| `src/loop_v2/gate_repro.py` | `modelmapper/loop/gate_repro.py` | loop | `modelmapper loop gate_repro` |  |
| `src/loop_v2/jlens.py` | `modelmapper/experiments/jlens.py` | experiments | `modelmapper experiment jlens` | estimate &#124; merge &#124; validate &#124; neurons &#124; tests |
| `src/loop_v2/map_core.py` | `modelmapper/loop/map_core.py` | loop | (library: imported by other modules) |  |
| `src/loop_v2/neuron_readout.py` | `modelmapper/experiments/neuron_readout.py` | experiments | `modelmapper experiment neuron_readout` |  |
| `src/loop_v2/numeric.py` | `modelmapper/experiments/numeric.py` | experiments | `modelmapper experiment numeric` | entities &#124; build &#124; map &#124; share &#124; magnitude &#124; position &#124; summary |
| `src/loop_v2/numeric_mag.py` | `modelmapper/experiments/numeric_mag.py` | experiments | `modelmapper experiment numeric_mag` | uncertain &#124; dose &#124; summary |
| `src/loop_v2/p1_check.py` | `modelmapper/loop/p1_check.py` | loop | `modelmapper loop p1_check` |  |
| `src/loop_v2/prepare.py` | `modelmapper/loop/prepare.py` | loop | `modelmapper prepare` |  |
| `src/loop_v2/ranking_check.py` | `modelmapper/experiments/ranking_check.py` | experiments | `modelmapper experiment ranking_check` | check &#124; disputed |
| `src/loop_v2/report.py` | `modelmapper/loop/report.py` | loop | `modelmapper report` |  |
| `src/loop_v2/run1_checks.py` | `modelmapper/loop/run1_checks.py` | loop | `modelmapper loop run1_checks` |  |
| `src/loop_v2/scales_steer.py` | `modelmapper/experiments/scales_steer.py` | experiments | `modelmapper experiment scales_steer` |  |
| `src/loop_v2/shared_neurons.py` | `modelmapper/experiments/shared_neurons.py` | experiments | `modelmapper experiment shared_neurons` |  |
| `src/loop_v2/specificity.py` | `modelmapper/experiments/specificity.py` | experiments | `modelmapper experiment specificity` | neighbours &#124; questions &#124; run &#124; summary |
| `src/loop_v2/subject_pass.py` | `modelmapper/experiments/subject_pass.py` | experiments | `modelmapper experiment subject_pass` | questions &#124; map &#124; summary |
| `src/loop_v2/tag_check.py` | `modelmapper/loop/tag_check.py` | loop | `modelmapper loop tag_check` |  |
| `src/luna_firing_judge.py` | `modelmapper/archive/luna_firing_judge.py` | archive | `modelmapper archive luna_firing_judge` |  |
| `src/luna_judge_extras.py` | `modelmapper/archive/luna_judge_extras.py` | archive | `modelmapper archive luna_judge_extras` |  |
| `src/luna_judge_summary.py` | `modelmapper/archive/luna_judge_summary.py` | archive | `modelmapper archive luna_judge_summary` |  |
| `src/make_dossiers.py` | `modelmapper/core/make_dossiers.py` | core | `modelmapper core make_dossiers` |  |
| `src/make_units.py` | `modelmapper/experiments/make_units.py` | experiments | `modelmapper experiment make_units` |  |
| `src/map_loop.py` | `modelmapper/loop/map_loop.py` | loop | `modelmapper map` |  |
| `src/map_loop_check.py` | `modelmapper/archive/map_loop_check.py` | archive | `modelmapper archive map_loop_check` |  |
| `src/map_loop_check_batched.py` | `modelmapper/archive/map_loop_check_batched.py` | archive | `modelmapper archive map_loop_check_batched` |  |
| `src/markers.py` | `modelmapper/core/markers.py` | core | (library: imported by other modules) |  |
| `src/meaning_jobs.py` | `modelmapper/archive/meaning_jobs.py` | archive | `modelmapper archive meaning_jobs` |  |
| `src/meaning_jobs_above_noise.py` | `modelmapper/archive/meaning_jobs_above_noise.py` | archive | `modelmapper archive meaning_jobs_above_noise` |  |
| `src/meaning_jobs_luna_groups.py` | `modelmapper/archive/meaning_jobs_luna_groups.py` | archive | `modelmapper archive meaning_jobs_luna_groups` |  |
| `src/meaning_jobs_summary.py` | `modelmapper/archive/meaning_jobs_summary.py` | archive | `modelmapper archive meaning_jobs_summary` |  |
| `src/meaning_judge.py` | `modelmapper/archive/meaning_judge.py` | archive | `modelmapper archive meaning_judge` |  |
| `src/meaning_judge_summary.py` | `modelmapper/archive/meaning_judge_summary.py` | archive | `modelmapper archive meaning_judge_summary` |  |
| `src/meaning_space.py` | `modelmapper/archive/meaning_space.py` | archive | `modelmapper archive meaning_space` |  |
| `src/meaning_space_summary.py` | `modelmapper/archive/meaning_space_summary.py` | archive | `modelmapper archive meaning_space_summary` |  |
| `src/merge_scans.py` | `modelmapper/core/merge_scans.py` | core | `modelmapper core merge_scans` |  |
| `src/neuron_stats.py` | `modelmapper/core/neuron_stats.py` | core | `modelmapper stats` |  |
| `src/neurons.py` | `modelmapper/core/neurons.py` | core | (library: imported by other modules) |  |
| `src/norm_heads.py` | `modelmapper/archive/norm_heads.py` | archive | `modelmapper archive norm_heads` |  |
| `src/oai.py` | `modelmapper/core/oai.py` | core | (library: imported by other modules) |  |
| `src/p1_p3_run.py` | `modelmapper/archive/p1_p3_run.py` | archive | `modelmapper archive p1_p3_run` |  |
| `src/p1_prompts.py` | `modelmapper/archive/p1_prompts.py` | archive | `modelmapper archive p1_prompts` |  |
| `src/p2_check.py` | `modelmapper/archive/p2_check.py` | archive | `modelmapper archive p2_check` |  |
| `src/p4_checks.py` | `modelmapper/archive/p4_checks.py` | archive | `modelmapper archive p4_checks` |  |
| `src/p4_family.py` | `modelmapper/archive/p4_family.py` | archive | `modelmapper archive p4_family` |  |
| `src/p5_score.py` | `modelmapper/archive/p5_score.py` | archive | `modelmapper archive p5_score` |  |
| `src/people_fluency.py` | `modelmapper/experiments/people_fluency.py` | experiments | `modelmapper experiment people_fluency` |  |
| `src/people_prompts.py` | `modelmapper/experiments/people_prompts.py` | experiments | `modelmapper experiment people_prompts` |  |
| `src/people_rules.py` | `modelmapper/experiments/people_rules.py` | experiments | `modelmapper experiment people_rules` |  |
| `src/pipeline_fast.py` | `modelmapper/core/pipeline_fast.py` | core | `modelmapper concept` |  |
| `src/rank_save.py` | `modelmapper/archive/rank_save.py` | archive | `modelmapper archive rank_save` |  |
| `src/recall_variants.py` | `modelmapper/core/recall_variants.py` | core | `modelmapper core recall_variants` |  |
| `src/regrade.py` | `modelmapper/core/regrade.py` | core | `modelmapper regrade` |  |
| `src/replicate_compare.py` | `modelmapper/experiments/replicate_compare.py` | experiments | `modelmapper experiment replicate_compare` |  |
| `src/route_cut.py` | `modelmapper/archive/route_cut.py` | archive | `modelmapper archive route_cut` |  |
| `src/route_sources.py` | `modelmapper/archive/route_sources.py` | archive | `modelmapper archive route_sources` |  |
| `src/same_layer.py` | `modelmapper/archive/same_layer.py` | archive | `modelmapper archive same_layer` |  |
| `src/says_name.py` | `modelmapper/experiments/says_name.py` | experiments | `modelmapper experiment says_name` |  |
| `src/scan.py` | `modelmapper/core/scan.py` | core | `modelmapper scan` |  |
| `src/search_verify.py` | `modelmapper/archive/search_verify.py` | archive | `modelmapper archive search_verify` |  |
| `src/shared_parts.py` | `modelmapper/experiments/shared_parts.py` | experiments | `modelmapper experiment shared_parts` |  |
| `src/shared_parts_lexical.py` | `modelmapper/experiments/shared_parts_lexical.py` | experiments | `modelmapper experiment shared_parts_lexical` |  |
| `src/shuffled_judge.py` | `modelmapper/experiments/shuffled_judge.py` | experiments | `modelmapper experiment shuffled_judge` |  |
| `src/single_neuron_steer.py` | `modelmapper/experiments/single_neuron_steer.py` | experiments | `modelmapper experiment single_neuron_steer` |  |
| `src/smoke_test.py` | `modelmapper/core/smoke_test.py` | core | `modelmapper core smoke_test` |  |
| `src/spliced_control.py` | `modelmapper/core/spliced_control.py` | core | `modelmapper core spliced_control` |  |
| `src/spotcheck_build.py` | `modelmapper/experiments/spotcheck_build.py` | experiments | `modelmapper experiment spotcheck_build` |  |
| `src/spotcheck_score.py` | `modelmapper/experiments/spotcheck_score.py` | experiments | `modelmapper experiment spotcheck_score` |  |
| `src/staging.py` | `modelmapper/archive/staging.py` | archive | `modelmapper archive staging` |  |
| `src/staging_summary.py` | `modelmapper/archive/staging_summary.py` | archive | `modelmapper archive staging_summary` |  |
| `src/steer_retest.py` | `modelmapper/experiments/steer_retest.py` | experiments | `modelmapper experiment steer_retest` |  |
| `src/steer_retest_summary.py` | `modelmapper/experiments/steer_retest_summary.py` | experiments | `modelmapper experiment steer_retest_summary` |  |
| `src/steer_select.py` | `modelmapper/experiments/steer_select.py` | experiments | `modelmapper experiment steer_select` |  |
| `src/steer_test.py` | `modelmapper/core/steer_test.py` | core | `modelmapper steer` |  |
| `src/strata_grade.py` | `modelmapper/core/strata_grade.py` | core | `modelmapper core strata_grade` |  |
| `src/strata_scan.py` | `modelmapper/core/strata_scan.py` | core | `modelmapper core strata_scan` |  |
| `src/strict_judge.py` | `modelmapper/core/strict_judge.py` | core | `modelmapper core strict_judge` |  |
| `src/summarise_concepts.py` | `modelmapper/experiments/summarise_concepts.py` | experiments | `modelmapper experiment summarise_concepts` |  |
| `src/summarise_gg.py` | `modelmapper/experiments/summarise_gg.py` | experiments | `modelmapper experiment summarise_gg` |  |
| `src/team_sources_summary.py` | `modelmapper/archive/team_sources_summary.py` | archive | `modelmapper archive team_sources_summary` |  |
| `src/ur_demo.py` | `modelmapper/archive/ur_demo.py` | archive | `modelmapper archive ur_demo` |  |
| `src/wave_budget.py` | `modelmapper/archive/wave_budget.py` | archive | `modelmapper archive wave_budget` |  |
| `src/wave_free.py` | `modelmapper/core/wave_free.py` | core | `modelmapper wave` |  |
| `src/wave_summary.py` | `modelmapper/experiments/wave_summary.py` | experiments | `modelmapper experiment wave_summary` |  |
| `src/wiki_extra.py` | `modelmapper/archive/wiki_extra.py` | archive | `modelmapper archive wiki_extra` |  |
| `src/wiki_families.py` | `modelmapper/archive/wiki_families.py` | archive | `modelmapper archive wiki_families` |  |
| `src/wiki_sample.py` | `modelmapper/core/wiki_sample.py` | core | `modelmapper core wiki_sample` |  |
| `src/wiki_sitelinks.py` | `modelmapper/core/wiki_sitelinks.py` | core | `modelmapper core wiki_sitelinks` |  |
| `src/writer_lens.py` | `modelmapper/archive/writer_lens.py` | archive | `modelmapper archive writer_lens` |  |
| `src/writer_sources.py` | `modelmapper/archive/writer_sources.py` | archive | `modelmapper archive writer_sources` |  |
| `src/writer_specificity.py` | `modelmapper/archive/writer_specificity.py` | archive | `modelmapper archive writer_specificity` |  |

### Data files that moved with the ledger (byte-for-byte)

| old path | new path |
|---|---|
| `src/ledger/vocab.yaml` | `modelmapper/ledger/vocab.yaml` |
| `src/ledger/salience.sql` | `modelmapper/ledger/salience.sql` |
| `src/ledger/contrib/research_claims/` (`pack.json`, `claims.jsonl`, `experiments.jsonl`) | `modelmapper/ledger/contrib/research_claims/` |
