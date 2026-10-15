# Caps audit (16 Oct 2026)

Every truncation or limit in the pipeline: what it is, who sees what, and whether it can distort a score.
Trigger: James spotted that I set caps "as a precaution" without agreeing them, and that they can create scoring
issues. Confirmed at once: I judged Jev "lenient" from the first 160 characters I chose to print, while Jev had
judged the whole reply (it was right; see ISSUES #12).

| # | Cap | Value | Where | Who sees it | Risk to scores | Action |
|---|---|---|---|---|---|---|
| 1 | Steered reply length | **60 new tokens** (58–317 chars) | `steer_test.generate`, `pipeline_fast.GEN_TOKENS` | Jev judge, logs | **Understates on-topic** (less room to bring the topic up); **may understate fluency** (replies cut mid-sentence) | Raise to 150–200; judge with "the reply may be cut off at the end" in the fluency instruction; compare |
| 2 | Random-token snippets for the specificity check | **320 chars** of left context | `intruder_test.random_snippets` (reused by `label.py` / `label_batch.py`) | Jev | **Asymmetry:** graded examples get 500–600 chars, random snippets only 320, so explanations get less context to "fit" random text. That **flatters specificity** (random fit too low) | Same length as the graded examples |
| 3 | Coverage-grading context | 600 chars (`label.py`), 500 (`label_batch.py`, `strata_grade.py`) | `clip()` | Jev | Long-context / summary-motif neurons look like misfits, which **pushes "clean" down** (ISSUES #15). The two labellers also differ from each other | One shared value; sweep 500 / 1,500 / 3,000 on a census subsample |
| 4 | Luna's dossier context per example | 900 chars (`make_dossiers`, `atlas.write_dossiers`); **400 chars** (`label_batch`) | dossiers | Luna | The census labeller (400) sees much less than the replication labeller (900); shorter context means vaguer or wrong explanations | One shared value; test 400 vs 1,500 |
| 5 | Luna explanation length | **≤ 35 words** in `label_batch` (vs "1–2 sentences" in `label.py`) | system prompt | Luna, then Jev | Forced brevity can drop the second facet of a neuron, so **coverage falls and "clean" falls** (census) | Remove the cap; compare |
| 6 | Examples shown vs graded | Luna sees 15 (`label_batch`) or 20; Jev grades 10 or 20 | `--examples`, `--grade` | Luna, Jev | Grading 10 makes coverage noisier (one example = 10 points of coverage) | Grade 20 |
| 7 | Document length | **first 1,024 tokens** of every web document | `scan.docs`, atlas, strata | the model being scanned | Neurons needing longer context never show their real top firings (the essay warns about this explicitly) | Test 4,096 on a subsample |
| 8 | Atlas depth | top 40 per pole, 10M tokens | `build_atlas` | everything downstream | Rare concepts under-represented (Golden Gate: 5 mentions in 10M) | Known; enriched scans where needed |
| 9 | Jev blind-verdict context | 400 chars × 20 examples | `label.jev_verdict` | Jev | Unused metric; fine | – |
| 10 | Intruder test context | 320 chars | `intruder_test` | Jev | Same as #2/#3 | Align |
| 11 | Concept pipeline selections | label top 10; steer top 2/4/6 late (≥ L/2); positions and single-ablation for top 20; head looks for top 5 heads × 8 prompts | `concept_pipeline` | everything | Layer cut-off is ISSUES #11; the rest are logging limits | Sweep (planned) |
| 12 | Recall prompts / controls | 20 / 8 per concept; bar P ≥ 0.2 | `PROMPT_SYSTEM`, `MIN_P` | model | Template problem (ISSUES #9/#10) dominates | Fix template first |
| 13 | **My display truncation when reporting to James** | 140–230 chars per reply | ad-hoc prints | James (and me) | **Created a false "lenient judge" alarm**; can hide evidence either way | Show full text by default, or say explicitly when truncating |

**Rule from now on:** no silent caps. Any limit gets a named constant, a comment saying why, an entry here, and
James's agreement if it can touch a score. Graders and labellers see the same context as each other, and controls
(random snippets) match the examples they're compared against.
