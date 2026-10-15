# Blind labelling protocol (agreed 15 Oct 2026)

## Blinding
- Each unit is a dossier file named only by a random code (e.g. `U381.md`). The dossier holds no layer or neuron
  number, no hint whether it's a candidate, a random neuron or a planted mixture, and no mention of the Golden Gate,
  the essay or our hypotheses.
- The answer key (`results/dossiers_<model>_KEY.json`) is written separately, and labellers are told to read only
  the dossier files they're given.
- Explanations are written by **GPT-6 Luna** (`src/oai.py`), one dossier per call, so no labeller sees other units.
- **Jev** grades: one yes/no per example ("does the explanation fit the marked token?"), and the same explanation
  against 20 random web tokens (specificity). See `src/label.py`.
- Double-negative firings are removed from the positive pole before the labeller sees it (pre-registered).

## What the labeller sees
For each pole, the 20 strongest firings in 10M tokens of web text. Each shows ~200 tokens of context, the firing token
marked `[[like this]]`, other strong tokens marked `«like this»`, and strength relative to the top firing.

## Instructions given to the labeller (verbatim core)

> You're studying units inside a language model. For each unit you get its strongest positive firings and its
> strongest negative firings on web text. Judge each pole separately.
>
> Read the surrounding text, not just the marked token. Units often fire on punctuation or on the word *before*
> something becomes relevant, summarising or anticipating the context. A unit may be better described by a shared
> pattern or distinction than by a list of topics.
>
> For each pole give:
> 1. **Explanation:** one or two sentences, as specific as the evidence allows.
> 2. **Verdict**, one of:
>    - `ONE` — one main job covers the examples (small, obviously related offshoots allowed)
>    - `TWO_RELATED` — two jobs, with a stated sensible connection
>    - `MIXTURE` — two or more jobs with no connection you can name
>    - `NONE` — no hypothesis
> 3. **Coverage:** which example numbers your explanation covers (e.g. "1–14, 16, 18").
> 4. **Misfits:** example numbers that don't fit, and what they are about.
> 5. **If `TWO_RELATED`:** the connection, in one sentence. It must be something you could test: a shared pattern,
>    not "both are things".
>
> Be honest about mess: calling a unit `MIXTURE` or `NONE` is as useful as `ONE`. Don't stretch an explanation to
> cover everything.

## Scoring (from PREREG, as amended 15 Oct)
- **Clean pole:** Luna verdict `ONE`, AND Jev finds ≥ 16/20 of the examples fit, AND the explanation fits ≤ 4/20 random
  web tokens (so it can lose).
- We report the verdict counts per unit type (candidate / random / planted mixture). The planted mixtures tell us how
  often the labeller *misses* known mess. That's the leniency check.

## Decisions
- Four verdict categories as in the handover.
- The essay's reading tips stay in the instruction; the planted mixtures and the random-token check measure leniency.
- Luna writes, Jev grades (James, 15 Oct: keep it fast and cheap; no Claude subagents).
