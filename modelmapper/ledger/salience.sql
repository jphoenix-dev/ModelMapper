-- Salience rules v1 (LEDGER.md "Derived views"). Rebuilt by build.py; edit here to change what the front end sees.
-- strength = effect / noise floor where both exist (knockouts), else the value of a 0-1 score (AUC) or null.

CREATE OR REPLACE TABLE ev_scored AS
SELECT e.*,
       CASE WHEN e.value IS NOT NULL AND e.noise_floor IS NOT NULL AND e.noise_floor > 0 THEN e.value / e.noise_floor END AS strength
FROM evidence e;

-- one row per (unit, concept): what we know, strongest first
CREATE OR REPLACE TABLE unit_concept AS
SELECT unit_id, any_value(unit_key) AS unit_key, any_value(unit_kind) AS unit_kind, any_value(layer) AS layer, concept_id,
       max(value) FILTER (WHERE evidence_type = 'knockout_drop') AS best_drop,
       min(rank) FILTER (WHERE evidence_type = 'knockout_drop') AS best_rank,
       max(strength) AS strength,
       bool_or(passes) AS any_passes,
       count(*) AS n_evidence,
       list_distinct(list(evidence_type)) AS evidence_types,
       list_distinct(list(experiment_id)) AS experiments,
       any_value(value_text) FILTER (WHERE evidence_type = 'role') AS role
FROM ev_scored
WHERE unit_id IS NOT NULL AND concept_id IS NOT NULL AND evidence_type <> 'null_knockout_drop'
GROUP BY unit_id, concept_id;

CREATE OR REPLACE TABLE unit_summary AS
WITH ranked AS (
  SELECT *, row_number() OVER (PARTITION BY unit_id ORDER BY any_passes DESC, strength DESC NULLS LAST, best_drop DESC NULLS LAST) AS r
  FROM unit_concept),
roles AS (
  SELECT unit_id, map(list(role), list(cnt)) AS role_votes FROM (
    SELECT unit_id, role, count(*) AS cnt FROM unit_concept WHERE role IS NOT NULL GROUP BY unit_id, role) GROUP BY unit_id),
toks AS (
  SELECT unit_id, arg_max(value_text, n_) AS readout_tokens FROM (
    SELECT unit_id, value_text, count(*) AS n_ FROM evidence WHERE evidence_type = 'readout_tokens' GROUP BY unit_id, value_text)
  GROUP BY unit_id),
other AS (
  SELECT unit_id, count(*) AS n_evidence, list_distinct(list(evidence_type)) AS evidence_types,
         list_distinct(list(experiment_id)) AS experiments
  FROM evidence WHERE unit_id IS NOT NULL GROUP BY unit_id)
SELECT o.unit_id, any_value(u.unit_key) AS unit_key, any_value(u.unit_kind) AS unit_kind, any_value(u.layer) AS layer,
       count(r.concept_id) FILTER (WHERE r.any_passes) AS n_concepts_passing,
       count(r.concept_id) AS n_concepts,
       list({'concept_id': r.concept_id, 'drop': r.best_drop, 'rank': r.best_rank, 'strength': r.strength, 'role': r.role}
            ORDER BY r.r) FILTER (WHERE r.r <= 10) AS top_concepts,
       any_value(ro.role_votes) AS role_votes, any_value(t.readout_tokens) AS readout_tokens,
       any_value(o.n_evidence) AS n_evidence, any_value(o.evidence_types) AS evidence_types, any_value(o.experiments) AS experiments
FROM other o
LEFT JOIN ranked r ON r.unit_id = o.unit_id
LEFT JOIN units u ON u.unit_id = o.unit_id
LEFT JOIN roles ro ON ro.unit_id = o.unit_id
LEFT JOIN toks t ON t.unit_id = o.unit_id
GROUP BY o.unit_id;

CREATE OR REPLACE TABLE concept_summary AS
WITH ranked AS (
  SELECT *, row_number() OVER (PARTITION BY concept_id, unit_kind ORDER BY any_passes DESC, best_drop DESC NULLS LAST, strength DESC NULLS LAST) AS r
  FROM unit_concept)
SELECT c.concept_id, c.label, c.kind, c.slug, c.qid,
       (SELECT max(value) FROM evidence e WHERE e.concept_id = c.concept_id AND e.evidence_type = 'recall') AS recall,
       list({'unit_key': r.unit_key, 'drop': r.best_drop, 'rank': r.best_rank, 'role': r.role, 'strength': r.strength} ORDER BY r.r)
         FILTER (WHERE r.unit_kind = 'neuron' AND r.r <= 30) AS top_neurons,
       list({'unit_key': r.unit_key, 'drop': r.best_drop, 'rank': r.best_rank} ORDER BY r.r)
         FILTER (WHERE r.unit_kind = 'head' AND r.r <= 10) AS top_heads,
       count(*) FILTER (WHERE r.any_passes) AS n_units_passing
FROM concepts c LEFT JOIN ranked r ON r.concept_id = c.concept_id
GROUP BY c.concept_id, c.label, c.kind, c.slug, c.qid;

CREATE OR REPLACE TABLE layer_summary AS
SELECT model_id, layer, unit_kind,
       count(DISTINCT unit_id) AS n_units_with_evidence,
       count(DISTINCT unit_id) FILTER (WHERE passes) AS n_units_passing,
       count(*) FILTER (WHERE evidence_type = 'knockout_drop' AND rank <= 5) AS top5_slots,
       count(*) FILTER (WHERE evidence_type = 'role' AND value_text = 'say') AS role_say,
       count(*) FILTER (WHERE evidence_type = 'role' AND value_text = 'attribute') AS role_attribute,
       count(*) FILTER (WHERE evidence_type = 'role' AND value_text = 'late-other') AS role_late_other,
       count(*) FILTER (WHERE evidence_type = 'role' AND value_text = 'early') AS role_early,
       list_distinct(list(experiment_id)) AS experiments
FROM evidence WHERE unit_id IS NOT NULL
GROUP BY model_id, layer, unit_kind;

CREATE OR REPLACE TABLE head_summary AS
SELECT unit_id, any_value(unit_key) AS unit_key, any_value(layer) AS layer,
       count(*) FILTER (WHERE evidence_type = 'knockout_drop') AS n_concepts_top50,
       count(*) FILTER (WHERE evidence_type = 'knockout_drop' AND value >= 0.5) AS n_knockout_ge_0_5,
       median(value) FILTER (WHERE evidence_type = 'knockout_drop') AS median_drop,
       list({'concept_id': concept_id, 'drop': value} ORDER BY value DESC) FILTER (WHERE evidence_type = 'knockout_drop') [1:10] AS top_concepts
FROM evidence WHERE unit_kind = 'head'
GROUP BY unit_id;
