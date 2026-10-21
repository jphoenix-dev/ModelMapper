"""The public functions: `from modelmapper import map_concept, map_batch, steer, load_model, lookup, check`.

Importing modelmapper is light: torch and the model load only when a function needs them.
"""
from __future__ import annotations

import json
import subprocess
import time
import traceback
from pathlib import Path

from modelmapper.records import MapResult
from modelmapper.settings import DEFAULT_MODEL, short_name

DEFAULT_OUT = Path.cwd() / "maps"


def check_prompt_format(model, measurer=None):
    """Fail before any labeller call (which costs money) if the model cannot be asked the questions: no chat template
    and MODEL_MAPPER_PROMPT is not 'plain'. Loads only the tokenizer when no measurer is given."""
    from modelmapper.measure import prompt_format
    if measurer is not None:
        return prompt_format(measurer.tok)
    from transformers import AutoTokenizer
    from modelmapper.settings import model_path
    return prompt_format(AutoTokenizer.from_pretrained(model_path(model)))


class LabellerError(RuntimeError):
    """The labeller step failed (no key, the spend cap, an endpoint error or an unusable reply); the message says which."""


def _commit():
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def load_model(model=None):
    """A Measurer (modelmapper.measure) for `model` (default MODEL_MAPPER_MODEL, else Qwen/Qwen3-8B): load once, reuse."""
    from modelmapper.measure import Measurer
    return Measurer(model or DEFAULT_MODEL)


def _labeller_hint(e):
    msg = str(e)
    if "API key" in msg or "HTTP 401" in msg or "HTTP 403" in msg:
        return "Set up the labeller key (SETUP.md, 'Labeller key'): OPENAI_API_KEY in .env, or OAI_KEY_FILE."
    if "budget" in msg.lower():
        return "The labeller spend cap was reached: raise budget (map-concept --budget)."
    if "rewrite items" in msg or "no answer/recall" in msg:
        return "The labeller's reply did not have the expected form; try again, or another labeller model."
    if "HTTP 400" in msg:
        return "The endpoint refused the request; it must support JSON mode (response_format json_object)."
    return "See SETUP.md, Troubleshooting."


def map_concept(concept, model=None, out=None, budget=0.5, spec=None, measurer=None, band=None, log=print) -> MapResult:
    """Map one concept: the labeller writes questions (or pass `spec`, e.g. a saved or pinned question set), the model
    answers, the labeller judges which answers name the concept, then mapping.map_concept measures. Saves
    <out>/<model>/<concept>.json (+ arrays/<concept>.npz, spec_cache/) and returns a MapResult. band: override the
    hardware band for the in / uncertain / out calls (default: the model's calibrated band in guards.BANDS, 0.13 for
    Qwen3-8B; other models' calls are 'uncalibrated'). The value used and its source are recorded."""
    from modelmapper import guards, mapping, questions as Q
    from modelmapper.measure import slug
    model = (measurer.model_name if measurer else None) or model or DEFAULT_MODEL
    folder = Path(out or DEFAULT_OUT) / short_name(model)
    cache = folder / "spec_cache"; timing, notes = {}, []
    check_prompt_format(model, measurer)              # raises PromptFormatError before the labeller is paid for anything
    if spec is None:
        log(f"[1/4] asking: the labeller writes questions about {concept!r}")
        try:
            spec = Q.write_questions(concept, cache, timing, budget)
        except Exception as e:  # noqa: BLE001
            raise LabellerError(f"{type(e).__name__}: {e}\n{_labeller_hint(e)}") from e
    if measurer is None:
        log(f"      loading {model}")
        measurer = load_model(model)
    if "answer_forms_extra" not in spec:      # a spec that already has them (saved, or pinned) needs no labeller here
        spec = Q.model_answers(measurer, spec, concept, cache)
        spec, note = Q.judge_answers(concept, spec, cache, timing, budget)
        if note:
            notes.append(note); log(f"      note: {note}")
    priced = "" if Q.oai.MODEL == "gpt-6-luna" else f", estimated at Luna's prices (your labeller is {Q.oai.MODEL})"
    log(f"[2-4/4] scoring, switching off the top 200 one at a time, and random controls "
        f"(labeller cost ${Q.USAGE.usd():.3f}{priced})")
    rec = mapping.map_concept(measurer, concept, spec, folder / "arrays" / f"{slug(concept)}.npz", _commit())
    rec["v2"]["prepare_timing_s"] = timing
    if notes:
        rec["v2"]["map_one_notes"] = notes
    guards.annotate(rec, band)
    res = MapResult.from_record(rec)
    res.save(folder / f"{slug(concept)}.json")
    return res


def map_batch(concepts, model=None, out=None, budget=0.5, resume=True, band=None, log=print) -> list[MapResult]:
    """map_concept over a list, with one model load. One record per concept in <out>/<model>/, plus records.jsonl (the
    same records, one per line: the format the map is built from). resume: concepts that already have a record are
    read back, not re-run. A concept that fails is recorded as status 'failed' with the error and traceback, and the
    batch goes on; a CUDA out-of-memory error stops the batch (the device state may be bad)."""
    import torch
    from modelmapper.measure import slug
    model = model or DEFAULT_MODEL
    folder = Path(out or DEFAULT_OUT) / short_name(model); folder.mkdir(parents=True, exist_ok=True)
    measurer, results = None, []
    check_prompt_format(model)                        # once, before any concept (and any labeller call)
    for i, c in enumerate(dict.fromkeys(x.strip() for x in concepts if x.strip()), 1):
        path = folder / f"{slug(c)}.json"
        if resume and path.exists():
            r = MapResult.load(path)
            if r.status != "failed":
                log(f"[{i}] {c}: already mapped ({r.status}), kept"); results.append(r); continue
        log(f"[{i}] {c}")
        t0 = time.time()
        try:
            if measurer is None:
                measurer = load_model(model)
            r = map_concept(c, out=out, budget=budget, measurer=measurer, band=band, log=lambda s: log("    " + s))
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001
            rec = {"concept": c, "row": {"status": "failed", "detail": f"{type(e).__name__}: {e}"[:500],
                                         "traceback": traceback.format_exc()[-3000:]},
                   "rank": None, "v2": {"provenance": {"model": model, "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}}
            r = MapResult.from_record(rec); r.save(path)
        with open(folder / "records.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(r.record, ensure_ascii=False) + "\n")
        log(f"    {r.status} ({time.time() - t0:.0f} s)" + (f": {r.detail}" if r.detail else ""))
        results.append(r)
    return results


def steer(units, model=None, prompts=None, max_new=60, judge=None, baseline=True, reply_only=True, measurer=None):
    """Hold neurons at values while the model answers (modelmapper.steering.steer). units: ["L23:2392=x8", ...]."""
    from modelmapper import steering
    return steering.steer(model or (measurer.model_name if measurer else DEFAULT_MODEL), units, prompts, max_new,
                          judge, measurer, baseline, reply_only)


def lookup(query, records=None, refresh=False):
    """What is already known about a neuron ("L31:8930") or a concept ("Eiffel Tower"): from the published map, or
    from your own records folder (records=...). See modelmapper.known."""
    from modelmapper import known as L
    return L.lookup(query, records=records, refresh=refresh)


def check(apis=False, out=None, quick=False):
    """Run the install check (as `modelmapper check`); returns the report as a dict."""
    from modelmapper import selfcheck as C
    return C.run(apis=apis, out=out, quick=quick)
