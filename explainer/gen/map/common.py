"""Shared helpers for the map build: paths, unit ids, slugs, and the Finding record every source emits.

A finding is one claim about one unit (a neuron, a head, a layer range, a concept or the whole model), with the
numbers behind it and a pointer to the source that produced it. Sources (sources/*.py) emit findings and concept
maps; build_map.py merges them into the static JSON the viewer reads. New experiments add a source module or a
curated findings/*.jsonl file; nothing in the viewer has to change.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                   # explainer/gen/map -> repo root


def _dotenv(key):
    p = Path(os.environ.get("MAP_DOTENV", "C:/dev/already-clean/.env"))
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    return None


# Where the inputs live. Every path can be overridden from the environment.
RUN1 = Path(os.environ.get("MAP_RUN1", "D:/already-clean-data/loop_v2/run1"))
FRAME = Path(os.environ.get("MAP_FRAME", "D:/already-clean-data/loop_v2/frame"))
EARLIER = Path(os.environ.get("MAP_EARLIER", "C:/dev/already-clean/results"))   # earlier map-loop / rank_save runs
RDATA = Path(os.environ.get("MAP_RESEARCH_DATA", "D:/already-clean-data"))       # research-branch experiment outputs
RESULTS = REPO / "results"
MODELS = Path(os.environ.get("MODEL_MAPPER_MODELS") or _dotenv("MODEL_MAPPER_MODELS") or "C:/dev/models/hf")
OUT = Path(os.environ.get("MAP_OUT", REPO / "explainer" / "site" / "map" / "data"))
CACHE = HERE / "cache"

N_LAYERS, N_NEURONS, N_HEADS = 36, 12288, 32
REPO_URL = "https://github.com/jphoenix-dev/ModelMapper"

_NID = re.compile(r"^L(\d+)[:_]N?(\d+)$")
_HID = re.compile(r"^L(\d+)H(\d+)$")


def nid(s):
    """'L27:6026' or 'L0_N492' -> 'L27:6026' (canonical)."""
    m = _NID.match(str(s).strip())
    if not m:
        raise ValueError(f"not a neuron id: {s!r}")
    return f"L{int(m.group(1))}:{int(m.group(2))}"


def nlayer(n):
    return int(n[1:].split(":")[0])


def hlayer(h):
    return int(_HID.match(h).group(1))


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower().replace("&", " and ")).strip("-")
    return s[:80] or "concept"


def question(prompt):
    """The user's question from a chat-templated recall prompt."""
    m = re.search(r"<\|im_start\|>user\n(.*?)(?: Answer with just the name\.)?<\|im_end\|>", prompt, re.S)
    return (m.group(1) if m else prompt).strip()


def finding(unit, kind, title, source, *, detail=None, metric=None, status="exploratory", strength=None,
            concept=None, docs=None, fid=None):
    """One claim about one unit. unit: 'L27:6026' | 'L34H28' | 'layers:28-35' | 'concept:<slug>' | 'model'."""
    f = {"unit": unit, "kind": kind, "title": title, "source": source, "status": status}
    for k, v in (("detail", detail), ("metric", metric), ("strength", strength), ("concept", concept),
                 ("docs", docs), ("id", fid)):
        if v is not None:
            f[k] = v
    return f


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


BYTE_TOKEN = "▯"   # shown for a token that decodes to a partial UTF-8 sequence (the tokenizer's U+FFFD)


def clean_text(s):
    return s.replace("�", BYTE_TOKEN)


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(clean_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    return p.stat().st_size
