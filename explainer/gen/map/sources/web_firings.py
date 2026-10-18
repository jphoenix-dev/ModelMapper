"""Luna reads concept neurons' web-text firings: is each firing about the concept, related, or unrelated?"""
from __future__ import annotations

from collections import Counter

from common import RESULTS, finding, read_json, slug

SOURCE = {
    "id": "webfire",
    "title": "Concept neurons in ordinary web text (Luna firing judge)",
    "kind": "input side",
    "status": "exploratory",
    "date": "2026-10-17",
    "method": (
        "For concept neurons found by the maps, 12 of their firings in FineWeb are sampled across the activation range "
        "and shown to GPT-6 Luna with the concept named. Each snippet is judged A (about the concept), B (related: same "
        "topic, a closely related thing or a broader category) or C (unrelated). Random neurons judged against the same "
        "concepts are the control (95% unrelated)."),
    "reproduce": "python src/luna_firing_judge.py   (research/attention branch)",
    "docs": ["RESULTS.md#17-oct-2026-night-luna-reads-the-concept-neurons-web-text-firings"],
    "caveats": ["Concept neurons are mostly polysemantic in the wild (12% dedicated, 22% ingredient, 57% mostly "
                "unrelated) but specific in use: off-topic firings are harmless (interference test)."],
}


def load(ctx):
    p = RESULTS / "luna_firing_judge.json"
    if not p.exists():
        return
    for key, j in read_json(p)["jobs"].items():
        if j.get("control") or not j.get("valid") or not j.get("labels"):
            continue
        n = key.split("|")[0]
        c = Counter(j["labels"]); t = sum(c.values())
        share = {k: round(c.get(k, 0) / t, 2) for k in "ABC"}
        title = (f"{share['A']:.0%} of its web firings are about {j['concept']}, {share['B']:.0%} related, "
                 f"{share['C']:.0%} unrelated")
        ctx.add_finding(finding(
            n, "web firings", title, "webfire", concept=slug(j["concept"]),
            detail=(f"unrelated firings' theme: {j['theme_of_unrelated']}" if j.get("theme_of_unrelated") not in
                    (None, "none") else None),
            metric={"about": share["A"], "related": share["B"], "unrelated": share["C"], "n": t}))
