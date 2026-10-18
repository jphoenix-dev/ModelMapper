"""Blind check of frame.py's named/common tags: Luna labels a random sample (seed 81) without seeing our tag; report
agreement, and whether the named-vs-common differences in run 1 hold on the concepts where both labels agree.
Usage: python src/loop_v2/tag_check.py <run_dir> [n=300] -> <run_dir>/tag_check.json, tag_check.txt
"""
import glob
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oai  # noqa: E402

run = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
SYS = "You classify encyclopedia topics. Reply in JSON only."


def main():
    F = json.loads(Path("D:/already-clean-data/loop_v2/frame/frame.json").read_text(encoding="utf-8"))["draw"]
    sample = random.Random(81).sample(F, N)

    def ask(x):
        q = (f'Encyclopedia article title: "{x["concept"]}". Is this a NAMED ENTITY (a proper noun naming one specific '
             'individual thing: a particular person, place, organisation, work, event, language, etc.) or a COMMON concept '
             '(a general kind of thing, substance, activity, idea, quality, field - normally written in lowercase in running '
             'text)? Reply {"label": "named" | "common"}')
        for _ in range(3):
            try:
                return x["concept"], oai.chat_json(q, SYS).get("label")
            except Exception:  # noqa: BLE001
                pass
        return x["concept"], None
    with ThreadPoolExecutor(32) as ex:
        luna = dict(ex.map(ask, sample))
    ours = {x["concept"]: x["kind"] for x in sample}
    both = [c for c in luna if luna[c] in ("named", "common")]
    agree = [c for c in both if luna[c] == ours[c]]
    recs = {}
    for b in glob.glob(str(run / "results/**/map/*/records.jsonl"), recursive=True):
        for line in Path(b).read_text(encoding="utf-8").splitlines():
            r = json.loads(line); recs[r["concept"]] = r
    L = [f"tag check: {len(both)} of {N} labelled by Luna; agreement with frame.py {len(agree)}/{len(both)} ({len(agree) / len(both):.0%}); Luna {oai.usage}"]
    for a in ("named", "common"):
        for b in ("named", "common"):
            L.append(f"  ours {a:<6} / Luna {b:<6}: {sum(ours[c] == a and luna[c] == b for c in both)}")
    dis = [f"{c} (ours {ours[c]}, Luna {luna[c]})" for c in both if luna[c] != ours[c]]
    L.append("disagreements: " + "; ".join(dis[:40]))
    for lab in ("named", "common"):
        rs = [recs[c] for c in agree if ours[c] == lab and c in recs and recs[c]["row"]["status"] == "ok"]
        if rs:
            say = sum(any(x["role"] == "say" for x in r["v2"]["roles_top30"][:5]) for r in rs)
            L.append(f"agreed {lab}: mapped {len(rs)}; 'say' neuron in top-5 {say / len(rs):.0%}")
    txt = "\n".join(L)
    (run / "tag_check.json").write_text(json.dumps({"luna": luna, "ours": ours}, indent=1, ensure_ascii=False), encoding="utf-8")
    (run / "tag_check.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
