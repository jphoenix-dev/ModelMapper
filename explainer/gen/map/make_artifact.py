"""Package the map as a self-contained claude.ai artifact folder (for review on a phone, or sharing privately).

    python -I -X utf8 explainer/gen/map/make_artifact.py <out_dir> [--shards 60]

Writes <out_dir>/index.html (the map page with its CSS and JS inlined, no document skeleton: the artifact host adds
one) and <out_dir>/data/ (the same JSON, with concepts packed into shard files so the file count fits the host's
limits). Links to the explainer point at the deployed Eiffel Tower Qwen site.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parents[1] / "site"
EXPLAINER_URL = "https://eiffel-tower-qwen.vercel.app/"


def neurons_page(out):
    """The neuron map (map/index.html) on its own, with only the data it reads."""
    import shutil
    src = SITE / "map"
    html = (src / "index.html").read_text(encoding="utf-8")                 # the neuron map is the map's index page
    css = (SITE / "style.css").read_text(encoding="utf-8")
    own = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    js = (src / "neurons.js").read_text(encoding="utf-8") + "\n" + (src / "graphs.js").read_text(encoding="utf-8")
    title = re.search(r"<title>.*?</title>", html).group(0)
    fonts = re.search(r'<link rel="stylesheet" href="https://fonts\.googleapis\.com[^>]*>', html).group(0)
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1).replace('<script src="neurons.js"></script>', "").replace('<script src="graphs.js"></script>', "")
    body = body.replace('href="../"', f'href="{EXPLAINER_URL}"')
    page = (f'{title}\n<meta name="claude:viewport-fit-sides" content="cover">\n{fonts}\n'
            f"<style>\n{css}\n{own}\n</style>\n{body}\n<script>window.MAP_LINKS = false;</script>\n<script>\n{js}\n</script>\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    graphs = [f"graphs/{p.name}" for p in sorted((src / "data" / "graphs").glob("*.json"))] if (src / "data" / "graphs").exists() else []
    for p in ["neuronmap.json", "concepts.json"] + [f"n/L{L}.json" for L in range(36)] + [f"push/L{L}.json" for L in range(36)] + graphs:
        (out / "data" / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "data" / p, out / "data" / p)
    print(f"{out / 'index.html'}: {len(page) / 1e3:.0f} kB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shards", type=int, default=60)
    ap.add_argument("--page", default="map", choices=["map", "neurons"])
    args = ap.parse_args()
    out = Path(args.out)
    if args.page == "neurons":
        return neurons_page(out)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-I", "-X", "utf8", str(HERE / "build_map.py"), "--shards", str(args.shards),
                    "--out", str(out / "data")], check=True)

    lookup = HERE / "lookup_site"                                            # the lookup pages, not released yet
    html = (lookup / "index.html").read_text(encoding="utf-8")
    css = (SITE / "style.css").read_text(encoding="utf-8") + "\n" + (lookup / "map.css").read_text(encoding="utf-8")
    js = (lookup / "map.js").read_text(encoding="utf-8")
    title = re.search(r"<title>.*?</title>", html).group(0)
    fonts = re.search(r'<link rel="stylesheet" href="https://fonts\.googleapis\.com[^>]*>', html).group(0)
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
    body = body.replace('<script src="map.js"></script>', "")
    body = body.replace('href="../"', f'href="{EXPLAINER_URL}"')
    page = (f'{title}\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n{fonts}\n'
            f"<style>\n{css}\n</style>\n{body}\n"
            f"<script>window.MAP_EXPLAINER = {EXPLAINER_URL!r};</script>\n<script>\n{js}\n</script>\n")
    (out / "index.html").write_text(page, encoding="utf-8")
    files = sorted(p for p in (out / "data").rglob("*") if p.is_file())
    print(f"{out / 'index.html'}: {len(page) / 1e3:.0f} kB; data: {len(files)} files, "
          f"{sum(p.stat().st_size for p in files) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
