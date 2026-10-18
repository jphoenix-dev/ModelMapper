/* Figure 2 of the neuron map: the concept picked in Figure 1 as a graph, drawn like the circuit diagrams in "Neural
   Networks Are Already Clean": its 30 key neurons in labelled groups, in rows by layer (the answer at the top), with
   lines as thick as the measured links between groups. Hover over or tap a group to see its neurons; click it to take
   it out of the picture. Data: data/cgraph.json (build_concept_graphs.py), plus Figure 1's data via window.NeuronMap. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  if (!$("graph")) return;
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const ROWS = [["top", "The answer"], ["answer", "Layers 20–35 · push the answer's word"], ["late", "Layers 20–35 · other jobs"], ["early", "Layers 0–19"]];
  const KIND = {
    name: "Its name is a convenience label that passed our blind check, not the neuron's true meaning.",
    names: "Neurons with checked names that differ from each other; each name is a convenience label.",
    say: "Read from the weights: when these neurons fire as they do here, they push up the answer's own word.",
    say_early: "Early neurons, yet the answer is among the 10 words their weights push most. Early neurons mostly work through later ones, so this is a hint, not a job.",
    shared: "Shared answer machinery: these neurons matter for answers of every kind, not just this one.",
    era: "A tested job: these neurons pick out things from one century, however the date is asked for.",
    scale: "A tested two-ended scale: firing one way pushes towards one end, the other way towards the opposite end.",
    unclear: "The answer suffers without each of these neurons, so they matter here, but we can't yet say what they do.",
  };
  const hoverable = window.matchMedia && window.matchMedia("(hover: hover)").matches;
  let CG = null, cur = null, G = null, removed = new Set(), focus = -1, pinned = -1;

  // how often Qwen gets it right, or for near-certain answers how often it gets it wrong (shared with Figure 1)
  const state = (lo) => window.NeuronMap.state(lo);
  const transition = (base, drop, what) => window.NeuronMap.transition(base, drop, what);

  function wrap(label, max) {
    const words = String(label).split(/\s+/), lines = [""];
    words.forEach((w) => { const L = lines[lines.length - 1]; if ((L + " " + w).trim().length > max && L) lines.push(w); else lines[lines.length - 1] = (L + " " + w).trim(); });
    if (lines.length > 3) { lines.length = 3; lines[2] = lines[2].replace(/.{0,2}$/, "…"); }
    return lines;
  }

  // the strongest group in the middle of its row, weaker ones further out
  function centred(ids) {
    const s = ids.slice().sort((a, b) => G.groups[b].drop - G.groups[a].drop), out = [];
    s.forEach((g, j) => (j % 2 ? out.unshift(g) : out.push(g)));
    return out;
  }

  function layout(w) {
    const narrow = w < 600, gap = narrow ? 8 : 14, minBw = narrow ? 86 : 118, bh = narrow ? 46 : 50, rowGap = narrow ? 64 : 76;
    const perLine = Math.max(1, Math.floor((w - 8 + gap) / (minBw + gap)));
    const pos = {}, rowsY = {};
    let y = 22;
    ROWS.forEach(([r], ri) => {
      rowsY[r] = y;
      const ids = r === "top" ? ["A"] : centred(G.groups.map((g, j) => (g.row === ri - 1 ? j : -1)).filter((j) => j >= 0));
      if (!ids.length) { y += 34; return; }
      for (let s = 0; s < ids.length; s += perLine) {
        const line = ids.slice(s, s + perLine), bw = Math.min(narrow ? 120 : 168, (w - 16 - gap * (line.length - 1)) / line.length);
        const x0 = (w - (line.length * bw + (line.length - 1) * gap)) / 2;
        line.forEach((id, j) => { pos[id] = { x: x0 + j * (bw + gap), y, w: bw, h: bh }; });
        y += bh + 12;
      }
      y += rowGap - 12;
    });
    return { pos, rowsY, h: y - rowGap + 24, narrow };
  }

  function draw() {
    const svg = $("graph");
    if (!G) { svg.innerHTML = ""; svg.style.height = "0px"; return; }
    const w = svg.parentElement.clientWidth || 880, Lo = layout(w);
    svg.setAttribute("viewBox", `0 0 ${w} ${Lo.h}`); svg.style.height = Lo.h + "px";
    const css = getComputedStyle(document.documentElement), ink = css.getPropertyValue("--ink").trim(), muted = css.getPropertyValue("--muted").trim();
    const rule = css.getPropertyValue("--rule").trim(), wash = css.getPropertyValue("--wash").trim(), acc = css.getPropertyValue("--accent").trim();
    const f = focus >= 0 ? focus : pinned;
    let s = `<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="${muted}"/></marker>` +
      `<marker id="ahA" viewBox="0 0 8 8" refX="7" refY="4" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="${acc}"/></marker></defs>`;
    ROWS.forEach(([r, label]) => { s += `<text x="8" y="${Lo.rowsY[r] - 7}" class="rowlab">${esc(label)}</text>`; });
    // links between groups: the strongest, plus every link of the group in focus
    const live = G.links.filter(([a, b]) => !removed.has(a) && !removed.has(b));
    const maxL = Math.max(1, ...live.map((e) => Math.abs(e[2])));
    const strong = new Set(live.slice().sort((x, y) => Math.abs(y[2]) - Math.abs(x[2])).slice(0, 18));
    live.forEach((e) => {
      const [a, b, wt] = e, mine = f >= 0 && (a === f || b === f);
      if (!strong.has(e) && !mine) return;
      const pa = Lo.pos[a], pb = Lo.pos[b]; if (!pa || !pb) return;
      const ax = pa.x + pa.w / 2, bx = pb.x + pb.w / 2;
      let d;
      if (Math.abs(pa.y - pb.y) < 2) {                      // same line: an arc over the top
        const lift = 26 + Math.abs(ax - bx) * 0.12;
        d = `M${ax},${pa.y} C${ax},${pa.y - lift} ${bx},${pb.y - lift} ${bx},${pb.y}`;
      } else if (pa.y > pb.y) {                             // up the page, to a later group
        const my = (pa.y + pb.y + pb.h) / 2; d = `M${ax},${pa.y} C${ax},${my} ${bx},${my} ${bx},${pb.y + pb.h}`;
      } else {                                              // down the page (an answer-row neuron feeding a later one)
        const my = (pa.y + pa.h + pb.y) / 2; d = `M${ax},${pa.y + pa.h} C${ax},${my} ${bx},${my} ${bx},${pb.y}`;
      }
      const op = f < 0 ? 0.55 : mine ? 0.9 : 0.08, sw = (0.8 + 5 * Math.abs(wt) / maxL).toFixed(2);
      s += `<path d="${d}" fill="none" stroke="${wt >= 0 ? muted : acc}" stroke-opacity="${op}" stroke-width="${sw}"${wt < 0 ? ' stroke-dasharray="5 4"' : ""} marker-end="url(#${wt >= 0 ? "ah" : "ahA"})"/>`;
    });
    // the answer-row groups push the answer itself: lines as thick as the answer's loss without them
    const A = Lo.pos.A, maxD = Math.max(1, ...G.groups.map((g) => g.drop));
    G.groups.forEach((g, j) => {
      if (g.row !== 0 || removed.has(j) || !Lo.pos[j]) return;
      const p = Lo.pos[j], x1 = p.x + p.w / 2, x2 = A.x + A.w / 2, my = (p.y + A.y + A.h) / 2, mine = f === j;
      s += `<path d="M${x1},${p.y} C${x1},${my} ${x2},${my} ${x2},${A.y + A.h}" fill="none" stroke="${ink}" stroke-opacity="${f < 0 ? 0.45 : mine ? 0.85 : 0.08}" stroke-width="${(0.8 + 6 * g.drop / maxD).toFixed(2)}" marker-end="url(#ah)"/>`;
    });
    const nOff = removed.size;
    s += `<g><rect x="${A.x}" y="${A.y}" width="${A.w}" height="${A.h}" rx="3" fill="#fff" stroke="${ink}" stroke-width="1.4"/>` +
      wrap(`“${G.answer}”`, Lo.narrow ? 14 : 20).slice(0, 2).map((ln, j, arr) => `<text x="${A.x + A.w / 2}" y="${A.y + A.h / 2 + 1 + (j - (arr.length - 1) / 2) * 14 - (G.base != null ? 6 : -4)}" text-anchor="middle" class="ans">${esc(ln)}</text>`).join("") +
      (G.base != null ? `<text x="${A.x + A.w / 2}" y="${A.y + A.h - 8}" text-anchor="middle" class="sub">${esc(state(G.base).replace(" time in ", " in ").replace(" of the time", ""))}</text>` : "") + `</g>`;
    // the groups
    G.groups.forEach((g, j) => {
      const p = Lo.pos[j]; if (!p) return;
      const off = removed.has(j), on = f === j;
      s += `<g class="gbox" data-g="${j}" tabindex="0" role="button" aria-label="${esc(g.label)}: ${g.n} neuron${g.n > 1 ? "s" : ""}${off ? ", taken out" : ""}">` +
        `<rect x="${p.x}" y="${p.y}" width="${p.w}" height="${p.h}" rx="3" fill="${off ? "#fff" : g.kind === "unclear" ? "#fff" : wash}" stroke="${on ? ink : off ? rule : "#aab3ba"}" stroke-width="${on ? 1.8 : 1}"${off || g.kind === "unclear" ? ' stroke-dasharray="4 3"' : ""}/>`;
      // how much the answer suffers without them (one at a time, added up): a bar along the bottom
      if (!off) s += `<rect x="${p.x + 1}" y="${p.y + p.h - 4}" width="${Math.max(2, (p.w - 2) * g.drop / maxD).toFixed(1)}" height="3" fill="${acc}" opacity="0.75"/>`;
      wrap(g.label, Math.max(9, Math.floor(p.w / 7.2))).forEach((ln, i, arr) => {
        s += `<text x="${p.x + p.w / 2}" y="${p.y + p.h / 2 + 2 + (i - (arr.length - 1) / 2) * 13}" text-anchor="middle" class="glab${off ? " off" : g.kind === "unclear" ? " muted" : ""}">${esc(ln)}</text>`;
      });
      s += `<text x="${p.x + p.w - 4}" y="${p.y + 11}" text-anchor="end" class="cnt">${g.n}</text></g>`;
    });
    svg.innerHTML = s;
    $("graphReset").hidden = !nOff;
  }

  const plural = (n, a, b) => (n === 1 ? a : b);
  function linkWords(j) {
    const lab = (k) => `<a href="#" class="glink" data-g="${k}">${esc(G.groups[k].label)}</a>`;
    const out = G.links.filter((e) => e[0] === j && !removed.has(e[1])), inn = G.links.filter((e) => e[1] === j && !removed.has(e[0]));
    const fmt = (arr, side) => arr.sort((x, y) => Math.abs(y[2]) - Math.abs(x[2])).map((e) => `${lab(e[side])} <span class="muted">(${e[3]} ${plural(e[3], "link", "links")})</span>`).join(", ");
    const feeds = out.filter((e) => e[2] > 0), holds = out.filter((e) => e[2] < 0);
    const bits = [];
    if (feeds.length) bits.push(`it feeds ${fmt(feeds, 1)}`);
    if (holds.length) bits.push(`it holds back ${fmt(holds, 1)}`);
    const fedBy = inn.filter((e) => e[2] > 0), heldBy = inn.filter((e) => e[2] < 0);
    if (fedBy.length) bits.push(`it is fed by ${fmt(fedBy, 0)}`);
    if (heldBy.length) bits.push(`it is held back by ${fmt(heldBy, 0)}`);
    if (!bits.length) return G.anyLinks ? `<span class="muted">No measured links to the other groups.</span>` : `<span class="muted">Links weren't measured for this concept.</span>`;
    const s = bits.join("; ");
    return s.charAt(0).toUpperCase() + s.slice(1) + ".";
  }

  function panel(j) {
    const box = $("graphInfo");
    if (!G) return;
    if (j < 0) {
      box.innerHTML = `<span class="muted">${hoverable ? "Hover over a group to see its neurons. Click it once to pin it here, again to take it out of the picture, and a third time to put it back." : "Tap a group to see its neurons. Tap it again to take it out of the picture, and a third time to put it back."}</span>`;
      return;
    }
    const g = G.groups[j], NM = window.NeuronMap, off = removed.has(j);
    const ls = g.layers[0] === g.layers[1] ? `layer ${g.layers[0]}` : `layers ${g.layers[0]}–${g.layers[1]}`;
    const top = g.members[0];
    const alone = G.base != null && top.drop > 0 ? ` On a typical question, ${transition(G.base, top.drop / 100, g.n === 1 ? "it" : "its strongest neuron alone")}.` : "";
    box.innerHTML = `<b>${esc(g.label)}</b> <span class="muted">· ${g.n} ${plural(g.n, "neuron", "neurons")} · ${ls}</span><br>` +
      `${esc(KIND[g.kind] || "")}${alone}<br>${linkWords(j)}<br>` +
      `<span class="muted">${g.n === 1 ? "The neuron" : "Its neurons, strongest first"} (open in Figure 1):</span> ` + g.members.slice(0, 12).map((m) => {
        const nm = NM && NM.nameOf(m.id);
        return `<a href="#fig-map" class="nlink" data-n="${esc(m.id)}">${esc(m.id)}</a>${nm && g.kind !== "name" ? ` “${esc(nm)}”` : ""}`;
      }).join(" · ") + (g.n > 12 ? ` <span class="muted">and ${g.n - 12} more</span>` : "") +
      `<br><button type="button" class="chip" id="graphToggle">${off ? "Put it back" : "Take it out of the picture"}</button>` +
      ` <span class="muted">Taking a group out only hides it here; we haven't yet measured what switching all its neurons off at once does to the answer.</span>`;
  }

  function build(c) {
    removed = new Set(); focus = -1; pinned = -1;
    const NM = window.NeuronMap, e = c && CG && CG.c[c[0]];
    if (!c || c[14] || !e || !NM) {
      G = null; draw();
      $("graphQ").innerHTML = `<span class="muted">${c && c[14] ? `This figure is for the concepts in Figure 1's picker; centuries and facts come from different tests.` : `Pick a concept in Figure 1 to see its graph here.`}</span>`;
      $("graphInfo").innerHTML = ""; $("graphReset").hidden = true; return;
    }
    const top = c[4];
    G = { answer: (c[2] || c[1]).trim(), base: c[11], anyLinks: e[1].length > 0, links: e[1],
      groups: e[0].map(([row, kind, label, pos, drop]) => {
        const members = pos.map((p) => ({ k: top[p][0], id: NM.unit(top[p][0]), drop: top[p][1] })).sort((a, b) => b.drop - a.drop);
        const Ls = members.map((m) => +m.id.slice(1).split(":")[0]);
        return { row, kind, label, members, n: members.length, drop, layers: [Math.min(...Ls), Math.max(...Ls)] };
      }) };
    $("graphQ").innerHTML = `<b>${esc(c[1])}</b>: its 30 key neurons in ${G.groups.length} groups${G.anyLinks ? `, with ${e[1].reduce((a, x) => a + x[3], 0).toLocaleString("en-GB")} measured links between the groups` : ""}.`;
    draw(); panel(-1);
  }

  const svg = $("graph");
  const boxOf = (e) => { const b = e.target.closest(".gbox"); return b ? +b.dataset.g : -1; };
  // hover shows a group (unless another is pinned); clicks cycle: pin it, take it out, put it back and unpin
  svg.addEventListener("mouseover", (e) => { if (!hoverable) return; const j = boxOf(e); if (j >= 0 && j !== focus) { focus = j; draw(); if (pinned < 0) panel(j); } });
  svg.addEventListener("mouseleave", () => { if (!hoverable) return; focus = -1; draw(); panel(pinned); });
  function cycle(j) {
    if (pinned !== j) pinned = j;
    else if (!removed.has(j)) removed.add(j);
    else { removed.delete(j); pinned = -1; }
    draw(); panel(pinned >= 0 ? pinned : hoverable ? focus : -1);
  }
  svg.addEventListener("click", (e) => {
    const j = boxOf(e);
    if (j >= 0) cycle(j);
    else if (pinned >= 0) { pinned = -1; draw(); panel(-1); }
  });
  svg.addEventListener("keydown", (e) => { const j = boxOf(e); if (j >= 0 && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); cycle(j); } });
  $("graphInfo").addEventListener("click", (e) => {
    const t = e.target.closest("#graphToggle");
    if (t) { const j = focus >= 0 ? focus : pinned; if (j >= 0) { removed.has(j) ? removed.delete(j) : removed.add(j); pinned = j; draw(); panel(j); } return; }
    const g = e.target.closest(".glink");
    if (g) { e.preventDefault(); pinned = +g.dataset.g; focus = -1; draw(); panel(pinned); return; }
    const a = e.target.closest(".nlink");
    if (a && window.NeuronMap) { e.preventDefault(); window.NeuronMap.show(a.dataset.n); $("fig-map").scrollIntoView({ behavior: "smooth" }); }
  });
  $("graphReset").addEventListener("click", () => { removed = new Set(); draw(); panel(pinned); });
  document.addEventListener("neuronmap:concept", (e) => { cur = e.detail; if (CG) build(cur); });
  let rt = null;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => G && draw(), 80); });

  fetch("data/cgraph.json").then((r) => (r.ok ? r.json() : null)).then((d) => {
    if (!d) { $("fig-graph").hidden = true; return; }
    CG = d; build(cur || (window.NeuronMap && window.NeuronMap.concept()));
  }).catch(() => { $("fig-graph").hidden = true; });
})();
