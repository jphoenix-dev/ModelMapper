/* Figure 2 of the neuron map: an attribution graph for one question, drawn like the circuit diagrams in "Neural
   Networks Are Already Clean" and Neuronpedia: labelled groups of Qwen3-8B's own neurons in rows (clues in the
   question at the bottom, the concept's parts in the middle, the answer at the top), with links as thick as the
   attribution between them. Hover or tap a group to see its neurons; click it to remove it and see the answer change
   (measured, against random neurons). Data: data/graphs/index.json and data/graphs/<slug>.json (build_graphs.py). */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  if (!$("graph")) return;
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pc = (p) => (p == null ? "–" : p >= 0.995 ? p.toFixed(3) : p >= 0.1 ? p.toFixed(2) : p.toFixed(3));
  const ROWS = [["top", "Answer"], ["answer", "Writes the answer"], ["concept", "The concept"], ["clues", "Clues in the question"]];
  let IDX = [], at = 0, G = null, removed = -1, focus = -1;

  function wrap(label, max) {
    const words = String(label).split(/\s+/), lines = [""];
    words.forEach((w) => { const L = lines[lines.length - 1]; if ((L + " " + w).trim().length > max && L) lines.push(w); else lines[lines.length - 1] = (L + " " + w).trim(); });
    return lines.slice(0, 3);
  }

  function layout(w) {
    const narrow = w < 600, bw = narrow ? 92 : 124, bh = narrow ? 40 : 46, gap = narrow ? 8 : 14, rowGap = narrow ? 92 : 108, top = 20;
    const pos = {}, rowsY = {};
    ROWS.forEach(([r], j) => { rowsY[r] = top + 24 + j * rowGap; });
    // the answer alone on top; groups on their rows, the faint "other" box at the end of the concept row
    const rowItems = { top: [{ key: "A" }], answer: [], concept: [], clues: [] };
    G.groups.forEach((g) => (rowItems[g.row] || rowItems.concept).push({ key: g.id }));
    if (G.other && G.other.n) rowItems.concept.push({ key: "O" });
    Object.entries(rowItems).forEach(([r, items]) => {
      const total = items.length * bw + (items.length - 1) * gap, x0 = Math.max(8, (w - total) / 2);
      items.forEach((it, j) => { pos[it.key] = { x: x0 + j * (bw + gap), y: rowsY[r], w: bw, h: bh }; });
    });
    const h = rowsY.clues + bh + 24;
    return { pos, rowsY, bw, bh, h, narrow };
  }

  function draw() {
    const svg = $("graph"), w = svg.parentElement.clientWidth || 880, Lo = layout(w);
    svg.setAttribute("viewBox", `0 0 ${w} ${Lo.h}`); svg.style.height = Lo.h + "px";
    const css = getComputedStyle(document.documentElement), ink = css.getPropertyValue("--ink").trim(), muted = css.getPropertyValue("--muted").trim();
    const rule = css.getPropertyValue("--rule").trim(), wash = css.getPropertyValue("--wash").trim(), acc = css.getPropertyValue("--accent").trim(), accw = css.getPropertyValue("--accent-wash").trim();
    let s = "";
    ROWS.forEach(([r, label]) => { s += `<text x="8" y="${Lo.rowsY[r] - 8}" class="rowlab">${esc(label)}</text>`; });
    // links: width by attribution, solid when they push towards the answer, dashed when against
    const maxW = Math.max(1e-9, ...G.edges.map((e) => Math.abs(e[2])));
    G.edges.forEach(([a, b, wt]) => {
      const pa = Lo.pos[a === -2 ? "O" : a], pb = Lo.pos[b === -1 ? "A" : b === -2 ? "O" : b];
      if (!pa || !pb || a === removed || b === removed) return;
      const lit = focus < 0 || a === focus || b === focus;
      const x1 = pa.x + pa.w / 2, y1 = pa.y, x2 = pb.x + pb.w / 2, y2 = pb.y + pb.h;
      const up = y1 > y2, my = (y1 + y2) / 2;
      const d = up ? `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}` : `M${x1},${y1 + pa.h} C${x1},${my + 40} ${x2},${my - 40} ${x2},${pb.y}`;
      s += `<path d="${d}" fill="none" stroke="${wt >= 0 ? muted : acc}" stroke-opacity="${lit ? 0.75 : 0.12}" stroke-width="${(1 + 6 * Math.abs(wt) / maxW).toFixed(2)}"${wt < 0 ? ' stroke-dasharray="5 4"' : ""}/>`;
    });
    // the answer
    const A = Lo.pos.A, pAns = removed >= 0 ? (G.groups[removed].ablation || {}).p : (G.p_base ?? G.p_answer);
    s += `<g><rect x="${A.x}" y="${A.y}" width="${A.w}" height="${A.h}" rx="3" fill="#fff" stroke="${ink}" stroke-width="1.4"/>` +
      `<text x="${A.x + A.w / 2}" y="${A.y + A.h / 2 - 3}" text-anchor="middle" class="ans">${esc(G.answer)}</text>` +
      `<text x="${A.x + A.w / 2}" y="${A.y + A.h / 2 + 13}" text-anchor="middle" class="sub">${pAns != null ? "P = " + pc(pAns) : ""}</text></g>`;
    // groups
    G.groups.forEach((g) => {
      const p = Lo.pos[g.id]; if (!p) return;
      const off = removed === g.id, against = (g.attr || 0) < 0;
      s += `<g class="gbox" data-g="${g.id}" tabindex="0" role="button" aria-label="${esc(g.label)}: ${g.n} neurons">` +
        `<rect x="${p.x}" y="${p.y}" width="${p.w}" height="${p.h}" rx="3" fill="${off ? "#fff" : against ? accw : wash}" stroke="${focus === g.id ? ink : off ? rule : "#aab3ba"}" stroke-width="${focus === g.id ? 1.8 : 1}"${off ? ' stroke-dasharray="4 3"' : ""}/>`;
      wrap(g.label, Lo.narrow ? 13 : 18).forEach((ln, j, arr) => {
        s += `<text x="${p.x + p.w / 2}" y="${p.y + p.h / 2 + 4 + (j - (arr.length - 1) / 2) * 13}" text-anchor="middle" class="glab${off ? " off" : ""}">${esc(ln)}</text>`;
      });
      s += `<text x="${p.x + p.w - 4}" y="${p.y + 11}" text-anchor="end" class="cnt">${g.n}</text></g>`;
    });
    if (Lo.pos.O) { const p = Lo.pos.O; s += `<rect x="${p.x}" y="${p.y}" width="${p.w}" height="${p.h}" rx="3" fill="none" stroke="${rule}" stroke-dasharray="4 3"/><text x="${p.x + p.w / 2}" y="${p.y + p.h / 2 + 4}" text-anchor="middle" class="glab muted">other (${G.other.n})</text>`; }
    svg.innerHTML = s;
  }

  function panel(gid) {
    const box = $("graphInfo");
    if (gid < 0) {
      box.innerHTML = `<span class="muted">Hover over or tap a group to see its neurons; click it to remove them and see what happens to the answer.</span>`;
      return;
    }
    const g = G.groups[gid], a = g.ablation;
    const rem = a && a.p != null ? `<b>Removing these ${g.n} neuron${g.n > 1 ? "s" : ""}:</b> P(${esc(G.answer)}) ${pc(G.p_base ?? G.p_answer)} → <b>${pc(a.p)}</b>` +
      (a.p_random && a.p_random.length ? `; ${g.n} random neurons from the same layers: ${a.p_random.map(pc).join(", ")}.` : ".") : `<span class="muted">Removal not measured yet.</span>`;
    box.innerHTML = `<b>${esc(g.label)}</b> <span class="muted">· ${g.n} neuron${g.n > 1 ? "s" : ""} · ${(g.attr || 0) >= 0 ? "pushes towards" : "pushes against"} the answer</span><br>${rem}<br>` +
      `<span class="muted">Neurons (strongest first):</span> ` + g.nodes.slice(0, 12).map((n) => {
        const k = window.NeuronMap && window.NeuronMap.indexOf(n.neuron);
        const nm = window.NeuronMap && window.NeuronMap.nameOf(n.neuron);
        return `<a href="#fig-map" class="nlink" data-n="${esc(n.neuron)}">${esc(n.neuron)}</a>${nm ? ` “${esc(nm)}”` : ""} <span class="muted">at “${esc((n.token || "").trim())}”</span>`;
      }).join(" · ") + (g.n > 12 ? ` <span class="muted">and ${g.n - 12} more</span>` : "");
  }

  function load(j) {
    at = (j + IDX.length) % IDX.length; removed = -1; focus = -1;
    $("graphPos").textContent = `${at + 1} / ${IDX.length}`;
    $("graphPick").value = String(at);
    fetch(`data/graphs/${IDX[at].slug}.json`).then((r) => r.json()).then((g) => {
      G = g;
      $("graphQ").innerHTML = `<b>${esc(g.concept)}</b>: “${esc(g.question)}” <span class="muted">→ ${esc(g.answer)}</span>`;
      const gate = g.gate || g.gates || {}, rc = gate.knockout_rank_corr ?? gate.spearman_top30, j = g.joint;
      $("graphNote").innerHTML = [
        j && j.p != null ? `Switching off all ${j.n} neurons in this graph at once takes P(${esc(g.answer)}) from ${pc(g.p_base)} to ${pc(j.p)}; the same number of random neurons: ${j.p_random.map(pc).join(", ")}.` : "",
        rc != null ? (rc >= 0.7 ? `Check passed: switching single neurons off changes the answer in line with the graph (rank correlation ${rc.toFixed(2)}).`
          : `<b>Check failed for this question:</b> switching single neurons off doesn't track the graph well (rank correlation ${rc.toFixed(2)}, pass mark 0.7), so read it with care.`) : ""].filter(Boolean).join(" ");
      draw(); panel(-1);
    });
  }

  const svg = $("graph");
  svg.addEventListener("mouseover", (e) => { const b = e.target.closest(".gbox"); if (b && removed < 0) { focus = +b.dataset.g; draw(); panel(focus); } });
  svg.addEventListener("click", (e) => {
    const b = e.target.closest(".gbox"); if (!b) return;
    const gid = +b.dataset.g; removed = removed === gid ? -1 : gid; focus = gid; draw(); panel(gid);
  });
  svg.addEventListener("keydown", (e) => { const b = e.target.closest(".gbox"); if (b && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); b.dispatchEvent(new MouseEvent("click", { bubbles: true })); } });
  $("graphInfo").addEventListener("click", (e) => {
    const a = e.target.closest(".nlink"); if (!a) return;
    if (window.NeuronMap) { e.preventDefault(); window.NeuronMap.show(a.dataset.n); document.getElementById("fig-map").scrollIntoView({ behavior: "smooth" }); }
  });
  $("graphPrev").addEventListener("click", () => load(at - 1));
  $("graphNext").addEventListener("click", () => load(at + 1));
  $("graphPick").addEventListener("change", (e) => load(+e.target.value));
  window.addEventListener("resize", () => G && draw());

  fetch("data/graphs/index.json").then((r) => (r.ok ? r.json() : [])).then((idx) => {
    IDX = idx;
    if (!IDX.length) { $("fig-graph").hidden = true; return; }
    $("graphPick").innerHTML = IDX.map((g, j) => `<option value="${j}">${esc(g.concept)}</option>`).join("");
    load(0);
  }).catch(() => { $("fig-graph").hidden = true; });
})();
