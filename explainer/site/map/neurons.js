/* Figure 1 of the neuron map: every mapped neuron of Qwen3-8B on a canvas (36 layer rows; layers 20-35 grouped into
   subject columns, earlier layers in their real order), a chosen concept's 30 key neurons lit as a sweep moves down
   the layers, the answer's logit-lens rank per layer, a zoom into one subject column, and a plain description of any
   neuron on hover or tap. Data: data/neuronmap.json (layout, concepts, checked convenience names) and
   data/n/L<layer>.json (each neuron's details, fetched when first needed). */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const css = getComputedStyle(document.documentElement);
  const C = { ink: css.getPropertyValue("--ink").trim() || "#1d2226", muted: css.getPropertyValue("--muted").trim() || "#5d6770",
    rule: css.getPropertyValue("--rule").trim() || "#e2e5e8", wash: css.getPropertyValue("--wash").trim() || "#f4f5f6",
    acc: css.getPropertyValue("--accent").trim() || "#8a5a2e", cool: css.getPropertyValue("--cool").trim() || "#8a9aa5" };
  const LINKS = window.MAP_LINKS === true;        // links to the lookup pages: off until those pages are released
  const ROLE_TEXT = { 1: "It writes the answer: the answer is among the words it pushes hardest.",
    2: "It pushes the answer up, as one of its words.",
    3: "Its direct push doesn't favour the start of the answer, so it matters some other way: later in the answer, or through other neurons.",
    4: "It is an early neuron: it has no direct say in the words, and works through later neurons." };
  const N_LAYERS = 36;
  // one colour per kind of job (order = data/neuronmap.json job_types); all readable on white
  const JOBC = { say: "#8a5a2e", attribute: "#2c7a6f", entity: "#4f5fa8", relation: "#8e44ad", fact: "#c0392b",
    era: "#1f78b4", digit: "#6b8e23", scale: "#b8860b", shared: "#7f8c8d", label: "#34495e", named: "#1d2226" };
  const jobColor = (j) => (j >= 0 && D.job_types[j] ? JOBC[D.job_types[j].code] : C.acc);
  let jobType = -1;                                  // the "kinds of job" view: which kind is highlighted (-1 = off)
  // Effects are measured as drops in the answer's log-odds, averaged over the questions. For readers we turn them into
  // the chance of the right answer on a typical question (mean log-odds through the logistic), before and after; an
  // odds ratio ("35,000x less likely") overstates big effects on answers the model was already near-certain of.
  const P = (lo) => 1 / (1 + Math.exp(-lo));
  // enough digits to see a change near either end (99.97% vs 99.6%; 0.11% vs 0.003%)
  const pct = (p) => {
    const v = p * 100;
    if (v >= 99.5) { const rem = 100 - v; if (rem < 0.0001) return "over 99.9999%"; return v.toFixed(Math.min(4, Math.max(1, Math.ceil(-Math.log10(rem))))) + "%"; }
    if (v >= 10) return Math.round(v) + "%";
    if (v >= 1) return v.toFixed(1) + "%";
    if (v < 0.0001) return "under 0.0001%";
    return v.toFixed(Math.min(4, Math.max(2, Math.ceil(-Math.log10(v)) + 1))) + "%";
  };
  // a before/after pair of chances, with enough decimals that a real change never rounds away ("99.2% → 98.9%", not
  // "99% → 99%"); if it still does at 4 decimals, the change is negligible
  const pcts = (pa, pb) => {
    let a = pct(pa), b = pct(pb);
    for (let d = 1; d <= 4 && a === b; d++) { a = (pa * 100).toFixed(d) + "%"; b = (pb * 100).toFixed(d) + "%"; }
    return [a, b];
  };
  const chance = (base, drop) => {                   // compact, for lists: "99.2% → 96%"
    if (base == null) return null;
    const [a, b] = pcts(P(base), P(base - Math.max(0, drop)));
    return a === b ? `${a}, almost no change` : `${a} → ${b}`;
  };
  const transition = (base, drop, what) => {        // a phrase: "switching it off takes ... from 99.2% to 96%"
    if (base == null) return null;
    const [a, b] = pcts(P(base), P(base - Math.max(0, drop)));
    return a === b ? `switching off ${what} barely changes the chance of the right answer (${a})`
      : `switching off ${what} takes the chance of the right answer from ${a} to ${b}`;
  };

  let D, N, byLayer, cur = null, depth = 35, hover = -1, pinned = -1, showHeads = false, zoom = -1, CUT = 20, onlyLit = false, peek = -1;
  const layerCache = {}, pushCache = {};
  let jLayerSet = null;                              // layers whose J-lens readings are shown (empty until published)
  const jLayers = () => (jLayerSet = jLayerSet || fetch("data/push_j/layers.json").then((r) => (r.ok ? r.json() : []))
    .then((a) => new Set((Array.isArray(a) ? a : a.layers || []).map((x) => (typeof x === "number" ? x : x.show ? x.layer : -1))))
    .catch(() => new Set()));
  const cv = $("map"), ctx = cv.getContext("2d");
  let G = {};

  /* ---------- geometry and visibility ---------- */
  function geometry() {
    const w = cv.clientWidth, narrow = w < 600, dpr = window.devicePixelRatio || 1;
    // the subject names sit in a gap between layers 19 and 20, where the grouping by subject starts
    const rowH = narrow ? 14 : 19, top = 22, sp = zoom < 0 ? (narrow ? 66 : 78) : 36, left = narrow ? 20 : 28, lensW = narrow ? 34 : 64;
    const headsW = showHeads ? (narrow ? 40 : 76) : 0, gap = narrow ? 6 : 10;
    const mapW = w - left - lensW - headsW - gap * (showHeads ? 2 : 1);
    const h = top + rowH * N_LAYERS + sp + 28;
    cv.style.height = h + "px"; cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    G = { w, h, narrow, rowH, top, sp, left, mapW, headsX: left + mapW + gap, headsW, lensX: w - lensW, lensW };
  }
  const rowOf = (my) => { const yb = G.top + CUT * G.rowH; return my < yb ? Math.floor((my - G.top) / G.rowH) : my < yb + G.sp ? -1 : CUT + Math.floor((my - yb - G.sp) / G.rowH); };
  // where neuron k sits (x in 0..10000 of the map width), or null when the zoom hides it
  function nx(k) {
    if (zoom < 0) return N.x[k];
    if (N.L[k] < CUT) return (N.s[k] >> zoom) & 1 ? N.x[k] : null;
    if (N.r[k] !== zoom) return null;
    const r = D.regions[zoom]; return ((N.x[k] - r.x0) / Math.max(1, r.x1 - r.x0)) * 9600 + 200;
  }
  const px = (x) => G.left + (x / 10000) * G.mapW;
  const py = (L, y) => G.top + L * G.rowH + (L >= CUT ? G.sp : 0) + (y / 1000) * G.rowH;

  /* ---------- drawing ---------- */
  function draw() {
    const g = G; ctx.clearRect(0, 0, g.w, g.h);
    const font = (s) => `${s}px ${css.getPropertyValue("--sans")}`;
    ctx.font = font(g.narrow ? 9.5 : 11); ctx.textAlign = "left";
    const yCut = py(CUT, 0) - 3, yGap = py(CUT - 1, 1000) + 2;
    ctx.fillStyle = C.muted; ctx.fillText("Question goes in ↓", g.left, g.top - 8);
    ctx.fillText(g.narrow ? "↑ layers 0–19: not grouped by subject" : "↑ layers 0–19: not grouped by subject, as their jobs are less clear", g.left, yGap + 12);
    // subject columns, from layer 20 down, named in the gap above them
    if (zoom < 0) D.regions.forEach((r, j) => {
      const x0 = px(r.x0), x1 = px(r.x1);
      if (j % 2 === 0) { ctx.fillStyle = C.wash; ctx.fillRect(x0 - 2, yCut, x1 - x0 + 4, py(N_LAYERS, 0) - yCut + 2); }
      ctx.save(); ctx.translate((x0 + x1) / 2 - 2, yCut - 4); ctx.rotate(-Math.PI / 5);
      ctx.fillStyle = C.ink; ctx.fillText(r.label, 0, 0); ctx.restore();
    });
    else {
      ctx.fillStyle = C.wash; ctx.fillRect(g.left - 2, yCut, g.mapW + 4, py(N_LAYERS, 0) - yCut + 2);
      ctx.fillStyle = C.ink; ctx.font = font(13); ctx.fillText("↓ " + D.regions[zoom].name + " (tap “All subjects” to go back)", g.left, yCut - 5); ctx.font = font(g.narrow ? 9.5 : 11);
    }
    ctx.textAlign = "right"; ctx.fillStyle = C.muted;
    for (let L = 0; L < N_LAYERS; L += 5) ctx.fillText(L, g.left - 6, py(L, 800));
    ctx.fillText(35, g.left - 6, py(35, 800));
    ctx.textAlign = "left"; ctx.fillText("Answer comes out ↓", g.left, g.h - 8);
    // every mapped neuron: darker where it has a checked convenience name
    const lit = new Map(); if (cur) cur[4].forEach(([k, d]) => lit.set(k, d));
    const sz = zoom < 0 ? (g.narrow ? 1.6 : 1.9) : (g.narrow ? 2.6 : 3.2);
    if (!onlyLit) for (let k = 0; k < N.L.length; k++) {
      if (lit.has(k)) continue;
      const x = nx(k); if (x == null) continue;
      if (jobType >= 0) { ctx.fillStyle = C.cool; ctx.globalAlpha = 0.12; }
      else if (N.nm[k] >= 0 || (N.era[k] >= 0 && D.eras[N.era[k]].tested)) { ctx.fillStyle = C.ink; ctx.globalAlpha = 0.55; }
      else { ctx.fillStyle = C.cool; ctx.globalAlpha = N.key[k] ? 0.42 : 0.16; }
      ctx.fillRect(px(x) - sz / 2, py(N.L[k], N.y[k]) - sz / 2, sz, sz);
    }
    ctx.globalAlpha = 1;
    // the "kinds of job" view: every neuron with the chosen kind of job, in its colour
    if (jobType >= 0) {
      const b = 1 << jobType, r = zoom < 0 ? (g.narrow ? 1.8 : 2.4) : 3.4;
      ctx.fillStyle = jobColor(jobType);
      for (let k = 0; k < N.L.length; k++) {
        if (!(N.jt[k] & b)) continue;
        const x = nx(k); if (x == null) continue;
        ctx.beginPath(); ctx.arc(px(x), py(N.L[k], N.y[k]), r, 0, 7); ctx.fill();
      }
    }
    // the sweep
    if (cur && depth < N_LAYERS - 1) {
      const ys = depth + 1 === CUT ? yCut : py(depth + 1, 0);
      ctx.fillStyle = "rgba(255,255,255,0.6)"; ctx.fillRect(g.left - 2, ys, g.mapW + 4, py(N_LAYERS, 0) - ys);
      ctx.strokeStyle = C.acc; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(g.left - 4, py(depth + 1, 0)); ctx.lineTo(g.left + g.mapW + 4, py(depth + 1, 0)); ctx.stroke();
    }
    // the concept's neurons
    if (cur) {
      const maxD = Math.max(0.5, ...cur[4].map((t) => t[1] / 100));
      cur[4].slice().reverse().forEach(([k, d]) => {
        const xx = nx(k); if (xx == null || N.L[k] > depth) return;
        const r = litRadius(d, maxD);
        const x = px(xx), y = py(N.L[k], N.y[k]);
        const t3 = cur[4].find((t) => t[0] === k);
        ctx.globalAlpha = N.L[k] === depth ? 0.25 : 0.12; ctx.fillStyle = cur[14] === "fact" ? jobColor(t3 && t3.length > 2 ? t3[2] : -1) : C.acc;
        ctx.beginPath(); ctx.arc(x, y, r * 2.2, 0, 7); ctx.fill();
        ctx.globalAlpha = 0.9; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
        ctx.globalAlpha = 1; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke();
      });
    }
    // attention heads (optional)
    if (showHeads) {
      const cw = g.headsW / 32, hd = new Map((cur ? cur[6] : []).map(([h, d]) => [h, d])), hmax = Math.max(50, ...hd.values());
      ctx.fillStyle = C.muted; ctx.fillText("heads", g.headsX, g.top - 8);
      for (let L = 0; L < N_LAYERS; L++) for (let h = 0; h < 32; h++) {
        const v = hd.get(`L${L}H${h}`), on = v && L <= depth;
        ctx.fillStyle = on ? C.acc : C.wash; ctx.globalAlpha = on ? 0.25 + 0.75 * Math.min(1, v / hmax) : 1;
        ctx.fillRect(g.headsX + h * cw, py(L, 120), Math.max(1, cw - 0.6), g.rowH * 0.76);
      }
      ctx.globalAlpha = 1;
    }
    // the answer's rank at each layer (logit lens): grows from 1,000th to 1st, coloured once in the top 10
    ctx.fillStyle = C.muted; ctx.fillText(g.narrow ? "answer" : "answer's rank", g.lensX, g.top - 8);
    const lens = cur && cur[5];
    for (let L = 0; L < N_LAYERS; L++) {
      ctx.fillStyle = C.wash; ctx.fillRect(g.lensX, py(L, 150), g.lensW, g.rowH * 0.7);
      if (!lens || L > depth) continue;
      const rank = lens[L + 1], f = Math.max(0, 1 - Math.log10(Math.max(1, rank)) / 3);
      ctx.fillStyle = rank <= 10 ? C.acc : C.cool; ctx.fillRect(g.lensX, py(L, 150), Math.max(1, f * g.lensW), g.rowH * 0.7);
    }
    if (cur && !lens) { ctx.fillStyle = C.muted; ctx.fillText("not", g.lensX + 2, py(2, 800)); ctx.fillText("measured", g.lensX + 2, py(3, 800)); }
    if (hover >= 0 && nx(hover) != null) {
      ctx.strokeStyle = C.ink; ctx.lineWidth = pinned === hover ? 2.2 : 1.5; ctx.beginPath(); ctx.arc(px(nx(hover)), py(N.L[hover], N.y[hover]), pinned === hover ? 8 : 6, 0, 7); ctx.stroke();
    }
    if (peek >= 0 && peek !== hover && nx(peek) != null) {        // what's under the pointer while another neuron is pinned
      ctx.strokeStyle = C.muted; ctx.lineWidth = 1; ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.arc(px(nx(peek)), py(N.L[peek], N.y[peek]), 6, 0, 7); ctx.stroke(); ctx.setLineDash([]);
    }
  }

  /* ---------- pointer: neurons, and the column names (click to zoom) ---------- */
  // the drawn radius of a lit neuron (shared by draw and the hit test, so what you see is what you can tap)
  function litRadius(d, maxD) {
    return (G.narrow ? 1.8 : 2.4) + (G.narrow ? 4 : 6.5) * Math.sqrt(Math.max(0, d / 100) / maxD);
  }
  // every highlighted neuron whose drawn dot (plus a margin) covers the point, best match first
  function litHits(mx, my) {
    if (!cur) return [];
    const maxD = Math.max(0.5, ...cur[4].map((t) => t[1] / 100)), pad = G.narrow ? 7 : 5, hits = [];
    for (const [k, d] of cur[4]) {
      const x = nx(k); if (x == null || N.L[k] > depth) continue;
      const r = litRadius(d, maxD) + pad, dist = Math.hypot(px(x) - mx, py(N.L[k], N.y[k]) - my);
      if (dist <= r) hits.push([dist / r, k]);
    }
    return hits.sort((a, b) => a[0] - b[0]).map((h) => h[1]);
  }
  function nearest(mx, my) {
    // 1) highlighted neurons first: anywhere inside the dot as drawn, plus a margin, wins over the dots beneath it
    const hits = litHits(mx, my);
    if (hits.length) return hits[0];
    if (jobType >= 0) {                              // the highlighted kind of job, with its larger dots
      let best = -1, bd = (G.narrow ? 12 : 8) ** 2;
      for (let k = 0; k < N.L.length; k++) {
        if (!(N.jt[k] & (1 << jobType))) continue;
        const x = nx(k); if (x == null) continue;
        const dd = (px(x) - mx) ** 2 + (py(N.L[k], N.y[k]) - my) ** 2;
        if (dd < bd) { bd = dd; best = k; }
      }
      return best;
    }
    // 2) otherwise the nearest background neuron, within a few pixels
    const L = rowOf(my); if (L < 0) return -1;
    let best = -1, bd = (G.narrow ? 9 : 6) ** 2;
    const litSet = new Set(cur ? cur[4].map((t) => t[0]) : []);
    for (const LL of [L - 1, L, L + 1]) {
      if (LL < 0 || LL >= N_LAYERS) continue;
      for (const k of byLayer[LL]) {
        if (onlyLit && !litSet.has(k)) continue;
        if (jobType >= 0 && !(N.jt[k] & (1 << jobType))) continue;
        const x = nx(k); if (x == null) continue;
        const dx = px(x) - mx, dy = py(N.L[k], N.y[k]) - my;
        const dd = dx * dx + dy * dy;
        if (dd < bd) { bd = dd; best = k; }
      }
    }
    return best;
  }
  function columnAt(mx, my) {
    if (zoom >= 0 || rowOf(my) !== -1) return -1;
    return D.regions.findIndex((r) => mx >= px(r.x0) - 4 && mx <= px(r.x1) + 4);
  }
  async function describe(k) {
    const L = N.L[k], i = N.i[k];
    $("info").innerHTML = `<b>Layer ${L}, neuron ${i}.</b> <span class="muted">Loading…</span>`;
    const layer = await (layerCache[L] = layerCache[L] || fetch(`data/n/L${L}.json`).then((r) => r.json()));
    if (hover !== k && pinned !== k) return;
    const v = layer[String(i)] || {}, f = v.findings || [];
    const mine = cur && cur[4].find((t) => t[0] === k);
    const uses = (v.uses || []).slice().sort((a, b) => (b[3] || 0) - (a[3] || 0));
    const names = []; uses.forEach((u) => { const n = D.cname[u[0]]; if (n && !names.some((x) => x[0] === n) && names.length < 4) names.push([n, u[3]]); });
    const nUses = new Set(uses.map((u) => D.cname[u[0]])).size;
    let h = (pinned === k ? `<button type="button" class="chip pin" id="unpin" title="Unpin: hovering will show other neurons again">📌 Pinned · unpin</button> ` : "") +
      `<b>Layer ${L} of 36, neuron ${i}</b>`;
    if (mine && !cur[14] && cur[11] != null) h += ` · on a typical question about <b>${esc(cur[1])}</b>, ${transition(cur[11], mine[1] / 100, "this neuron")}`;
    if (mine && cur[14] === "fact") {
      const d = mine[1] / 100, how = d >= 2 ? "much less likely" : d >= 0.5 ? "clearly less likely" : "a little less likely";
      h += ` · switching it off makes “${esc(cur[2])}” ${how} as the answer to “${esc(cur[1])}: ${esc(cur[16])}?”`;
    }
    h += ".<br>";
    const kinds = D.job_types.map((t, j) => (N.jt[k] & (1 << j) ? `<span style="color:${JOBC[t.code]}">●</span> ${esc(t.label)}` : null)).filter(Boolean);
    if (kinds.length) h += `<span class="muted">Kinds of job found so far:</span> ${kinds.join(" · ")}<br>`;
    const era = N.era && N.era[k] >= 0 ? D.eras[N.era[k]] : null;
    if (era) {
      const a = era.aq, ways = [["year", "a year"], ["century", "a century"], ["decade", "a decade"], ["period", "a period name"]];
      const list = (xs, and) => (xs.length > 1 ? xs.slice(0, -1).join(", ") + ` ${and} ` + xs[xs.length - 1] : xs[0] || "");
      const hi = ways.filter(([f]) => a[f] >= 0.9).map(([, w]) => w), lo = ways.filter(([f]) => a[f] < 0.6).map(([, w]) => w);
      const mid = ways.filter(([f]) => a[f] >= 0.8).map(([, w]) => w), adj = esc(era.c.replace(" ", "-"));
      if (era.tested) h += `<b>Tested job: it picks out the ${esc(era.c)}.</b> <span class="muted">In a targeted test on 280 historical people, places and works, its firing separated ${adj} ones from the rest almost perfectly, whether the question asked for ${list(hi, "or")}.</span><br>`;
      else if (lo.length && mid.length) h += `<b>Tested: it picks out ${adj} answers only when they are written as ${list(mid, "or")}, not as ${list(lo, "or")}.</b> <span class="muted">So it seems to track how the date is written (its digits), not the era itself.</span><br>`;
      else h += `<b>Tested: it leans towards the ${esc(era.c)}</b> <span class="muted">but not cleanly enough in every way of asking for us to call it its job.</span><br>`;
      h += era.ko != null ? `Switching it off makes the model worse at ${adj} dates (when asked for ${list(era.ko_formats.map((f) => "a " + f), "or")}).<br>`
        : `Switching it off on its own barely changes those dates: other neurons seem to cover for it.<br>`;
    }
    const sc = D.scales && D.scales[String(k)];
    if (sc) {
      const ends = (sc.ends || []).filter(Boolean).map(esc).join(" and ");
      h += sc.status === "one-sided"
        ? `<b>Tested: maybe a two-ended scale</b> (${esc(sc.pos)} ↔ ${esc(sc.neg)}). <span class="muted">Its two ends were described from its weights and checked against the concepts that use them, but pushing the neuron moved the model's answers towards only one end.</span><br>`
        : `<b>Tested: ${sc.status === "likely" ? "probably " : ""}a two-ended scale.</b> Firing positively pushes towards <b>${esc(sc.pos)}</b>; firing negatively, towards the opposite end, <b>${esc(sc.neg)}</b>. ` +
          `<span class="muted">${sc.status === "likely" ? "Its description is less certain, but " : "Described from its weights alone, checked against the concepts that use each end, and "}pushing it each way moves the model's answers towards that end${ends ? ` (tested on ${ends})` : ""}.</span><br>`;
    }
    if (N.nm[k] >= 0) h += `<b>Convenience name: “${esc(D.names[N.nm[k]])}”</b><sup class="fn" tabindex="0" data-fn="A short label to help you read the map, not the neuron's true meaning. It passed a blind check: given only this name, a second AI picked out a concept that needs this neuron, which the namer hadn't been shown.">1</sup><br>`;
    else if (!era) h += `<b>Job not yet clear.</b> <span class="muted">It matters for the concepts below, but we can't yet sum up what it does in a few words with confidence.</span><br>`;
    if (ROLE_TEXT[N.role[k]]) h += ROLE_TEXT[N.role[k]] + "<br>";
    // A neuron's activity can be positive or negative, and the two push the model in opposite directions, often with
    // different jobs. For each side: the blind web-text description (if any) and the words it pushes towards (weights).
    const getJson = (cache, url) => (cache[url] = cache[url] || fetch(url).then((r) => (r.ok ? r.json() : {})).catch(() => ({})));
    const push = (await getJson(pushCache, `data/push/L${L}.json`))[String(i)];
    // the J-lens reading (what a nudge here does to the final layer, through the later layers), in layers where it
    // passed its reliability check (data/push_j/layers.json, fixed before the results)
    const jl = (await jLayers()).has(L) ? (await getJson(pushCache, `data/push_j/L${L}.json`))[String(i)] : null;
    if (hover !== k && pinned !== k) return;
    // The sign we have is the neuron's sign at one moment: as Qwen starts its answer (the last token of the question).
    // Answers are often several tokens ("A" + "est" + "hetics"), and a neuron can matter at a later one, firing the
    // other way. So we tag the side it takes at the start, and separately flag a side whose pushed words are part of
    // the answer itself.
    const side = mine && !cur[14] && mine.length > 2 ? mine[2] : 0;
    const cleanW = (ws) => ws.map((w) => w.trim()).filter((w, j, a) => w && !/^<\|.*\|>$/.test(w) && a.findIndex((x) => x.toLowerCase() === w.toLowerCase()) === j).slice(0, 8);
    const ans = cur && !cur[14] ? String(cur[2] || cur[1]).toLowerCase().replace(/[^a-zÀ-ɏ㐀-鿿]+/g, "") : "";
    const inAnswer = (ws) => ws.filter((w) => { const x = w.toLowerCase().replace(/[^a-zÀ-ɏ㐀-鿿]+/g, ""); return x.length >= 3 && ans.includes(x); });
    [["pos", 1, "positively"], ["neg", -1, "negatively"]].forEach(([pole, s, word], j) => {
      const lab = f.find((x) => x.kind === "label" && x.metric && x.metric.pole === pole);
      const ws = push ? cleanW(push[j]) : [], wj = jl ? cleanW(jl[j]) : [];
      if (!lab && !ws.length && !wj.length) return;
      const atStart = side === s, part = ans ? [...new Set(inAnswer(ws).concat(inAnswer(wj)))] : [];
      const words = (a) => a.map((w) => `<code>${esc(w)}</code>`).join("");
      const tags = (atStart ? ` <span class="tag-on">· how it fires as Qwen starts answering “${esc(cur[1])}”</span>` : "") +
        (part.length ? ` <span class="tag-on">· pushes part of the answer (“${part.slice(0, 3).map(esc).join("”, “")}”)</span>` : "");
      h += `<div class="side${atStart || part.length ? " on" : ""}"><b>When it fires ${word}</b>${tags}: ` +
        (lab ? `in web text, “${esc(lab.title)}”${lab.metric.clean ? " (checked: one clear job)" : ""}. ` : "") +
        (wj.length ? (ws.length ? `<span class="words">Directly, it pushes the model towards ${words(ws)}</span><br>` : "") +
          `<span class="words">Through the later layers, it pushes towards ${words(wj)} <span class="muted">(J-lens)</span></span>`
          : ws.length ? `<span class="words">It pushes the model towards ${words(ws)}</span>` : "") + `</div>`;
    });
    if (side && push && ans) {
      const startW = cleanW(push[side === 1 ? 0 : 1]), otherW = cleanW(push[side === 1 ? 1 : 0]);
      if (!inAnswer(startW).length && inAnswer(otherW).length)
        h += `<span class="muted">The answer is written in several pieces, so this neuron probably does its part later in the answer, firing the other way.</span><br>`;
    }
    if (push && L < 20) h += jl ? `<span class="muted">(The direct words are read from its weights, which is much less reliable this early in the model; the J-lens words follow its push through the later layers.)</span><br>`
      : `<span class="muted">(Pushed words are read from its weights; that reading is much less reliable this early in the model.)</span><br>`;
    // its jobs in the facts test, in one plain sentence (job classes; the raw knockout summaries live on the neuron page)
    const jc = {};
    f.forEach((x) => {
      const m = x.kind === "job class" && /^facts/.test(x.source || "") && /is an? ([\w+-]+) neuron(?: \((.*)\))?/.exec(x.title || "");
      if (!m) return;
      const c = m[1] === "entity+relation" ? "relation" : m[1];
      (jc[c] = jc[c] || new Set()); if (m[2] && m[2] !== "?") jc[c].add(m[2]);
    });
    const ents = (c) => { const a = [...jc[c]]; return a.slice(0, 3).map(esc).join(", ") + (a.length > 3 ? ` and ${a.length - 3} more` : ""); };
    const fb = [];
    if (jc.entity) fb.push(`recognises ${ents("entity")} at the name`);
    if (jc["fact-specific"]) fb.push(`is needed for one fact${jc["fact-specific"].size > 1 ? " each" : ""} about ${ents("fact-specific")}`);
    if (jc.relation) fb.push(`is needed for one kind of question across many concepts (seen with ${ents("relation")})`);
    if (jc.generic) fb.push("breaks facts of every kind");
    if (fb.length) h += `In a separate test on facts about landmarks, people and works, it ${fb.join("; it ")}.<br>`;
    f.filter((x) => x.kind === "fact job").slice(0, 1).forEach((x) => { h += `${esc(x.title)}<br>`; });
    // Links with the concept's other key neurons, in one plain sentence: the 3 strongest each way, strength in words
    // (a link's size is how much the other neuron's activity changes, which can exceed 100% when it flips sign)
    if (mine && cur[10] && cur[10].length) {
      // a partner neuron: its checked name, else what it does (from its role), else its layer
      const ROLE_NOUN = { 1: ["an answer-writing neuron", "answer-writing neurons"], 2: ["a neuron that pushes the answer", "neurons that push the answer"],
        4: ["an early neuron", "early neurons"] };
      const nm1 = (q) => (N.nm[q] >= 0 ? [`“${esc(D.names[N.nm[q]])}”`, null] : ROLE_NOUN[N.role[q]] || [`a layer-${N.L[q]} neuron`, `layer-${N.L[q]} neurons`]);
      const NUM = ["", "", "two", "three"];
      const pick = (ls) => ls.filter((l) => Math.abs(l[2]) >= 10).sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]));
      const outL = pick(cur[10].filter((l) => l[0] === k)), inL = pick(cur[10].filter((l) => l[1] === k));
      const list = (ls, other) => {                 // merge repeats: "two answer-writing neurons (one strongly)"
        const groups = new Map();
        ls.forEach((l) => { const [one, many] = nm1(l[other]), key = one; const g = groups.get(key) || { one, many, n: 0, strong: 0 }; g.n++; if (Math.abs(l[2]) >= 50) g.strong++; groups.set(key, g); });
        return [...groups.values()].map((g) => (g.n > 1 && g.many ? `${NUM[g.n] || g.n} ${g.many}` : g.one) + (g.strong ? (g.n > 1 ? (g.strong === g.n ? (g.n === 2 ? " (both strongly)" : " (all strongly)") : ` (${g.strong === 1 ? "one" : NUM[g.strong] || g.strong} strongly)`) : " (strongly)") : "")).join(", ");
      };
      const phrase = (ls, pos, neg, other) => {
        const sup = ls.filter((l) => l[2] > 0).slice(0, 3), sp = ls.filter((l) => l[2] < 0).slice(0, 3), bits = [];
        if (sup.length) bits.push(`${pos} ${list(sup, other)}`);
        if (sp.length) bits.push(`${neg} ${list(sp, other)}`);
        return bits.join("; ");
      };
      const parts = [phrase(outL, "it feeds", "it holds back", 1), phrase(inL, "it is fed by", "it is held back by", 0)].filter(Boolean);
      const more = outL.length + inL.length - [...outL.filter((l) => l[2] > 0).slice(0, 3), ...outL.filter((l) => l[2] < 0).slice(0, 3), ...inL.filter((l) => l[2] > 0).slice(0, 3), ...inL.filter((l) => l[2] < 0).slice(0, 3)].length;
      if (parts.length) h += `Within ${esc(cur[1])}'s key neurons, ${parts.join("; ")}${more > 0 ? `, and ${more} weaker link${more > 1 ? "s" : ""}` : ""}. <span class="muted">(Measured by switching one neuron off and watching the other.)</span><br>`;
    }
    if (names.length) h += `Matters most for: ${names.map(([n, d]) => { const ch = chance(D.cbase[n], d); return `${esc(n)}${ch ? ` (${ch})` : ""}`; }).join(", ")}${nUses > names.length ? `, and ${nUses - names.length} more` : ""}. <span class="muted">(The chance of the right answer on a typical question, before → after switching this neuron off.)</span>`;
    if (LINKS) h += ` <a href="index.html#/neuron/L${L}:${i}">Everything we know about it →</a>`;
    $("info").innerHTML = h;
  }
  function pointer(e, tap) {
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (tap) { const col = columnAt(mx, my); if (col >= 0) { setZoom(col); return; } }
    cv.style.cursor = columnAt(mx, my) >= 0 ? "pointer" : "crosshair";
    const k = nearest(mx, my);
    // while a neuron is pinned, hovering only shows a faint ring; the description stays put
    if (!tap && pinned >= 0) { if (k !== peek) { peek = k; draw(); } return; }
    if (k === hover && !tap) return;
    if (k < 0 && !tap) { hover = -1; draw(); return; }
    if (tap && k >= 0 && k === pinned) { unpin(); return; }      // tapping the pinned neuron again unpins it
    if (tap && k < 0) return;
    hover = k; peek = -1; if (tap) pinned = k; draw(); describe(k);
    if (tap && k >= 0) { const j = stepList().indexOf(k); if (j >= 0) { stepIdx = j; updateStepper(); } }   // keep Next/Previous in step
    // dots that overlap under the finger: offer them all, so a tap never silently picks the wrong one
    if (tap) {
      const hits = litHits(mx, my);
      $("overlap").innerHTML = hits.length > 1 ? `<span class="muted">${hits.length} neurons overlap here:</span> ` +
        hits.map((q) => `<button type="button" class="chip${q === k ? " sel" : ""}" data-k="${q}">layer ${N.L[q]}, neuron ${N.i[q]}${N.nm[q] >= 0 ? ` · ${esc(D.names[N.nm[q]])}` : ""}</button>`).join("") : "";
    }
  }
  $("overlap").addEventListener("click", (e) => {
    const b = e.target.closest("[data-k]"); if (!b) return;
    const k = +b.dataset.k; pinned = hover = k; draw(); describe(k);
    $("overlap").querySelectorAll(".chip").forEach((c) => c.classList.toggle("sel", +c.dataset.k === k));
    const j = stepList().indexOf(k); if (j >= 0) { stepIdx = j; updateStepper(); }
  });
  cv.addEventListener("mousemove", (e) => pointer(e, false));
  cv.addEventListener("click", (e) => pointer(e, true));
  cv.addEventListener("mouseleave", () => { peek = -1; hover = pinned; draw(); });
  function unpin() {
    pinned = -1; peek = -1; hover = -1; draw();
    $("info").innerHTML = `<span class="muted">Unpinned. Hover over a dot to see what that neuron does, or tap one to pin it here.</span>`;
  }
  $("info").addEventListener("click", (e) => { if (e.target.closest("#unpin")) unpin(); });

  /* ---------- stepping through the highlighted neurons (easier than tapping dots on a phone) ---------- */
  let stepIdx = -1;
  // the neurons to step through, top of the model first: the concept's / fact's / century's neurons, or every
  // neuron of the chosen kind of job; only those visible in the current zoom
  function stepList() {
    let ks;
    if (jobType >= 0) { ks = []; for (let k = 0; k < N.L.length; k++) if (N.jt[k] & (1 << jobType)) ks.push(k); }
    else ks = cur ? cur[4].map((t) => t[0]) : [];
    const drop = new Map(cur ? cur[4].map((t) => [t[0], t[1]]) : []);
    return ks.filter((k) => nx(k) != null).sort((a, b) => N.L[a] - N.L[b] || (drop.get(b) || 0) - (drop.get(a) || 0) || N.x[a] - N.x[b]);
  }
  function stepTo(j) {
    const ks = stepList();
    if (!ks.length) { updateStepper(); return; }
    stop(); stepIdx = Math.max(0, Math.min(ks.length - 1, j)); $("overlap").innerHTML = "";
    const k = ks[stepIdx];
    if (cur && depth < 35) setDepth(35);              // stepping shows all of the concept's neurons; the sweep is only on first pick
    pinned = hover = k; draw(); describe(k); updateStepper();
  }
  function restartSteps() {
    stepIdx = -1; pinned = hover = -1; $("overlap").innerHTML = "";
    if (cur && !timer) setDepth(35); else draw();
    $("info").innerHTML = `<span class="muted">Press <b>Next</b> to step through ${jobType >= 0 ? "these neurons" : "its neurons"} from the top of the model down, or hover over or tap a dot.</span>`;
    updateStepper();
  }
  function updateStepper() {
    const n = stepList().length;
    $("stPrev").disabled = stepIdx <= 0; $("stNext").disabled = n === 0 || stepIdx >= n - 1;
    $("stPos").textContent = n ? (stepIdx >= 0 ? `Neuron ${stepIdx + 1} of ${n.toLocaleString("en-GB")}` : `${n.toLocaleString("en-GB")} neuron${n > 1 ? "s" : ""} to step through`) : "";
  }
  $("info").addEventListener("click", (e) => { const s = e.target.closest("sup.fn"); if (s) s.classList.toggle("open"); });   // tap to show a footnote
  $("stNext").addEventListener("click", () => stepTo(stepIdx + 1));
  $("stPrev").addEventListener("click", () => stepTo(stepIdx - 1));
  $("stRestart").addEventListener("click", restartSteps);
  document.addEventListener("keydown", (e) => {
    if (e.target.closest && e.target.closest("input, select, textarea")) return;
    if (e.key === "ArrowRight" && !$("stNext").disabled) { e.preventDefault(); stepTo(stepIdx + 1); }
    if (e.key === "ArrowLeft" && !$("stPrev").disabled) { e.preventDefault(); stepTo(stepIdx - 1); }
  });

  /* ---------- zoom into one subject ---------- */
  function setZoom(j) {
    zoom = j; pinned = -1; hover = -1;
    document.querySelectorAll("#subjects .chip").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.z === j)));
    renderPicks();
    if (j >= 0 && cur && cur[3] !== D.regions[j].name) {
      const inSec = conceptsIn(j); if (inSec.length) { setConcept(inSec[0], true); geometry(); return; }
    }
    geometry(); draw(); updateSummary(); stepIdx = -1; updateStepper();
  }
  function conceptsIn(j) {
    const name = D.regions[j].name;
    return D.concepts.filter((c) => c[3] === name).sort((a, b) => b[9] - a[9] || (a[7] || 99) - (b[7] || 99) || a[1].localeCompare(b[1]));
  }
  function renderPicks() {
    const list = zoom >= 0 ? conceptsIn(zoom).slice(0, 12) : D.picks.map((s) => D.concepts.find((x) => x[0] === s));
    $("picks").innerHTML = list.map((c) => `<button type="button" class="chip" data-s="${esc(c[0])}">${esc(c[1])}</button>`).join("") ||
      `<span class="muted">No concepts from this subject have been mapped yet.</span>`;
    $("conceptList").innerHTML = (zoom >= 0 ? conceptsIn(zoom) : D.concepts).map((c) => `<option value="${esc(c[1])}">`).join("");
  }

  /* ---------- concept and sweep ---------- */
  let timer = null;
  function updateSummary() {
    if (jobType >= 0) {
      const t = D.job_types[jobType];
      $("summary").innerHTML = `<span style="color:${JOBC[t.code]}">●</span> <b>${esc(t.label)}</b>: ${t.n.toLocaleString("en-GB")} neuron${t.n === 1 ? "" : "s"} so far. ${esc(t.what || t.how)} <a href="#kinds">How we found these</a> · <a href="#" id="jobOff">Back to concepts</a>`;
      $("jobOff").addEventListener("click", (e) => { e.preventDefault(); setJobType(-1); });
      return;
    }
    const c = cur; if (!c) return;
    if (c[14] === "fact") {
      const cnt = {}; c[4].forEach((t) => { cnt[t[2]] = (cnt[t[2]] || 0) + 1; });
      const parts = Object.entries(cnt).sort((a, b) => b[1] - a[1]).map(([j, n]) => `<span style="color:${jobColor(+j)}">●</span> ${n} ${j >= 0 ? esc(D.job_types[+j].label.toLowerCase()) : "with no job class yet"}`);
      $("summary").innerHTML = `<b>${esc(c[1])}: ${esc(c[16])}?</b> → <b>${esc(c[2])}</b>. Switching off any of these ${c[4].length} neurons makes that answer less likely: ${parts.join(", ")}. ` +
        `<span class="muted">From a separate test of 333 facts: each neuron was switched off in turn on every fact, so its job is whatever it breaks. The answer bar on the right doesn't apply.</span>`;
      return;
    }
    if (c[14] === "era") {
      const units = c[4].map((t) => D.eras[N.era[t[0]]]), tested = units.filter((e) => e.tested).length;
      $("summary").innerHTML = `<b>${esc(c[1])}</b>: ${units.length} neuron${units.length > 1 ? "s" : ""} that pick out the ${esc(c[1])} in a targeted test (asking 280 historical people, places and works for their date in four ways)` +
        `${tested ? `; ${tested} of them do so however the question is asked` : ""}. Bigger dots: switching the neuron off lowers ${esc(c[1])} answers more. Hover over one for details.` +
        ` <span class="muted">These were found by a different test from the concept maps, so the answer bar on the right doesn't apply.</span>`;
      return;
    }
    const answer = c[2] && c[2].toLowerCase() !== c[1].toLowerCase() ? ` (Qwen answers “${esc(c[2])}”)` : "";
    const lens = c[5], first = lens ? lens.findIndex((r) => r <= 10) - 1 : null;
    const hidden = zoom >= 0 ? c[4].filter((t) => nx(t[0]) == null).length : 0;
    const k5 = c[12] && c[13] ? c[12][c[13].indexOf(5)] : null;
    $("summary").innerHTML = `<b>${esc(c[1])}</b>${answer}.` +
      (c[11] != null && c[12] ? ` On a typical question, ${transition(c[11], c[12][0], "its top neuron")}${k5 != null ? `, and switching off its top five takes it to ${pcts(P(c[11]), P(c[11] - Math.max(0, k5)))[1]}` : ""}.` : "") +
      (lens && first >= 0 ? ` The answer reaches the top 10 at layer ${first}.` : lens ? "" : " (Answer ranks not measured for this concept.)") +
      ` <a href="#fig-graph">Figure 2</a> groups them and shows how they feed each other.` +
      (hidden ? ` <span class="muted">${hidden} of its 30 neurons sit in other columns; <a href="#" id="unzoom">show all subjects</a> to see them.</span>` : "") +
      (LINKS ? ` <a href="index.html#/concept/${encodeURIComponent(c[0])}">Details →</a>` : "");
    const uz = $("unzoom"); if (uz) uz.addEventListener("click", (e) => { e.preventDefault(); setZoom(-1); });
  }
  function setJobType(j) {
    jobType = j; pinned = -1; hover = -1;
    document.querySelectorAll("#jobs .chip").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.j === j)));
    if (j >= 0) { stop(); cur = null; depth = 35; updateSummary(); restartSteps(); announce(); }
    else setConcept(D.concepts.find((x) => x[0] === D.picks[0]) || D.concepts[0], true);
  }
  function setConcept(c, play) {
    if (jobType >= 0) { jobType = -1; document.querySelectorAll("#jobs .chip").forEach((b) => b.setAttribute("aria-pressed", "false")); }
    cur = c; pinned = -1; hover = -1;
    stepIdx = -1; $("overlap").innerHTML = "";
    $("info").innerHTML = `<span class="muted">Press <b>Next</b> to step through its neurons from the top of the model down, or hover over or tap a dot.</span>`;
    $("pick").value = c[14] ? "" : c[1]; updateSummary(); updateStepper(); announce();
    if (play) sweep(); else setDepth(35);
  }
  // Figure 2 (graphs.js) follows the concept shown here
  function announce() { document.dispatchEvent(new CustomEvent("neuronmap:concept", { detail: cur })); }
  function setDepth(v) { depth = v; $("depth").value = v; $("depthVal").textContent = v < 0 ? "–" : v; draw(); }
  function sweep() {
    stop(); let v = -1; setDepth(v); $("play").textContent = "❚❚"; $("play").setAttribute("aria-label", "Pause");
    timer = setInterval(() => { v++; setDepth(v); if (v >= 35) stop(); }, 140);
  }
  function stop() { if (timer) clearInterval(timer); timer = null; $("play").textContent = "▶"; $("play").setAttribute("aria-label", "Play"); }
  $("play").addEventListener("click", () => (timer ? stop() : sweep()));
  $("depth").addEventListener("input", (e) => { stop(); setDepth(+e.target.value); });
  $("heads").addEventListener("change", (e) => { showHeads = e.target.checked; geometry(); draw(); });
  $("onlyLit").addEventListener("change", (e) => { onlyLit = e.target.checked; draw(); });
  const pickByName = () => { const c = D.byName[$("pick").value.toLowerCase()]; if (c) setConcept(c, true); };
  $("pick").addEventListener("change", pickByName);
  $("pick").addEventListener("keydown", (e) => { if (e.key === "Enter") pickByName(); });
  $("rand").addEventListener("click", () => { const pool = zoom >= 0 ? conceptsIn(zoom) : D.concepts; setConcept(pool[Math.floor(Math.random() * pool.length)], true); });
  $("picks").addEventListener("click", (e) => { const b = e.target.closest("[data-s]"); if (b) setConcept(D.concepts.find((x) => x[0] === b.dataset.s), true); });
  $("subjects").addEventListener("click", (e) => { const b = e.target.closest("[data-z]"); if (b) setZoom(+b.dataset.z); });
  $("jobs").addEventListener("click", (e) => { const b = e.target.closest("[data-j]"); if (b) setJobType(+b.dataset.j === jobType ? -1 : +b.dataset.j); });
  $("factPick").addEventListener("change", (e) => { const f = D.facts[+e.target.value]; if (f) { if (zoom >= 0) setZoom(-1); setConcept(f, true); } });
  $("factChips").addEventListener("click", (e) => { const b = e.target.closest("[data-f]"); if (b) { if (zoom >= 0) setZoom(-1); setConcept(D.facts[+b.dataset.f], true); $("factPick").value = b.dataset.f; } });
  $("eras").addEventListener("click", (e) => { const b = e.target.closest("[data-e]"); if (b) { if (zoom >= 0) setZoom(-1); setConcept(D.centuries[+b.dataset.e], true); } });
  window.addEventListener("resize", () => { geometry(); draw(); });

  fetch("data/neuronmap.json").then((r) => r.json()).then((d) => {
    D = d; N = d.neurons; CUT = d.early_cut || 20;
    byLayer = Array.from({ length: N_LAYERS }, () => []);
    for (let k = 0; k < N.L.length; k++) byLayer[N.L[k]].push(k);
    D.byName = {}; D.cname = {};
    D.concepts.forEach((c) => { D.byName[c[1].toLowerCase()] = c; });
    D.cbase = {};
    fetch("data/concepts.json").then((r) => r.json()).then((cj) => {
      const ni = cj.cols.indexOf("name"), bi = cj.cols.indexOf("baseline");
      cj.rows.forEach((r, j) => { D.cname[j] = r[ni]; if (r[bi] != null) D.cbase[r[ni]] = r[bi]; });
    });
    $("subjects").innerHTML = `<button type="button" class="chip" data-z="-1" aria-pressed="true">All subjects</button>` +
      D.regions.map((r, j) => (r.name === "Other" ? "" : `<button type="button" class="chip" data-z="${j}" aria-pressed="false">${esc(r.label)}</button>`)).join("");
    $("eras").innerHTML = (D.centuries || []).length ? `<span class="muted">Or a century:</span> ` +
      D.centuries.map((c, j) => `<button type="button" class="chip" data-e="${j}">${esc(c[1])}</button>`).join("") : "";
    // facts: a select grouped by thing, plus a few examples that show all three stages
    const byEnt = {}; (D.facts || []).forEach((f, j) => (byEnt[f[1]] = byEnt[f[1]] || []).push(j));
    $("factPick").innerHTML = `<option value="">Pick a fact…</option>` + Object.keys(byEnt).sort((a, b) => a.localeCompare(b)).map((e) =>
      `<optgroup label="${esc(e)}">${byEnt[e].map((j) => `<option value="${j}">${esc(e)}: ${esc(D.facts[j][16])}? → ${esc(D.facts[j][2])}</option>`).join("")}</optgroup>`).join("");
    const ex = [["Taj Mahal", "century"], ["Statue of Liberty", "century"], ["Eiffel Tower", "city"], ["Sagrada Família", "designer"]]
      .map(([e, r]) => (D.facts || []).findIndex((f) => f[1] === e && f[15] === r)).filter((j) => j >= 0);
    $("factChips").innerHTML = ex.map((j) => `<button type="button" class="chip" data-f="${j}">${esc(D.facts[j][1])}: ${esc(D.facts[j][16])}?</button>`).join("");
    $("jobs").innerHTML = `<span class="muted">Or see one kind of job:</span> ` + D.job_types.map((t, j) => t.n ? `<button type="button" class="chip" data-j="${j}" aria-pressed="false"><span style="color:${JOBC[t.code]}">●</span> ${esc(t.label)} <span class="muted">${t.n.toLocaleString("en-GB")}</span></button>` : "").join("");
    $("kinds").innerHTML = D.job_types.map((t) => `<li><span style="color:${JOBC[t.code]}">●</span> <b>${esc(t.label)}</b> <span class="muted">(${t.n.toLocaleString("en-GB")} neurons)</span>. ${esc(t.what || t.how)}` +
      (t.what ? `<details><summary>How we found these</summary><p>${esc(t.how)}</p></details>` : "") + `</li>`).join("");
    const nNamed = N.nm.filter((x) => x >= 0).length;
    $("namedCount").textContent = nNamed.toLocaleString("en-GB");
    // a small interface for Figure 2 (graphs.js): look up a neuron, and pin it here
    const posOf = {}; for (let k = 0; k < N.L.length; k++) posOf[`L${N.L[k]}:${N.i[k]}`] = k;
    window.NeuronMap = {
      indexOf: (id) => posOf[id],
      nameOf: (id) => { const k = posOf[id]; return k != null && N.nm[k] >= 0 ? D.names[N.nm[k]] : null; },
      concept: () => cur,
      chanceOf: (lo) => pct(P(lo)), transition,
      unit: (k) => `L${N.L[k]}:${N.i[k]}`,
      show: (id) => { const k = posOf[id]; if (k == null) return; if (onlyLit) { onlyLit = false; $("onlyLit").checked = false; } if (zoom >= 0) setZoom(-1); pinned = hover = k; draw(); describe(k); },
    };
    renderPicks(); geometry();
    setConcept(D.concepts.find((x) => x[0] === D.picks[0]) || D.concepts[0], true);
  }).catch((e) => { $("summary").textContent = "Couldn't load the map data (" + e.message + ")."; });
})();
