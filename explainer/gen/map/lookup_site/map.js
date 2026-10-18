/* Qwen3-8B map viewer. Static: reads the JSON in data/ (built by explainer/gen/map/build_map.py) and renders one of
   a few views from the URL hash, so every view is a shareable link:
     #/                     the whole model: layers, roles, heads, what each depth does
     #/concepts             browse and filter concepts
     #/concept/<slug>       one concept: its neurons, how it runs through the network, how it lights up
     #/layers  #/layer/<L>  one layer: its mapped neurons and heads
     #/neuron/L<l>:<n>      one neuron: its job, the concepts that need it, every finding about it
     #/heads   #/head/L<l>H<h>
     #/findings  #/about    every claim, and how each source measured it (plus the read-only data API)
   Every claim carries a "how we know" button that shows its source's method, status and reproduce command. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const view = $("view");
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (x, d = 2) => (x == null || isNaN(x) ? "–" : (+x).toFixed(d));
  const sgn = (x, d = 2) => (x == null ? "–" : (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d));
  const int = (x) => (x == null ? "–" : (+x).toLocaleString("en-GB"));
  const pct = (x) => (x == null ? "–" : Math.round(x * 100) + "%");
  const N_LAYERS = 36, N_NEURONS = 12288, N_HEADS = 32;
  const EXPLAINER = window.MAP_EXPLAINER || "../";      // where Eiffel Tower Qwen lives relative to the map

  const ROLE = {
    say: ["says the answer", "The answer is among the 10 words this neuron's output pushes up most (read from its weights)."],
    attribute: ["pushes the answer", "Its output raises the answer as one of its words (answer z ≥ 2 in the weights), without it being the top word."],
    "late-other": ["late, other", "A late neuron (layer ≥ 20) whose direct push does not favour the answer; it matters through other routes."],
    early: ["early", "Layer < 20: no direct word output. Early key neurons work through later neurons (LAYERS_AND_JOBS S2)."],
  };
  const roleTag = (r) => (r ? `<span class="role r-${esc(r)}" title="${esc((ROLE[r] || [])[1] || "")}">${esc((ROLE[r] || [r])[0])}</span>` : `<span class="small">no role (earlier map)</span>`);
  const nLink = (n) => `<a class="mono" href="#/neuron/${esc(n)}">${esc(n)}</a>`;
  const hLink = (h) => `<a class="mono" href="#/head/${esc(h)}">${esc(h)}</a>`;
  const layerOf = (n) => +String(n).slice(1).split(/[:H]/)[0];

  /* ---------------- data ---------------- */
  const cache = {};
  const get = (path) => (cache[path] = cache[path] || fetch("data/" + path).then((r) => { if (!r.ok) throw new Error(path + " " + r.status); return r.json(); }));
  let M, C, CI, LAY, HEADS, FIND;          // meta, concept rows, slug -> row index, layers, heads, findings
  const col = {};
  const row = (slug) => C.rows[CI[slug]];
  const cname = (i) => C.rows[i][col.name];
  const cslug = (i) => C.rows[i][col.slug];
  const srcName = (code) => (M.src_codes_rev[code] || code);

  /* ---------------- provenance popover ---------------- */
  const HOW = [];                         // registry: index -> {src, note}
  function how(src, note) { HOW.push({ src, note }); return `<button type="button" class="how" data-how="${HOW.length - 1}" aria-haspopup="dialog">how we know</button>`; }
  function srcInfo(id) { return M.sources.find((s) => s.id === id) || { title: id, method: "", status: "" }; }
  function docLinks(docs) {
    return (docs || []).map((d) => `<a href="${esc(M.repo)}/blob/main/${esc(d.split("#")[0])}${d.includes("#") ? "#" + esc(d.split("#")[1]) : ""}">${esc(d)}</a>`).join(" · ");
  }
  function openHow(btn) {
    const h = HOW[+btn.dataset.how]; if (!h) return;
    const s = srcInfo(h.src), pop = $("pop");
    const note = h.note && typeof h.note === "object" ? h.note : { text: h.note };
    pop.innerHTML = `<button class="x" aria-label="Close">×</button>
      <h4>${esc(s.title)}</h4>
      <div class="chips"><span class="tag ${/pre-reg/.test(s.status) ? "prereg" : ""}">${esc(note.status || s.status)}</span>${s.date ? `<span class="tag">${esc(s.date)}</span>` : ""}${note.strength ? `<span class="tag ${esc(note.strength.replace(/ /g, "-"))}">${esc(note.strength)}</span>` : ""}</div>
      ${note.text ? `<p><b>This claim:</b> ${esc(note.text)}</p>` : ""}
      <p><b>How it was measured:</b> ${esc(s.method)}</p>
      ${s.reproduce ? `<p><b>Reproduce:</b></p><code class="cmd">${esc(s.reproduce)}</code>` : ""}
      ${(s.caveats || []).length ? `<p><b>Caveats:</b></p><ul>${s.caveats.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>` : ""}
      ${(note.docs || s.docs || []).length ? `<p><b>Write-up:</b> ${docLinks(note.docs || s.docs)}</p>` : ""}`;
    pop.hidden = false;
    const r = btn.getBoundingClientRect(), w = pop.offsetWidth;
    pop.style.left = Math.max(16, Math.min(window.scrollX + r.left, window.scrollX + document.documentElement.clientWidth - w - 16)) + "px";
    pop.style.top = window.scrollY + r.bottom + 6 + "px";
    pop.querySelector(".x").focus();
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest("button.how");
    if (b) { e.preventDefault(); openHow(b); return; }
    if (e.target.closest(".x") || !e.target.closest("#pop")) $("pop").hidden = true;
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { $("pop").hidden = true; hideSugg(); } });

  /* ---------------- hover tips (charts) ---------------- */
  const tip = $("tip");
  document.addEventListener("mousemove", (e) => {
    const t = e.target.closest && e.target.closest("[data-tip]");
    if (!t) { tip.hidden = true; return; }
    tip.innerHTML = t.dataset.tip; tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - w - 8) + "px";
    tip.style.top = (e.clientY + 16 + h > window.innerHeight ? e.clientY - h - 10 : e.clientY + 16) + "px";
  });
  document.addEventListener("click", (e) => {
    const t = e.target.closest && e.target.closest("[data-go]");
    if (t) location.hash = t.dataset.go;
  });

  /* ---------------- findings ---------------- */
  function findingCard(f) {
    const label = f.kind === "label";
    const m = f.metric || {};
    let nums = "";
    if (label) nums = `<span class="tag">${esc(m.pole === "neg" ? "negative pole" : "positive pole")}</span><span class="tag">${esc(m.verdict)}</span>` +
      (m.clean ? `<span class="tag strong" title="Passes the calibrated one-job bar">verified one job</span>` : `<span class="tag weak">not verified as one job</span>`) +
      `<span class="tag">coverage ${fmt(m.coverage)}</span><span class="tag">fits random text ${fmt(m.random_fit)}</span>`;
    else if (f.kind === "web firings") nums = `<span class="tag">about it ${pct(m.about)}</span><span class="tag">related ${pct(m.related)}</span><span class="tag">unrelated ${pct(m.unrelated)}</span>`;
    else if (f.kind === "fact job") nums = m.own != null ? `<span class="tag">own fact ${sgn(m.own)}</span><span class="tag">same entity ${sgn(m.same_entity)}</span><span class="tag">same relation ${sgn(m.same_relation)}</span>` : "";
    else if (f.kind === "entity neuron") nums = `<span class="tag">own facts ${sgn(m.own)}</span>${m.input_about_entity != null ? `<span class="tag">web firings about it ${pct(m.input_about_entity)}</span>` : ""}`;
    else if (m.measure) nums = (m.value != null ? `<span class="tag">${esc(m.measure.replace(/_/g, " "))} ${esc(typeof m.value === "number" ? fmt(m.value) : m.value)}</span>` : "") +
      (m.noise_floor != null ? `<span class="tag">noise floor ${fmt(m.noise_floor)}</span>` : "") + (m.n_passing ? `<span class="tag">${m.n_passing} passing</span>` : "") +
      (m.context ? m.context.split(";").filter((c) => !/^text=/.test(c)).slice(0, 4).map((c) => `<span class="tag k">${esc(c)}</span>`).join("") : "");
    const unit = f.unit && /^L\d+:\d+$/.test(f.unit) ? nLink(f.unit) : f.unit && /^L\d+H\d+$/.test(f.unit) ? hLink(f.unit) : "";
    const concept = f.concept && CI[f.concept] != null ? `<a href="#/concept/${esc(f.concept)}">${esc(cname(CI[f.concept]))}</a>` : "";
    return `<div class="card ${label ? "label" : ""}"><p class="t">${label ? "“" + esc(f.title) + "”" : esc(f.title)}</p>
      ${f.detail ? `<p class="d">${esc(f.detail)}</p>` : ""}
      <div class="meta"><span class="tag k">${esc(f.kind)}</span>${f.strength ? `<span class="tag ${esc(String(f.strength).replace(/ /g, "-"))}">${esc(f.strength)}</span>` : ""}
      ${/pre-reg/.test(f.status || "") ? `<span class="tag prereg">${esc(f.status)}</span>` : ""}${nums}${unit ? `<span class="small">${unit}</span>` : ""}${concept ? `<span class="small">${concept}</span>` : ""}
      ${how(f.source, { text: f.title, status: f.status, strength: f.strength, docs: f.docs })}</div></div>`;
  }

  /* ---------------- search ---------------- */
  const q = $("q"), sugg = $("sugg");
  let suggItems = [], suggSel = -1, tokens = null;
  function hideSugg() { sugg.hidden = true; suggSel = -1; }
  function runSearch(text) {
    const t = text.trim(), lo = t.toLowerCase(), out = [];
    if (!t) return out;
    let m;
    if ((m = /^l?(\d{1,2})\s*[:.\/ ]\s*n?(\d{1,5})$/i.exec(t)) && +m[1] < N_LAYERS && +m[2] < N_NEURONS) out.push({ go: `#/neuron/L${+m[1]}:${+m[2]}`, html: `Neuron <b class="mono">L${+m[1]}:${+m[2]}</b>` });
    if ((m = /^l?(\d{1,2})\s*h\s*(\d{1,2})$/i.exec(t)) && +m[1] < N_LAYERS && +m[2] < N_HEADS) out.push({ go: `#/head/L${+m[1]}H${+m[2]}`, html: `Head <b class="mono">L${+m[1]}H${+m[2]}</b>` });
    if ((m = /^(?:l|layer\s*)(\d{1,2})$/i.exec(t)) && +m[1] < N_LAYERS) out.push({ go: `#/layer/${+m[1]}`, html: `Layer <b>${+m[1]}</b>` });
    const starts = [], has = [];
    for (let i = 0; i < C.rows.length && starts.length < 8; i++) {
      const r = C.rows[i], n = r[col.name].toLowerCase(), a = String(r[col.answer] || "").toLowerCase();
      if (n.startsWith(lo) || a.startsWith(lo)) starts.push(i); else if (has.length < 8 && (n.includes(lo) || a.includes(lo))) has.push(i);
    }
    starts.concat(has).slice(0, 8).forEach((i) => out.push({ go: `#/concept/${cslug(i)}`, html: `${esc(cname(i))}<small>${esc(C.rows[i][col.status] === "mapped" ? "concept" : C.rows[i][col.status])}</small>` }));
    if (tokens && lo.length >= 3) {
      const hits = (tokens[lo] || tokens[" " + lo] || []).slice(0, 6);
      hits.forEach((n) => out.push({ go: `#/neuron/${n}`, html: `<b class="mono">${esc(n)}</b><small>pushes “${esc(t)}”</small>` }));
    }
    return out;
  }
  function renderSugg() {
    suggItems = runSearch(q.value);
    if (!suggItems.length) { hideSugg(); return; }
    sugg.innerHTML = suggItems.map((s, i) => `<li role="option" data-i="${i}" aria-selected="${i === suggSel}">${s.html}</li>`).join("");
    sugg.hidden = false;
  }
  q.addEventListener("input", () => { suggSel = -1; if (!tokens && q.value.length >= 3) get("tokens.json").then((t) => { tokens = t; renderSugg(); }); renderSugg(); });
  q.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); if (!suggItems.length) return; suggSel = (suggSel + (e.key === "ArrowDown" ? 1 : -1) + suggItems.length) % suggItems.length; renderSugg(); }
    else if (e.key === "Enter") { const s = suggItems[Math.max(0, suggSel)]; if (s) { location.hash = s.go; q.value = ""; hideSugg(); q.blur(); } }
  });
  sugg.addEventListener("mousedown", (e) => { const li = e.target.closest("li"); if (li) { location.hash = suggItems[+li.dataset.i].go; q.value = ""; hideSugg(); } });
  q.addEventListener("blur", () => setTimeout(hideSugg, 150));

  /* ---------------- shared chart helpers ---------------- */
  const roleColor = (r) => `var(--${{ say: "say", attribute: "attr", "late-other": "late", early: "early" }[r] || "late"})`;
  function bandsFor(L) { return (LAY.bands || []).filter((b) => { const [a, z] = b.unit.split(":")[1].split("-").map(Number); return L >= a && L <= (z == null || isNaN(z) ? a : z); }); }

  /* ================= views ================= */
  const VIEWS = {};

  VIEWS.home = function () {
    const c = M.counts;
    const pre = FIND.model.filter((f) => /pre-reg/.test(f.status));
    view.innerHTML = `
      <p class="crumb">An open, growing map · Qwen3-8B · 36 layers × 12,288 MLP neurons, 36 × 32 attention heads</p>
      <h1>What Qwen3-8B's neurons do</h1>
      <p class="dek2">For thousands of concepts, which of the model's 442,368 neurons it needs to name them, what those neurons push when they fire, and how each concept travels through the network, with every claim traceable to how it was measured.</p>
      <div class="stats">
        <div class="stat"><b>${int(c.mapped)}</b><span>concepts mapped (of ${int(c.concepts)} tried)</span></div>
        <div class="stat"><b>${int(c.neurons_in_maps)}</b><span>neurons in at least one concept's top 30</span></div>
        <div class="stat"><b>${int(c.findings)}</b><span>findings about neurons, heads and layers</span></div>
        <div class="stat"><b>${M.sources.length}</b><span>sources, each with its method and status</span></div>
      </div>
      <p class="lede">How to read it: a concept's <b>map</b> is the 30 neurons whose removal most lowers the model's odds of giving the right name, each tested one at a time against random neurons. A neuron's <b>job</b> comes from what it pushes (its weights), what makes it fire (web text), and causal tests on facts. Start from <a href="#/concept/eiffel-tower">the Eiffel Tower</a>, <a href="#" id="randc">a random concept</a>, or a layer below.</p>
      <figure class="fig"><h3 class="fh">The whole model, layer by layer</h3>
        <p class="lead">Each row is a layer, input at the top. The bar counts how often a layer's neurons are among a concept's 5 most important, coloured by what those neurons do; the grid shows the layer's 32 attention heads, darker where knocking a head out hurts more concepts. Click a row for the layer, a cell for a head.</p>
        <div class="legend">${Object.keys(ROLE).map((r) => roleTag(r)).join("")}<span>· heads: share of mapped concepts where the head's knockout costs ≥ ${M.strong_head} log-odds</span></div>
        <div class="scroll">${modelChart()}</div>
        <figcaption>Roles are read from each neuron's weights in run 1; for maps made earlier they are inferred from the same neuron's role in run 1 (grey where it never appeared there). ${how("run1", "Top-5 slots per layer and head knockouts aggregated over every mapped concept's primary map.")}</figcaption>
      </figure>
      <h2>What each depth does</h2>
      <p class="lede">Graded claims from the project's synthesis: <span class="tag strong">strong</span> means several independent methods agree.</p>
      <div class="cards">${(LAY.bands || []).map(findingCard).join("")}</div>
      <h2>Tested in advance</h2>
      <p class="lede">Predictions written down before the data existed, and how they came out.</p>
      <div class="cards">${pre.map(findingCard).join("")}</div>
      <h2>Other model-wide findings</h2>
      <div class="cards">${FIND.model.filter((f) => !/pre-reg/.test(f.status)).map(findingCard).join("")}</div>
      <h2>Where the map comes from</h2>
      <table class="data"><tr><th>Source</th><th>Kind</th><th>Status</th><th class="num">Maps</th><th class="num">Findings</th><th></th></tr>
      ${M.sources.map((s) => `<tr><td>${esc(s.title)}</td><td>${esc(s.kind)}</td><td>${esc(s.status)}</td><td class="num">${int(s.n_maps)}</td><td class="num">${int(s.n_findings)}</td><td>${how(s.id)}</td></tr>`).join("")}</table>
      <p class="small">New experiments add a source or a findings file and the map is rebuilt; see <a href="#/about">How we know</a>.</p>`;
    $("randc").addEventListener("click", (e) => { e.preventDefault(); randomConcept(); });
  };

  function randomConcept() {
    const mapped = C.rows.filter((r) => r[col.status] === "mapped");
    location.hash = "#/concept/" + mapped[Math.floor(Math.random() * mapped.length)][col.slug];
  }

  function modelChart() {
    const rowH = 15, top = 22, W = 1000, labW = 64, bandW = 70, barX = labW + bandW + 8, barW = 360, hx = barX + barW + 40, cell = 11;
    const L = LAY.layers, maxSlots = Math.max(...L.map((l) => l.top5_slots)) || 1, mapped = M.counts.mapped || 1;
    const H = top + rowH * N_LAYERS + 26;
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Layers of Qwen3-8B with key-neuron counts by role and attention-head importance">`;
    s += `<text x="${labW - 8}" y="${top - 8}" text-anchor="end">layer</text><text x="${labW + 4}" y="${top - 8}">depth bands</text><text x="${barX}" y="${top - 8}">key neurons (top-5 slots), by role</text><text x="${hx}" y="${top - 8}">heads 0–31</text>`;
    // bands (strong solid, moderate lighter), stacked in lanes
    const bands = LAY.bands || [];
    bands.forEach((b, i) => {
      const [a, z] = b.unit.split(":")[1].split("-").map(Number), lane = i % 5, x = labW + 4 + lane * 13;
      const y0 = top + a * rowH + 1, h = ((isNaN(z) ? a : z) - a + 1) * rowH - 2;
      const op = b.strength === "strong" ? 0.85 : b.strength === "moderate" ? 0.45 : 0.2;
      s += `<rect class="hot" data-go="#/layer/${a}" x="${x}" y="${y0}" width="9" height="${h}" rx="2" fill="var(--accent)" fill-opacity="${op}" data-tip="<b>${esc(b.id)} · ${esc(b.strength || "")}</b><br>${esc(b.title)}"/>`;
    });
    L.forEach((l) => {
      const y = top + l.layer * rowH;
      s += `<g class="hot" data-go="#/layer/${l.layer}" data-tip="<b>Layer ${l.layer}</b><br>${l.top5_slots} top-5 slots · ${int(l.neurons_in_maps)} neurons in maps · ${l.hubs} hubs${l.attr_share != null ? "<br>" + pct(l.attr_share) + " of attribution (mean)" : ""}${bandsFor(l.layer).map((b) => "<br>" + esc(b.id) + ": " + esc(b.title)).join("")}">`;
      s += `<rect x="0" y="${y}" width="${hx - 12}" height="${rowH}" fill="transparent"/>`;
      s += `<text x="${labW - 8}" y="${y + rowH - 4}" text-anchor="end">${l.layer}</text>`;
      let x = barX;
      ["say", "attribute", "late-other", "early"].forEach((r) => {
        const v = (l.top5_roles || {})[r] || 0; if (!v) return;
        const w = (v / maxSlots) * barW; s += `<rect x="${x}" y="${y + 2}" width="${w}" height="${rowH - 4}" fill="${roleColor(r)}"/>`; x += w;
      });
      const other = l.top5_slots - Object.values(l.top5_roles || {}).reduce((a, b) => a + b, 0);
      if (other > 0) { const w = (other / maxSlots) * barW; s += `<rect x="${x}" y="${y + 2}" width="${w}" height="${rowH - 4}" fill="var(--rule)"/>`; }
      s += `</g>`;
      for (let h = 0; h < N_HEADS; h++) {
        const v = (l.heads || {})[h]; const strong = v ? v[1] : 0, n = v ? v[0] : 0;
        const op = strong ? Math.min(1, 0.12 + Math.sqrt(strong / mapped) * 2.2) : n ? 0.06 : 0;
        s += `<rect class="hot" data-go="#/head/L${l.layer}H${h}" x="${hx + h * cell}" y="${y + 1.5}" width="${cell - 1.5}" height="${rowH - 3}" fill="${op ? "var(--accent)" : "var(--wash)"}" fill-opacity="${op || 1}" data-tip="<b>L${l.layer}H${h}</b><br>${n ? `in the top-50 heads of ${int(n)} concepts; knockout ≥ ${M.strong_head} in ${int(strong)}${v[2] != null ? "; median " + sgn(v[2]) : ""}` : "not among any concept's 50 most-attributed heads"}"/>`;
      }
    });
    s += `<text x="${labW - 8}" y="${H - 6}" text-anchor="end">output</text></svg>`;
    return s;
  }

  /* ---------------- concepts list ---------------- */
  VIEWS.concepts = function (args, params) {
    const sections = [...new Set(C.rows.map((r) => r[col.section]).filter(Boolean))].sort();
    view.innerHTML = `<p class="crumb"><a href="#/">Map</a> › Concepts</p><h1>Concepts</h1>
      <p class="lede">Every concept the loop tried. “Not recalled” means Qwen3-8B didn't reliably name it, which is a result too. k is the fewest neurons whose removal cuts the answer's odds at least 20-fold.</p>
      <div class="filters"><input id="cf" type="search" placeholder="Filter by name or answer" value="${esc(params.get("q") || "")}">
        <select id="cs"><option value="">All sections</option>${sections.map((s) => `<option${params.get("section") === s ? " selected" : ""}>${esc(s)}</option>`).join("")}</select>
        <select id="ck"><option value="mapped">Mapped</option><option value="">All</option><option value="not recalled">Not recalled</option><option value="twice">Measured twice</option></select>
        <select id="co"><option value="name">A–Z</option><option value="k">Fewest neurons</option><option value="kd">Most neurons</option></select></div>
      <p class="count" id="cn"></p><div class="scroll"><table class="data" id="ct"></table></div><p class="more"><button class="chip" id="cm" hidden>Show more</button></p>`;
    let limit = 200;
    const draw = () => {
      const f = $("cf").value.trim().toLowerCase(), sec = $("cs").value, k = $("ck").value, o = $("co").value;
      let rows = C.rows.filter((r) => (!f || r[col.name].toLowerCase().includes(f) || String(r[col.answer] || "").toLowerCase().includes(f)) &&
        (!sec || r[col.section] === sec) && (k === "" || (k === "twice" ? r[col.sources].length > 1 : r[col.status] === k)));
      if (o !== "name") rows = rows.slice().sort((a, b) => ((a[col.k_mean] ?? 99) - (b[col.k_mean] ?? 99)) * (o === "k" ? 1 : -1));
      $("cn").textContent = `${int(rows.length)} concepts`;
      $("ct").innerHTML = `<tr><th>Concept</th><th>Qwen answers</th><th>Section</th><th>Type</th><th class="num">k</th><th>Top neuron</th><th>Measured by</th></tr>` +
        rows.slice(0, limit).map((r) => `<tr class="${r[col.status] === "mapped" ? "" : "dim"}"><td><a href="#/concept/${esc(r[col.slug])}">${esc(r[col.name])}</a></td><td>${esc(r[col.answer] || "")}</td><td>${esc(r[col.section] || "")}</td><td>${esc(r[col.type] || "")}</td>
          <td class="num">${r[col.k_mean] ?? (r[col.status] === "mapped" ? "–" : esc(r[col.status]))}</td><td>${r[col.top1] ? nLink(r[col.top1]) : ""}</td><td>${r[col.sources].split("").map((c) => `<span class="tag">${esc(srcName(c))}</span>`).join(" ")}</td></tr>`).join("");
      $("cm").hidden = rows.length <= limit;
    };
    ["cf", "cs", "ck", "co"].forEach((id) => $(id).addEventListener("input", () => { limit = 200; draw(); }));
    $("cm").addEventListener("click", () => { limit += 400; draw(); });
    draw();
  };

  /* ---------------- one concept ---------------- */
  VIEWS.concept = async function (args) {
    const slug = args[0], i = CI[slug];
    if (i == null) { view.innerHTML = `<h1>Not found</h1><p>No concept “${esc(slug)}”. <a href="#/concepts">Browse concepts</a>.</p>`; return; }
    const d = col.shard != null ? (await get(`cs/${C.rows[i][col.shard]}.json`))[slug] : await get(`c/${slug}.json`);
    let mi = 0;
    const meta = d.meta || {};
    const chips = [meta.section && `<a class="tag" href="#/concepts?section=${encodeURIComponent(meta.section)}">${esc(meta.section)}</a>`, meta.category && `<span class="tag">${esc(meta.category)}</span>`,
      meta.type && `<span class="tag">${esc(meta.type)}</span>`, meta.kind && `<span class="tag" title="named entity or common noun">${esc(meta.kind)}</span>`,
      meta.tier && `<span class="tag" title="Wikipedia language editions (fame)">fame tier ${esc(meta.tier)}</span>`,
      meta.qid && `<a class="tag" href="https://www.wikidata.org/wiki/${esc(meta.qid)}">${esc(meta.qid)}</a>`].filter(Boolean).join("");
    const draw = () => {
      const m = d.maps[mi];
      HOW.length = 0;
      const mapped = m.status === "mapped";
      const tabs = d.maps.length > 1 ? `<div class="tabs" role="tablist">${d.maps.map((x, j) => `<button class="chip" role="tab" aria-pressed="${j === mi}" data-mi="${j}">${esc(x.source === "run1" ? "Run 1" : x.batch || x.source)}${x.status !== "mapped" ? " (" + esc(x.status) + ")" : ""}</button>`).join("")}</div>` : "";
      let h = `<p class="crumb"><a href="#/">Map</a> › <a href="#/concepts">Concepts</a></p><h1>${esc(d.name)}</h1>
        <p class="dek2">${mapped ? `Asked to name it, Qwen3-8B answers “${esc(m.answer || d.answer)}”${m.recall != null ? ` on ${m.recall} of the questions it was given` : ""}.` : `Qwen3-8B did not reliably name this concept, so it has no map (${esc(m.status)}).`}</p>
        <div class="chips">${chips}${d.explainer ? `<a class="tag" href="${esc(EXPLAINER)}#steer=${encodeURIComponent(d.explainer)}">Steer it in Eiffel Tower Qwen →</a>` : ""}</div>${tabs}`;
      if (mapped) {
        const noise = m.noise;
        h += `<div class="stats">
          <div class="stat"><b>${m.k_mean ?? "–"}</b><span>neurons to cut the answer's odds 20-fold</span></div>
          <div class="stat"><b>${m.k_flip ?? "–"}</b><span>neurons to flip most answers</span></div>
          <div class="stat"><b>${sgn(m.top && m.top[0][1])}</b><span>log-odds lost when the top neuron alone is off</span></div>
          <div class="stat"><b>${noise != null ? "±" + fmt(noise) : "–"}</b><span>noise floor: random neurons switched off</span></div></div>
          <p class="small">${how(m.source, { text: `${d.name}: k = ${m.k_mean}, baseline log-odds ${fmt(m.baseline)}` })} ${m.prov ? `Measured on ${esc(m.prov.gpu)} · code ${esc((m.prov.code_commit || "").slice(0, 8))} · ${esc(m.prov.started_utc || "")}` : m.file ? `From <code>${esc(m.file)}</code>` : ""}</p>`;
        h += `<figure class="fig"><h3 class="fh">How the concept runs through the network</h3>
          <p class="lead">Left to right is depth. Dots are the concept's 30 key neurons, placed by layer and by how much the answer suffers when each is switched off alone; the shaded band is the noise floor. ${m.lens ? "The line above tracks the answer through the residual stream (logit lens): its rank among 151,936 tokens if the model had to answer from that layer." : ""} Squares below are attention heads, darker when knocking them out costs more.</p>
          <div class="legend">${Object.keys(ROLE).map(roleTag).join("")}</div>
          ${flowChart(m)}<figcaption>Click a dot or square to open it. ${m.lens ? how("run1", "Logit lens: the saved residual stream at the answer position, per question and depth, through the final norm and unembedding; median rank across questions.") : ""}</figcaption></figure>`;
        if (m.light) h += `<figure class="fig"><h3 class="fh">What lights up</h3><p class="lead">All 442,368 neurons at once: rows are layers (input at the top), columns are blocks of 128 neurons. Brightness is attribution, how much each block's activity pushes the answer, before anything is switched off; outlined blocks hold the 30 key neurons. Bars on the right are each layer's share.</p>
          <div class="two2" style="grid-template-columns:minmax(0,5fr) minmax(0,1fr)"><canvas class="heat" id="heat" width="${m.light.bins * 6}" height="${N_LAYERS * 8}" aria-label="Attribution heat map, 36 layers by 96 blocks"></canvas>${shareBars(m.light.layer_share)}</div>
          <figcaption>Attribution is a first-order estimate; the dots above are the measured effects. ${how("run1", "Attribution = activation × gradient of the answer's log-odds, summed over positions and averaged over questions; max |value| per block, square-root scaled.")}</figcaption></figure>`;
        if (m.drops && m.ks) h += `<figure class="fig"><h3 class="fh">Switching neurons off, cumulatively</h3><p class="lead">The concept's top 1, 2, 3… neurons switched off together, against the same number of random neurons from the same layers. The dashed line is a 20-fold cut in the odds (3 log-odds).</p>${kChart(m)}</figure>`;
        h += `<h2>The ${m.top.length} key neurons</h2><div class="scroll"><table class="data"><tr><th>#</th><th>Neuron</th><th>What it does</th><th class="num">Drop alone</th><th>What it pushes (from its weights)</th></tr>
          ${m.top.map((t, r) => `<tr class="${noise != null && t[1] < 2 * noise ? "dim" : ""}"><td>${r + 1}</td><td>${nLink(t[0])}${t[8] ? ' <span class="tag" title="In the top 10 of more than 1% of concepts: shared machinery">hub</span>' : ""}</td><td>${roleTag(t[4])}${t[9] ? ' <span class="small" title="This map predates weights-based roles; role and words are this neuron\'s majority from the maps that measured them.">inferred</span>' : ""}</td>
            <td class="num">${sgn(t[1])} <span class="bar" style="width:${Math.min(80, Math.max(1, (t[1] / Math.max(m.top[0][1], 0.01)) * 80))}px"></span></td>
            <td><div class="jobs">${(t[7] || []).slice(0, 8).map((w) => `<code class="${isAns(w, m.answer || d.answer) ? "ans" : ""}">${esc(w)}</code>`).join("")}</div></td></tr>`).join("")}</table></div>
          <p class="small">Drop alone: fall in the answer's log-odds when that neuron is set to zero at every position, averaged over the questions. Rows in grey are within twice the noise floor. ${how(m.source)}</p>`;
        if (m.heads && m.heads.length) h += `<h3>Attention heads that matter most</h3><div class="chips">${m.heads.slice(0, 12).map(([hh, v]) => `<a class="tag k" href="#/head/${esc(hh)}">${esc(hh)} ${sgn(v)}</a>`).join("")}</div>`;
      }
      if (m.prompts && m.prompts.length) h += `<h3>Questions it was asked</h3><ol class="qlist">${m.prompts.map((p) => `<li>${esc(p)}</li>`).join("")}</ol>`;
      if (d.replication) {
        const r = d.replication;
        h += `<h2>Measured twice</h2><div class="note-box">This concept was mapped by two runs with independently written questions. Their top-10 lists share <b>${r.top10_overlap}</b> neurons${r.top1_same ? ", including the top one" : ""}; ${r.top5_in_other_top30} of the first run's top 5 are in the other's top 30. ${r.shared.length ? "Shared: " + r.shared.map(nLink).join(", ") + "." : ""} For comparison, prompt-set changes typically keep about 5 of 10.</div>`;
      }
      if (d.related && d.related.length) h += `<h2>Concepts that share its neurons</h2><p class="small">Counting shared top-10 neurons, ignoring hubs (neurons in the top 10 of more than ${M.hub_cut} concepts).</p><div class="chips">${d.related.map(([s, k, via]) => CI[s] != null ? `<a class="tag" href="#/concept/${esc(s)}" title="via ${esc(via.join(", "))}">${esc(cname(CI[s]))} · ${k}</a>` : "").join("")}</div>`;
      if (d.findings && d.findings.length) h += `<h2>Other findings about this concept's neurons</h2><div class="cards">${d.findings.slice(0, 30).map(findingCard).join("")}</div>`;
      view.innerHTML = h;
      view.querySelectorAll("[data-mi]").forEach((b) => b.addEventListener("click", () => { mi = +b.dataset.mi; draw(); }));
      if (mapped && m.light) heatCanvas($("heat"), m.light, m.top);
    };
    draw();
  };
  const isAns = (w, a) => { const x = String(w).trim().toLowerCase(); return x.length >= 2 && String(a || "").toLowerCase().includes(x); };

  function flowChart(m) {
    const W = 1000, padL = 46, padR = 14, plotW = W - padL - padR, x = (L) => padL + ((L + 1) / (N_LAYERS + 1)) * plotW;
    const lensH = m.lens ? 90 : 0, dotsTop = lensH + 18, dotsH = 230, headsTop = dotsTop + dotsH + 26, headsH = 34, H = headsTop + headsH + 30;
    const maxD = Math.max(...m.top.map((t) => t[1]), 0.5);
    const y = (v) => dotsTop + dotsH - (Math.sqrt(Math.max(0, v)) / Math.sqrt(maxD)) * dotsH;
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Key neurons and heads by layer, and the answer's rank through the residual stream">`;
    for (let L = 0; L < N_LAYERS; L += 5) s += `<line class="grid" x1="${x(L)}" x2="${x(L)}" y1="${lensH ? 6 : dotsTop}" y2="${headsTop + headsH}"/><text x="${x(L)}" y="${H - 8}" text-anchor="middle">${L}</text>`;
    s += `<text x="${padL}" y="${H - 8}" text-anchor="end">layer</text>`;
    if (m.lens) {
      const r = m.lens.rank, ly = (v) => 8 + (Math.log10(Math.max(1, v)) / Math.log10(151936)) * (lensH - 16);
      s += `<text x="${padL - 6}" y="12" text-anchor="end">rank 1</text><text x="${padL - 6}" y="${lensH - 6}" text-anchor="end">150k</text>`;
      let path = ""; r.forEach((v, j) => { path += (j ? "L" : "M") + x(j - 1).toFixed(1) + " " + ly(v).toFixed(1); });
      s += `<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2"/>`;
      const first = r.findIndex((v) => v <= 10);
      r.forEach((v, j) => { s += `<circle cx="${x(j - 1)}" cy="${ly(v)}" r="6" fill="transparent" data-tip="${j ? "After layer " + (j - 1) : "Embeddings"}: answer's first token ranks <b>${int(v)}</b>; P = ${fmt(m.lens.p[j], 3)}"/>`; });
      if (first >= 0) s += `<text x="${x(first - 1) + 6}" y="${ly(r[first]) + 14}" style="fill:var(--accent)">top 10 from layer ${first - 1}</text>`;
    }
    if (m.noise != null) s += `<rect class="noise" x="${padL}" y="${y(m.noise)}" width="${plotW}" height="${dotsTop + dotsH - y(m.noise)}"/>`;
    s += `<line class="axis" x1="${padL}" x2="${W - padR}" y1="${dotsTop + dotsH}" y2="${dotsTop + dotsH}"/>`;
    [0.1, 0.5, 1, 2, 4, 8].filter((v) => v <= maxD).forEach((v) => { s += `<text x="${padL - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text><line class="grid" x1="${padL}" x2="${W - padR}" y1="${y(v)}" y2="${y(v)}"/>`; });
    s += `<text x="8" y="${dotsTop + 8}" transform="rotate(-90 8 ${dotsTop + 8})" text-anchor="end">drop (log-odds)</text>`;
    m.top.slice().reverse().forEach((t) => {
      const L = layerOf(t[0]), rr = 4 + Math.sqrt(Math.max(0, t[1]) / maxD) * 7, jitter = ((+t[0].split(":")[1] % 7) - 3) * 2.2;
      s += `<circle class="hot" data-go="#/neuron/${esc(t[0])}" cx="${x(L) + jitter}" cy="${y(Math.max(0, t[1]))}" r="${rr}" fill="${roleColor(t[4])}" fill-opacity="0.85" stroke="${t[8] ? "var(--ink)" : "none"}" stroke-dasharray="${t[8] ? "2 2" : ""}"
        data-tip="<b class='mono'>${esc(t[0])}</b> · ${esc((ROLE[t[4]] || ["?"])[0])}<br>drop ${sgn(t[1])} log-odds${(t[7] || []).length ? "<br>pushes: <code>" + esc(t[7].slice(0, 5).join(" · ")) + "</code>" : ""}${t[8] ? "<br>hub (shared machinery)" : ""}"/>`;
    });
    s += `<text x="${padL - 6}" y="${headsTop + headsH / 2 + 4}" text-anchor="end">heads</text>`;
    const hd = (m.heads || []).filter((v) => v[1] > 0).slice(0, 30), hmax = Math.max(...hd.map((v) => v[1]), 0.5);
    hd.forEach(([hh, v], j) => {
      const L = layerOf(hh), stack = hd.slice(0, j).filter((o) => layerOf(o[0]) === L).length;
      s += `<rect class="hot" data-go="#/head/${esc(hh)}" x="${x(L) - 5}" y="${headsTop + (stack % 3) * 11}" width="10" height="10" fill="var(--accent)" fill-opacity="${0.15 + 0.85 * Math.min(1, v / hmax)}" data-tip="<b class='mono'>${esc(hh)}</b><br>knocked out: ${sgn(v)} log-odds"/>`;
    });
    return s + "</svg>";
  }

  function kChart(m) {
    const W = 460, H = 200, pl = 40, pr = 10, pt = 10, pb = 28, ks = m.ks, n = ks.length;
    const maxV = Math.max(3.5, ...m.drops, ...(m.random_drops || []));
    const x = (j) => pl + (j / (n - 1)) * (W - pl - pr), y = (v) => pt + (1 - Math.max(0, v) / maxV) * (H - pt - pb);
    const line = (arr, c, dash) => `<path d="${arr.map((v, j) => (j ? "L" : "M") + x(j) + " " + y(v)).join("")}" fill="none" stroke="${c}" stroke-width="2" ${dash ? 'stroke-dasharray="4 3"' : ""}/>` + arr.map((v, j) => `<circle cx="${x(j)}" cy="${y(v)}" r="3.5" fill="${c}" data-tip="${ks[j]} neurons off: ${sgn(v)} log-odds"/>`).join("");
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" style="max-width:520px">`;
    s += `<line class="grid" x1="${pl}" x2="${W - pr}" y1="${y(3)}" y2="${y(3)}"/><text x="${W - pr}" y="${y(3) - 4}" text-anchor="end">20× lower odds</text>`;
    ks.forEach((k, j) => { s += `<text x="${x(j)}" y="${H - 8}" text-anchor="middle">${k}</text>`; });
    s += `<text x="${pl - 6}" y="${y(0)}" text-anchor="end">0</text><text x="${pl - 6}" y="${y(maxV) + 8}" text-anchor="end">${fmt(maxV, 1)}</text>`;
    s += line(m.drops, "var(--accent)") + (m.random_drops ? line(m.random_drops, "var(--cool)", true) : "");
    s += `<text x="${x(n - 1)}" y="${y(m.drops[n - 1]) - 8}" text-anchor="end" style="fill:var(--accent)">top neurons</text>`;
    return s + `<text x="${pl}" y="${H - 8}" text-anchor="end">k</text></svg><div class="legend"><span style="color:var(--accent)">● the concept's top neurons</span><span>● random neurons, same layers</span></div>`;
  }

  function shareBars(share) {
    const W = 120, rowH = 8, H = N_LAYERS * rowH, mx = Math.max(...share);
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:100%;max-height:${H * 1.6}px">${share.map((v, L) => `<rect x="0" y="${L * rowH + 1}" width="${(v / mx) * (W - 4)}" height="${rowH - 2}" fill="var(--accent)" fill-opacity=".7" data-tip="Layer ${L}: ${pct(v)} of attribution"/>`).join("")}</svg>`;
  }

  function heatCanvas(cv, light, top) {
    const ctx = cv.getContext("2d"), bins = light.bins, cw = cv.width / bins, ch = cv.height / N_LAYERS;
    const css = getComputedStyle(document.documentElement);
    const rgb = css.getPropertyValue("--heat").trim() || "138,90,46", ink = css.getPropertyValue("--ink").trim() || "#1d2226";
    ctx.clearRect(0, 0, cv.width, cv.height);
    // every block's max picks up some background, so measure brightness above the concept's median block
    const vals = Array.from(light.strip, (c) => parseInt(c, 16) / 15), med = vals.slice().sort((a, b) => a - b)[vals.length >> 1];
    for (let L = 0; L < N_LAYERS; L++) for (let b = 0; b < bins; b++) {
      const v = Math.max(0, (vals[L * bins + b] - med) / (1 - med + 1e-9)); if (v < 0.02) continue;
      ctx.fillStyle = `rgba(${rgb},${Math.min(1, 0.1 + 0.9 * Math.pow(v, 0.8)).toFixed(3)})`; ctx.fillRect(b * cw, L * ch, cw, ch);
    }
    // the concept's key neurons (measured), outlined on top of the attribution picture
    ctx.strokeStyle = ink; ctx.lineWidth = 1.5;
    (top || []).forEach((t) => { const [L, n] = t[0].slice(1).split(":").map(Number); ctx.strokeRect(Math.floor(n / 128) * cw + 0.75, L * ch + 0.75, cw - 1.5, ch - 1.5); });
    const pos = (e) => { const r = cv.getBoundingClientRect(); return [Math.floor(((e.clientX - r.left) / r.width) * bins), Math.floor(((e.clientY - r.top) / r.height) * N_LAYERS)]; };
    cv.addEventListener("mousemove", (e) => { const [b, L] = pos(e); cv.dataset.tip = `Layer ${L}, neurons ${b * 128}–${b * 128 + 127}: ${pct(parseInt(light.strip[L * bins + b] || "0", 16) / 15)} of the brightest block (sqrt scale). Click for the layer.`; });
    cv.addEventListener("click", (e) => { const [, L] = pos(e); location.hash = "#/layer/" + L; });
  }

  /* ---------------- layers ---------------- */
  VIEWS.layers = function () {
    view.innerHTML = `<p class="crumb"><a href="#/">Map</a> › Layers</p><h1>Layers</h1><p class="lede">Qwen3-8B has 36 layers; each adds an attention step and an MLP of 12,288 neurons to a running “residual stream”. Pick a layer.</p>
      <div class="scroll">${modelChart()}</div>`;
  };
  VIEWS.layer = async function (args) {
    const L = +args[0];
    if (!(L >= 0 && L < N_LAYERS)) { view.innerHTML = "<h1>No such layer</h1>"; return; }
    const neurons = await get(`n/L${L}.json`), l = LAY.layers[L];
    const ids = Object.keys(neurons).sort((a, b) => neurons[b].n_uses - neurons[a].n_uses || neurons[b].findings.length - neurons[a].findings.length);
    const hs = Object.entries(l.heads || {}).sort((a, b) => b[1][1] - a[1][1]);
    let limit = 100;
    view.innerHTML = `<p class="crumb"><a href="#/">Map</a> › <a href="#/layers">Layers</a></p><h1>Layer ${L}</h1>
      <p class="dek2">${L < 20 ? "Early half: neurons here act on the input's own tokens and reach the output only through later layers." : L >= 28 ? "Late layers: neurons here push readable words straight into the output." : "Middle layers: a mix of relations, answer-format machinery and composition."}</p>
      <div class="stats"><div class="stat"><b>${int(l.neurons_in_maps)}</b><span>neurons in some concept's top 30</span></div><div class="stat"><b>${int(l.top5_slots)}</b><span>top-5 slots across concepts</span></div><div class="stat"><b>${l.hubs}</b><span>hubs (shared machinery)</span></div><div class="stat"><b>${l.attr_share != null ? pct(l.attr_share) : "–"}</b><span>of a concept's attribution, on average</span></div></div>
      ${bandsFor(L).length ? `<h2>What this depth does</h2><div class="cards">${bandsFor(L).map(findingCard).join("")}</div>` : ""}
      <figure class="fig"><h3 class="fh">Where the mapped neurons are</h3><p class="lead">All 12,288 neurons of layer ${L} in blocks of 64; bar height counts concepts that use a neuron in the block.</p>${layerStrip(neurons)}</figure>
      <h2>Attention heads</h2><div class="chips">${hs.length ? hs.map(([h, v]) => `<a class="tag k" href="#/head/L${L}H${h}" title="top-50 of ${v[0]} concepts; strong in ${v[1]}">L${L}H${h} · ${v[1]}</a>`).join("") : '<span class="small">No head in this layer is among any concept\'s 50 most-attributed heads.</span>'}</div>
      <h2>Neurons in maps</h2><div class="scroll"><table class="data" id="lt"></table></div><p class="more"><button class="chip" id="lm">Show more</button></p>
      <div class="layernav">${L > 0 ? `<a href="#/layer/${L - 1}">← Layer ${L - 1}</a>` : "<span></span>"}${L < N_LAYERS - 1 ? `<a href="#/layer/${L + 1}">Layer ${L + 1} →</a>` : ""}</div>`;
    const draw = () => {
      $("lt").innerHTML = `<tr><th>Neuron</th><th>Role</th><th class="num">Concepts</th><th>Used by</th><th>Pushes</th><th>Findings</th></tr>` + ids.slice(0, limit).map((n) => {
        const v = neurons[n], id = `L${L}:${n}`;
        return `<tr><td>${nLink(id)}${v.hub ? ' <span class="tag">hub</span>' : ""}</td><td>${v.role ? roleTag(v.role) : ""}</td><td class="num">${v.n_uses}</td>
          <td>${v.uses.slice(0, 3).map((u) => `<a href="#/concept/${esc(cslug(u[0]))}">${esc(cname(u[0]))}</a>`).join(", ")}${v.n_uses > 3 ? "…" : ""}</td>
          <td><div class="jobs">${(v.jobs || []).slice(0, 5).map((w) => `<code>${esc(w)}</code>`).join("")}</div></td><td>${v.findings.length ? `<span class="tag">${v.findings.length}</span>` : ""}</td></tr>`;
      }).join("");
      $("lm").hidden = ids.length <= limit;
    };
    $("lm").addEventListener("click", () => { limit += 300; draw(); });
    draw();
  };
  function layerStrip(neurons) {
    const bins = 192, per = N_NEURONS / bins, cnt = new Array(bins).fill(0), hub = new Array(bins).fill(0);
    Object.entries(neurons).forEach(([n, v]) => { const b = Math.floor(+n / per); cnt[b] += v.n_uses; if (v.hub) hub[b]++; });
    const mx = Math.max(...cnt, 1), W = 1000, H = 90, bw = W / bins;
    return `<svg class="chart" viewBox="0 0 ${W} ${H + 16}">${cnt.map((c, b) => c ? `<rect x="${b * bw}" y="${H - (Math.sqrt(c / mx)) * H}" width="${bw - 0.6}" height="${(Math.sqrt(c / mx)) * H}" fill="${hub[b] ? "var(--ink)" : "var(--accent)"}" fill-opacity=".8" data-tip="neurons ${b * per}–${b * per + per - 1}: used ${c} times${hub[b] ? " · contains a hub" : ""}"/>` : "").join("")}
      <text x="0" y="${H + 13}">0</text><text x="${W}" y="${H + 13}" text-anchor="end">12,287</text></svg>`;
  }

  /* ---------------- one neuron ---------------- */
  VIEWS.neuron = async function (args) {
    const id = args[0], m = /^L(\d+):(\d+)$/.exec(id || "");
    if (!m) { view.innerHTML = "<h1>Not a neuron id</h1><p>Use the form L31:11943.</p>"; return; }
    const L = +m[1], n = m[2], all = await get(`n/L${L}.json`), v = all[n];
    const head = `<p class="crumb"><a href="#/">Map</a> › <a href="#/layer/${L}">Layer ${L}</a></p><h1><code>${esc(id)}</code></h1>`;
    if (!v) { view.innerHTML = head + `<p class="dek2">Neuron ${esc(n)} of layer ${L}. It isn't in any concept's map yet and has no recorded findings.</p><p class="lede">${L < 20 ? "Early neurons rarely show up in answer-ranked maps (selection favours late layers), so absence here says little." : "Most neurons are not among any mapped concept's top 30."}</p>`; return; }
    const labels = v.findings.filter((f) => f.kind === "label"), other = v.findings.filter((f) => f.kind !== "label");
    const secCount = {};
    v.uses.forEach((u) => { const s = C.rows[u[0]][col.section] || "other"; secCount[s] = (secCount[s] || 0) + 1; });
    const total = v.uses.length;
    view.innerHTML = head + `<p class="dek2">Neuron ${esc(n)} of 12,288 in layer ${L} of 36${v.role ? " · " + esc((ROLE[v.role] || [v.role])[0]) : ""}. Used by ${int(v.n_uses)} concept map${v.n_uses === 1 ? "" : "s"}${v.hub ? "; a hub (shared answer machinery, in the top 10 of many concepts)" : ""}.</p>
      ${bandsFor(L).length ? `<div class="chips">${bandsFor(L).map((b) => `<span class="tag ${esc((b.strength || "").replace(/ /g, "-"))}" title="${esc(b.title)}">${esc(b.id)}: ${esc(b.title.slice(0, 60))}${b.title.length > 60 ? "…" : ""}</span>`).join("")}</div>` : ""}
      <h2>Its job</h2>
      ${v.jobs ? `<h3>What it pushes when it fires (from its weights)</h3><div class="jobs">${v.jobs.map((w) => `<code>${esc(w)}</code>`).join("")}</div>
        <p class="small">The words its output vector raises most through the final norm and unembedding, in the sign it takes when the concept is asked. Roles across concepts: ${Object.entries(v.roles).map(([r, k]) => roleTag(r) + " ×" + k).join(" ")}. ${L < 28 ? "For neurons before layer 28 this reading is weak: they act through later neurons (S1, S2)." : ""} ${how("run1", "Weights-based role and pushed words (map_core.roles).")}</p>` : `<p class="small">No weights-based reading yet (it appears only in maps made before loop v2).</p>`}
      ${labels.length ? `<h3>What makes it fire in web text (blind labels)</h3><div class="cards">${labels.map(findingCard).join("")}</div>` : ""}
      ${other.length ? `<h3>Causal tests and other findings</h3><div class="cards">${other.map(findingCard).join("")}</div>` : ""}
      <h2>Concepts that need it</h2>
      <div class="legend">${Object.entries(secCount).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([s, k]) => `<span>${esc(s)} ${Math.round((100 * k) / total)}%</span>`).join("")}</div>
      <div class="scroll"><table class="data"><tr><th>Concept</th><th class="num">Rank</th><th class="num">Drop alone</th><th class="num">vs noise</th><th>Source</th></tr>
      ${v.uses.slice().sort((a, b) => (b[3] || 0) - (a[3] || 0)).slice(0, 300).map((u) => {
        const r = C.rows[u[0]], noise = r[col.noise];
        return `<tr><td><a href="#/concept/${esc(r[col.slug])}">${esc(r[col.name])}</a></td><td class="num">${u[2]}</td><td class="num">${sgn(u[3])}</td><td class="num">${noise ? "×" + fmt(u[3] / noise, 0) : "–"}</td><td><span class="tag">${esc(srcName(String(u[1])))}</span></td></tr>`;
      }).join("")}</table></div>
      ${v.n_uses > 300 ? `<p class="small">Showing the 300 strongest of ${v.n_uses}.</p>` : ""}
      <p class="small">Drop alone: fall in that concept's answer log-odds with this neuron zeroed. “vs noise” divides by that concept's noise floor (random same-layer neurons, run 1 only). ${how("run1")}</p>`;
  };

  /* ---------------- heads ---------------- */
  VIEWS.heads = function () {
    const curated = FIND.heads;
    view.innerHTML = `<p class="crumb"><a href="#/">Map</a> › Heads</p><h1>Attention heads</h1>
      <p class="lede">Attention is the only part of the model that moves information between words. For each concept, its 50 most-attributed heads were knocked out one at a time; this grid counts in how many concepts each head's knockout cost at least ${M.strong_head} log-odds.</p>
      <div class="scroll">${headGrid()}</div>
      ${curated.length ? `<h2>Heads with a known role</h2><div class="cards">${curated.map(findingCard).join("")}</div>` : ""}
      <h2>Most often important</h2><div class="scroll"><table class="data"><tr><th>Head</th><th class="num">Concepts (top 50)</th><th class="num">Knockout ≥ ${M.strong_head}</th><th class="num">Median</th><th class="num">Max</th></tr>
      ${Object.entries(HEADS).sort((a, b) => b[1].strong - a[1].strong).slice(0, 40).map(([h, v]) => `<tr><td>${hLink(h)}</td><td class="num">${int(v.n)}</td><td class="num">${int(v.strong)}</td><td class="num">${sgn(v.median)}</td><td class="num">${sgn(v.max)}</td></tr>`).join("")}</table></div>`;
  };
  function headGrid() {
    const cell = 22, pl = 30, pt = 18, W = pl + N_HEADS * cell + 4, H = pt + N_LAYERS * 14 + 8, mx = Math.max(...Object.values(HEADS).map((v) => v.strong), 1);
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" style="max-width:${W * 1.4}px">`;
    for (let h = 0; h < N_HEADS; h += 4) s += `<text x="${pl + h * cell + cell / 2}" y="12" text-anchor="middle">${h}</text>`;
    for (let L = 0; L < N_LAYERS; L++) {
      s += `<text x="${pl - 6}" y="${pt + L * 14 + 11}" text-anchor="end">${L}</text>`;
      for (let h = 0; h < N_HEADS; h++) {
        const id = `L${L}H${h}`, v = HEADS[id], op = v && v.strong ? 0.12 + 0.88 * Math.sqrt(v.strong / mx) : 0;
        const f = FIND.heads.some((x) => x.unit === id);
        s += `<rect class="hot" data-go="#/head/${id}" x="${pl + h * cell}" y="${pt + L * 14}" width="${cell - 2}" height="12" fill="${op ? "var(--accent)" : "var(--wash)"}" fill-opacity="${op || 1}" stroke="${f ? "var(--ink)" : "none"}" data-tip="<b>${id}</b>${v ? `<br>top-50 in ${int(v.n)} concepts; knockout ≥ ${M.strong_head} in ${int(v.strong)}` : "<br>never among the 50 most-attributed"}${f ? "<br>has a written-up role" : ""}"/>`;
      }
    }
    return s + "</svg>";
  }
  VIEWS.head = function (args) {
    const id = args[0], m = /^L(\d+)H(\d+)$/.exec(id || "");
    if (!m) { view.innerHTML = "<h1>Not a head id</h1><p>Use the form L34H28.</p>"; return; }
    const v = HEADS[id];
    view.innerHTML = `<p class="crumb"><a href="#/">Map</a> › <a href="#/heads">Heads</a> › <a href="#/layer/${m[1]}">Layer ${m[1]}</a></p><h1><code>${esc(id)}</code></h1>
      <p class="dek2">Head ${m[2]} of 32 in layer ${m[1]}.${v && v.n ? ` Among the 50 most-attributed heads of ${int(v.n)} concepts; knocking it out costs ≥ ${M.strong_head} log-odds in ${int(v.strong)} of them (median ${sgn(v.median)}).` : " Not among any concept's 50 most-attributed heads."}</p>
      ${v && v.findings && v.findings.length ? `<h2>Known role</h2><div class="cards">${v.findings.map(findingCard).join("")}</div>` : ""}
      ${v && v.top.length ? `<h2>Concepts that lean on it most</h2><div class="scroll"><table class="data"><tr><th>Concept</th><th class="num">Knockout drop</th></tr>${v.top.map(([ci, d]) => `<tr><td><a href="#/concept/${esc(cslug(ci))}">${esc(cname(ci))}</a></td><td class="num">${sgn(d)}</td></tr>`).join("")}</table></div><p class="small">${how("run1", "Each of a concept's 50 most-attributed heads zeroed alone (its o_proj input slice); mean drop in the answer's log-odds.")}</p>` : ""}`;
  };

  /* ---------------- findings ---------------- */
  VIEWS.findings = async function () {
    view.innerHTML = `<p class="crumb"><a href="#/">Map</a> › Findings</p><h1>Findings</h1><p class="lede">Every claim on the map, from every source. Filter by kind or search the text; each links to its unit and to how it was measured.</p>
      <div class="filters"><input id="ff" type="search" placeholder="Search findings (e.g. Paris, relation, century)"><select id="fk"><option value="">All kinds</option></select><select id="fs"><option value="">All sources</option>${M.sources.map((s) => `<option value="${esc(s.id)}">${esc(s.title)}</option>`).join("")}</select></div>
      <p class="count" id="fn">Loading…</p><div class="cards" id="fl"></div><p class="more"><button class="chip" id="fm" hidden>Show more</button></p>`;
    const text = await fetch("data/findings.jsonl").then((r) => r.text());
    const all = text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
    const kinds = [...new Set(all.map((f) => f.kind))].sort();
    $("fk").innerHTML += kinds.map((k) => `<option>${esc(k)}</option>`).join("");
    let limit = 60;
    const draw = () => {
      HOW.length = 0;
      const t = $("ff").value.trim().toLowerCase(), k = $("fk").value, s = $("fs").value;
      const rows = all.filter((f) => (!k || f.kind === k) && (!s || f.source === s) && (!t || (f.title + " " + (f.detail || "") + " " + f.unit).toLowerCase().includes(t)));
      $("fn").textContent = `${int(rows.length)} findings`;
      $("fl").innerHTML = rows.slice(0, limit).map(findingCard).join("");
      $("fm").hidden = rows.length <= limit;
    };
    ["ff", "fk", "fs"].forEach((id) => $(id).addEventListener("input", () => { limit = 60; draw(); }));
    $("fm").addEventListener("click", () => { limit += 120; draw(); });
    draw();
  };

  /* ---------------- about: method, sources, API ---------------- */
  VIEWS.about = function () {
    const base = location.href.split("#")[0].replace(/index\.html$/, "") + "data/";
    view.innerHTML = `<p class="crumb"><a href="#/">Map</a> › How we know</p><h1>How we know</h1>
      <p class="lede">The map is assembled from saved results; nothing is computed in your browser except layout. Each source below says what it measured, how, its status (pre-registered or exploratory), how to reproduce it, and its caveats. Every “how we know” button on the map opens the relevant one.</p>
      ${M.sources.map((s) => `<h2>${esc(s.title)}</h2><div class="chips"><span class="tag ${/pre-reg/.test(s.status) ? "prereg" : ""}">${esc(s.status)}</span><span class="tag">${esc(s.kind)}</span><span class="tag">${esc(s.date || "")}</span><span class="tag">${int(s.n_maps)} maps · ${int(s.n_findings)} findings</span></div>
        <p class="lede">${esc(s.method)}</p><dl class="kv"><dt>Reproduce</dt><dd><code>${esc(s.reproduce || "")}</code></dd>${(s.docs || []).length ? `<dt>Write-up</dt><dd>${docLinks(s.docs)}</dd>` : ""}${(s.caveats || []).length ? `<dt>Caveats</dt><dd>${s.caveats.map(esc).join("<br>")}</dd>` : ""}</dl>`).join("")}
      <h2>Read the data yourself (people and agents)</h2>
      <p class="lede">Everything behind the map is plain JSON at stable paths next to this page. No key, no rate limit; please cache. Ids: neurons <code>L&lt;layer&gt;:&lt;index&gt;</code>, heads <code>L&lt;layer&gt;H&lt;head&gt;</code>, concepts by slug.</p>
<pre class="api">GET ${esc(base)}meta.json          model, counts, sources (method, status, reproduce, caveats)
GET ${esc(base)}concepts.json      {cols, rows}: every concept with status, k, top neuron, sources
GET ${esc(base)}${col.shard != null ? "cs/&lt;shard&gt;.json    concepts by slug (shard = concepts.json 'shard' column); each" : "c/&lt;slug&gt;.json      one concept"}: maps (top-30 neurons, drops, roles, pushed words,
                                   heads, k curves, null, logit lens, attribution strip), related, findings
GET ${esc(base)}n/L&lt;layer&gt;.json    every mapped neuron in a layer, keyed by index:
                                   uses [[concept row, source, rank, drop]], role, pushed words, findings
GET ${esc(base)}heads.json         every head seen in a map: counts, median drop, top concepts, findings
GET ${esc(base)}layers.json        per-layer aggregates and the depth-band findings
GET ${esc(base)}findings.jsonl     every finding, one JSON object per line
GET ${esc(base)}tokens.json        pushed word -> neurons (say / attribute neurons)

# e.g. what does L31:11943 do, and which concepts need it?
curl -s ${esc(base)}n/L31.json | jq '."11943" | {role, jobs, n_uses, findings: [.findings[].title]}'</pre>
      <h2>Adding to the map</h2>
      <p class="lede">The map is meant to grow. A new experiment either adds a source module (<code>explainer/gen/map/sources/</code>, emitting concept maps or findings with its own method text) or appends findings to a JSONL file in <code>explainer/gen/map/findings/</code>: one line per claim with <code>unit</code>, <code>kind</code>, <code>title</code>, <code>detail</code>, <code>strength</code>, <code>status</code> and <code>docs</code>. Rebuild with <code>python -I -X utf8 explainer/gen/map/build_map.py</code>. See <code>MAP.md</code> in the repository.</p>`;
  };

  /* ---------------- router ---------------- */
  async function route() {
    $("pop").hidden = true; tip.hidden = true; HOW.length = 0;
    const raw = location.hash.replace(/^#\/?/, ""), [path, qs] = raw.split("?"), parts = path.split("/").filter(Boolean).map(decodeURIComponent);
    const params = new URLSearchParams(qs || "");
    const name = parts[0] || "home", fn = VIEWS[name];
    document.querySelectorAll(".mnav a").forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#/" + (name === "concept" ? "concepts" : name === "neuron" || name === "layer" ? "layers" : name === "head" ? "heads" : name)));
    try {
      if (!fn) { view.innerHTML = `<h1>Not found</h1><p><a href="#/">Back to the map</a></p>`; return; }
      await fn(parts.slice(1), params);
      const t = view.querySelector("h1");
      document.title = (t && name !== "home" ? t.textContent + " · " : "") + "Qwen3-8B Map";
      window.scrollTo(0, 0);
    } catch (e) {
      console.error(e);
      view.innerHTML = `<h1>Couldn't load this view</h1><p class="lede">${esc(e.message)}</p><p><a href="#/">Back to the map</a></p>`;
    }
  }

  Promise.all([get("meta.json"), get("concepts.json"), get("layers.json"), get("heads.json"), get("findings.json")]).then(([m, c, l, h, f]) => {
    M = m; C = c; LAY = l; HEADS = h; FIND = f;
    C.cols.forEach((k, i) => (col[k] = i));
    CI = {}; C.rows.forEach((r, i) => (CI[r[col.slug]] = i));
    M.src_codes_rev = {}; Object.entries(M.source_codes || {}).forEach(([k, v]) => (M.src_codes_rev[String(v)] = k === "run1" ? "run 1" : k));
    $("built").textContent = `Built ${M.built_utc.slice(0, 10)} from code ${String(M.code_commit || "").slice(0, 8)}.`;
    window.addEventListener("hashchange", route);
    route();
  }).catch((e) => { view.innerHTML = `<h1>Couldn't load the map data</h1><p class="lede">${esc(e.message)}. Build it with <code>python -I -X utf8 explainer/gen/map/build_map.py</code>.</p>`; });
})();
