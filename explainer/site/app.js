/* Eiffel Tower Qwen: renders every figure from precomputed JSON (data/site.json + data/c/<id>.json). No model runs here. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const ACC = "#8a5a2e", COOL = "#8a9aa5", INK = "#1d2226", BLUE = "#2a5d77";
  const fmtS = (s) => (s === 0 ? "off" : "×" + s);
  const pct = (p) => (p >= 0.995 ? ">99%" : p < 0.005 ? "<1%" : Math.round(p * 100) + "%");
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function highlight(text, words) {
    const safe = esc(text);
    if (!words || !words.length) return safe;
    const re = new RegExp("\\b(" + words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")\\w*", "gi");
    return safe.replace(re, "<mark>$&</mark>");
  }
  // Replies stop at a fixed length. A cut can leave half an emoji (U+FFFD): drop it, and mark an unfinished ending with "…".
  function finish(s) {
    s = s.replace(/\uFFFD+/g, "").replace(/\s+$/, "");
    return s.length < 80 || /[.!?…"'”’)\]*]$/.test(s) ? s : s + " …";   // short answers were never cut
  }

  /* A dial that turns itself up while it is on screen, one step at a time; pauses on any touch of the dial or the
     pause button, and never starts by itself for people who ask for reduced motion. */
  function autoDial(range, playBtn, onStep, opts) {
    const o = Object.assign({ step: 2200, first: null, hold: 3800, onWrap: null }, opts || {});
    const dwell = (v) => (v === 0 && o.first ? o.first : o.step);   // how long to stay on step v before moving on
    let timer = null, visible = false, paused = reduced;
    const max = () => +range.max;
    function set(v) { range.value = v; onStep(+v); }
    function tick() {
      timer = null;
      if (paused || !visible) return;
      const v = +range.value;
      if (v < max()) { set(v + 1); timer = setTimeout(tick, v + 1 === max() ? o.hold : dwell(v + 1)); }
      else { if (o.onWrap) o.onWrap(); set(0); timer = setTimeout(tick, dwell(0)); }
    }
    function schedule() { if (!timer && !paused && visible) timer = setTimeout(tick, dwell(+range.value)); }
    function stop() { if (timer) clearTimeout(timer); timer = null; }
    function show() {
      playBtn.textContent = paused ? "▶" : "❚❚";
      playBtn.setAttribute("aria-label", paused ? "Play: turn the dial automatically" : "Pause");
    }
    function pause() { paused = true; stop(); show(); }
    playBtn.addEventListener("click", () => { paused = !paused; show(); if (paused) stop(); else schedule(); });
    range.addEventListener("pointerdown", pause);
    range.addEventListener("keydown", pause);
    range.addEventListener("input", () => onStep(+range.value));
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((es) => es.forEach((e) => { visible = e.isIntersecting; if (visible) schedule(); else stop(); }),
        { threshold: 0.45 }).observe(range.closest("figure"));
    }
    show();
    return { pause, restart() { stop(); set(0); schedule(); } };
  }

  /* knockout chart: mean P(answer) for top-k neurons off (accent) vs random (grey dashed), with the current k marked */
  function knockChart(svg, ko, h, kNow) {
    const W = 300, P = { l: 36, r: 10, t: 12, b: 34 }, ks = [0].concat(ko.curve.map((c) => c.k));
    const x = (k) => P.l + (ks.indexOf(k) / (ks.length - 1)) * (W - P.l - P.r);
    const y = (p) => P.t + (1 - p) * (h - P.t - P.b);
    const top = [[0, ko.baseline_p_mean]].concat(ko.curve.map((c) => [c.k, c.p_mean]));
    const rnd = [[0, ko.baseline_p_mean]].concat(ko.curve.map((c) => [c.k, c.p_mean_random]));
    let s = "";
    [0, 0.5, 1].forEach((v) => {
      s += `<line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}" stroke="#eceef0"/>`;
      s += `<text x="${P.l - 6}" y="${y(v) + 4}" text-anchor="end" font-size="10">${v * 100}%</text>`;
    });
    if (kNow != null) s += `<line x1="${x(kNow)}" x2="${x(kNow)}" y1="${P.t}" y2="${h - P.b}" stroke="${ACC}" stroke-opacity="0.35" stroke-width="6"/>`;
    ks.forEach((k) => (s += `<text x="${x(k)}" y="${h - P.b + 15}" text-anchor="middle" font-size="10">${k}</text>`));
    s += `<text x="${(P.l + W - P.r) / 2}" y="${h - 3}" text-anchor="middle" font-size="10">neurons switched off</text>`;
    s += `<polyline fill="none" stroke="${COOL}" stroke-width="1.6" stroke-dasharray="3 3" points="${rnd.map((r) => x(r[0]) + "," + y(r[1])).join(" ")}"/>`;
    s += `<polyline fill="none" stroke="${ACC}" stroke-width="2.2" points="${top.map((r) => x(r[0]) + "," + y(r[1])).join(" ")}"/>`;
    top.forEach((r) => (s += `<circle cx="${x(r[0])}" cy="${y(r[1])}" r="2.6" fill="${ACC}"/>`));
    const lastR = rnd[rnd.length - 1], lastT = top[top.length - 1];
    s += `<text x="${W - P.r}" y="${y(lastR[1]) - 6}" text-anchor="end" font-size="10" style="fill:${COOL}">random neurons</text>`;
    s += `<text x="${W - P.r}" y="${Math.min(h - P.b - 4, y(lastT[1]) + 14)}" text-anchor="end" font-size="10" style="fill:${ACC}">top neurons</text>`;
    svg.setAttribute("viewBox", `0 0 ${W} ${h}`);
    svg.innerHTML = s;
  }

  function init(D) {
    /* ---------- Figure 1: the Eiffel dial ---------- */
    const H = D.hero, HW = H.highlight;
    let fi = 0, hs = 0, sizedFor = -1;
    const hn = $("heroNeurons");
    H.units.forEach((u, i) => {
      const d = document.createElement("div"); d.className = "neuron";
      d.innerHTML = `<code>Neuron #${u[1]} · layer ${u[0]}</code>${esc(H.labels[i])}`;
      hn.appendChild(d);
    });
    H.prompts.forEach((q, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `Question ${i + 1} of ${H.prompts.length}: ${q}`; $("heroAll").appendChild(o); });
    $("heroAll").addEventListener("change", (e) => { fi = H.featured.indexOf(+e.target.value); heroR(); });
    $("heroDial").max = H.strengths.length - 1;
    const heroText = (s, q) => { const t = H.reply_only[String(s)][q]; return s ? highlight(finish(t), HW) : esc(finish(t)); };
    // Keep the reply box one height for a whole question, so the page doesn't jump as the dial turns.
    function sizeFor(q) {
      const box = $("heroReply"), probe = box.cloneNode(false);
      probe.removeAttribute("id"); probe.style.cssText = `position:absolute;visibility:hidden;min-height:0;width:${box.clientWidth}px`;
      box.parentNode.appendChild(probe);
      let h = 0;
      H.strengths.forEach((s) => { probe.innerHTML = heroText(s, q); h = Math.max(h, probe.offsetHeight); });
      probe.remove();
      box.style.minHeight = h + "px"; sizedFor = q;
    }
    window.addEventListener("resize", () => { sizedFor = -1; heroR(); });
    function heroR(step) {
      if (step != null) hs = step;
      const s = H.strengths[hs], q = H.featured[fi];
      if (sizedFor !== q) sizeFor(q);
      $("heroAll").value = q;
      $("heroAsk").textContent = H.prompts[q];
      $("heroReply").innerHTML = heroText(s, q);
      $("heroVal").textContent = fmtS(s);
      $("heroState").textContent = H.state[hs] || "";
      const g = Math.min(1, s);
      [...hn.children].forEach((n) => {
        n.style.background = g ? `rgba(138,90,46,${0.14 * g})` : "";
        n.style.borderColor = g ? `rgba(138,90,46,${0.25 + 0.6 * g})` : "";
      });
    }
    autoDial($("heroDial"), $("heroPlay"), heroR, { step: 3200, first: 4800, hold: 5200, onWrap: () => { fi = (fi + 1) % H.featured.length; } });
    $("heroNext").addEventListener("click", () => { fi = (fi + 1) % H.featured.length; heroR(); });
    heroR(0);

    /* ---------- Figure 2: switch it off, every question at once ---------- */
    const K = H.knock, KS = [0].concat(K.curve.map((c) => c.k));
    // rows: only questions the untouched model answers correctly (one says "Burj Khalifa" with nothing switched off)
    const rows = K.questions_plain.map((q, i) => i).filter((i) => /eiff/i.test(K.replies[String(i)].top["0"]));
    let ki = 0, control = false;
    $("offDial").max = KS.length - 1;
    function offR(step) {
      if (step != null) ki = step;
      const k = KS[ki];
      $("offRows").innerHTML = rows.map((i) => {
        const q = K.questions_plain[i];
        const r = K.replies[String(i)];
        const t = k === 0 ? r.top["0"] : (control ? r.random : r.top)[String(k)];
        const p = k === 0 ? K.baseline_p_per_q[i] : (control ? K.curve[ki - 1].p_per_q_random : K.curve[ki - 1].p_per_q)[i];
        const right = /eiffel/i.test(t);
        return `<li><span class="q">${esc(q)}</span><span class="a ${right ? "" : "wrong"}">${highlight(finish(t), HW)}</span><span class="p">${pct(p)}</span></li>`;
      }).join("");
      $("offVal").textContent = k;
      knockChart($("offChart"), K, 210, k);
    }
    autoDial($("offDial"), $("offPlay"), offR, { step: 2600, first: 3900, hold: 5200 });
    $("offControl").addEventListener("change", (e) => { control = e.target.checked; offR(); });
    offR(0);

    /* ---------- Figure 3: any concept ---------- */
    const pick = $("pick"), byGroup = {};
    D.concepts.forEach((c, i) => (byGroup[c.group] = byGroup[c.group] || []).push(i));
    Object.keys(byGroup).sort().forEach((g) => {
      const og = document.createElement("optgroup"); og.label = g;
      byGroup[g].sort((a, b) => D.concepts[a].name.localeCompare(D.concepts[b].name)).forEach((i) => {
        const o = document.createElement("option"); o.value = i; o.textContent = D.concepts[i].name; og.appendChild(o);
      });
      pick.appendChild(og);
    });
    if (D.failed && D.failed.length) {
      const og = document.createElement("optgroup"); og.label = "Didn't work";
      D.failed.forEach((f) => { const o = document.createElement("option"); o.disabled = true; o.textContent = f.name + " (" + f.why + ")"; og.appendChild(o); });
      pick.appendChild(og);
    }
    let ci = Math.max(0, D.concepts.findIndex((c) => c.name === D.default_concept)), oi = 0, as = 0, cur = null, token = 0;
    $("anyDial").max = D.strengths.length - 1;
    const anyAuto = autoDial($("anyDial"), $("anyPlay"), (v) => anyR(v), { step: 2800, first: 4200, hold: 5200, onWrap: () => { oi = (oi + 1) % 8; } });
    $("anyNext").addEventListener("click", () => { oi = (oi + 1) % 8; anyR(); });
    pick.addEventListener("change", (e) => load(+e.target.value));
    $("rand").addEventListener("click", () => { let j; do { j = Math.floor(Math.random() * D.concepts.length); } while (j === ci && D.concepts.length > 1); pick.value = j; load(j); });
    function load(i) {
      ci = i; oi = 0; const t = ++token, c = D.concepts[i];
      $("anyReply").innerHTML = '<span class="loading">Loading…</span>';
      fetch("data/c/" + c.id + ".json").then((r) => r.json()).then((x) => { if (t !== token) return; cur = x; anyFixed(); anyAuto.restart(); })
        .catch(() => { $("anyReply").textContent = "Couldn't load this concept's data."; });
    }
    function anyFixed() {
      const c = D.concepts[ci], ko = cur.knockout;
      const kTxt = ko.k_logodds3 ? `<b>${ko.k_logodds3}</b> neuron${ko.k_logodds3 === 1 ? "" : "s"} off cut the odds of the right answer about 20-fold` : "it takes <b>more than 20</b> neurons off to cut the odds 20-fold";
      $("anyFacts").innerHTML = `Recalled on ${c.recall_kept} of ${c.recall_total} questions · ${kTxt} · steered with ${c.n} neuron${c.n === 1 ? "" : "s"}`;
      knockChart($("anyChart"), ko, 180);
      $("anyWrites").innerHTML = cur.steer.units.map((u, j) =>
        `<li><b>layer ${u[0]} · #${u[1]}</b> pushes towards <code>${cur.steer.writes[j].map((w) => esc(JSON.stringify(w).slice(1, -1))).join("</code> <code>")}</code></li>`).join("");
    }
    function anyR(step) {
      if (step != null) as = step;
      if (!cur) return;
      const s = D.strengths[as], c = D.concepts[ci], q = c.order[oi];
      const txt = s ? cur.steer.replies[String(s)][q] : D.unsteered[q];
      $("anyAsk").textContent = D.prompts[q];
      $("anyReply").innerHTML = s ? highlight(finish(txt), c.highlight) : esc(finish(txt));
      $("anyVal").textContent = fmtS(s);
    }
    pick.value = ci; load(ci);

    /* ---------- Figure 4: shared parts ---------- */
    $("pairs").innerHTML = "<thead><tr><th>Related pair</th><th>Shared specific neurons</th><th></th></tr></thead><tbody>" +
      D.pairs.map((p) => `<tr><td>${esc(p[0])} &amp; ${esc(p[1])}</td><td class="num">${p[3]}</td><td><svg viewBox="0 0 160 10" width="160" height="10" aria-hidden="true"><rect x="0" y="1" width="${p[3] * 18}" height="8" fill="${ACC}" rx="1"/></svg></td></tr>`).join("") +
      `<tr><td><b>Related pairs, average</b></td><td class="num"><b>${D.pairs_mean.related}</b></td><td><svg viewBox="0 0 160 10" width="160" height="10" aria-hidden="true"><rect x="0" y="1" width="${D.pairs_mean.related * 18}" height="8" fill="${INK}" rx="1"/></svg></td></tr>` +
      `<tr><td><b>Random pairs, average</b></td><td class="num"><b>${D.pairs_mean.random}</b></td><td><svg viewBox="0 0 160 10" width="160" height="10" aria-hidden="true"><rect x="0" y="1" width="${Math.max(1.5, D.pairs_mean.random * 18)}" height="8" fill="${COOL}" rx="1"/></svg></td></tr></tbody>`;

    /* ---------- Figure 5: the stacked layers ---------- */
    (function () {
      // bands from the pre-registered layer test (RESULTS.md, H-DEPTH): which kind of neuron is over-represented where
      const bands = [
        { a: 0, b: 12, col: ACC, op: 0.55, title: "Read the question", sub: "words from the question, and what it is about (×2.0–2.3)" },
        { a: 13, b: 21, col: COOL, op: 0.35, title: "General processing", sub: "few concept-specific neurons (×0.2)" },
        { a: 22, b: 32, col: ACC, op: 0.85, title: "The concept's parts", sub: "landmarks, Paris, bridges, gold… (×1.4)" },
        { a: 33, b: 35, col: BLUE, op: 0.85, title: "Say the answer", sub: "shared by every concept (×1.4)" },
      ];
      const L = 36, rowH = 11, top = 26, x0 = 40, dots = 48, dx = 5.6, W = 640, per = 12288 / dots;
      const y = (l) => top + l * rowH;                       // layer 0 at the top: read it downwards, like the text
      const cx = (n) => x0 + 4 + Math.min(dots - 1, Math.floor(n / per)) * dx;
      let s = `<text x="${x0}" y="14" font-size="11">Question goes in ↓ · layer numbers on the left</text>`;
      for (let l = 0; l < L; l++) {
        const b = bands.find((q) => l >= q.a && l <= q.b);
        for (let i = 0; i < dots; i++) s += `<circle cx="${x0 + 4 + i * dx}" cy="${y(l) + 5}" r="1.7" fill="${b.col}" fill-opacity="${b.op}"/>`;
        if (l % 5 === 0 || l === 35) s += `<text x="${x0 - 8}" y="${y(l) + 8}" text-anchor="end" font-size="9.5">${l}</text>`;
      }
      s += `<text x="${x0}" y="${y(L - 1) + rowH + 16}" font-size="11">Answer comes out ↓</text>`;
      const bx = x0 + 4 + dots * dx + 10;
      bands.forEach((q) => {
        const yt = y(q.a) + 1, yb = y(q.b) + rowH - 1, ym = (yt + yb) / 2;
        s += `<path d="M${bx} ${yt} q 7 0 7 7 V ${ym - 5} q 0 5 6 5 q -6 0 -6 5 V ${yb - 7} q 0 7 -7 7" fill="none" stroke="${INK}" stroke-width="1.2"/>`;
        s += `<text x="${bx + 20}" y="${ym - 2}" font-size="13" style="fill:${INK};font-weight:600">${esc(q.title)}</text>`;
        s += `<text x="${bx + 20}" y="${ym + 13}" font-size="11">${esc(q.sub)}</text>`;
      });
      // the six Eiffel Tower neurons from Figure 1, at their real layer and position; the dial grows them
      H.units.forEach((u, i) => {
        s += `<g class="hot" tabindex="0" data-i="${i}"><circle cx="${cx(u[1])}" cy="${y(u[0]) + 5}" r="9" fill="${ACC}" fill-opacity="0.18"/>` +
             `<circle class="core" cx="${cx(u[1])}" cy="${y(u[0]) + 5}" r="3.5" fill="${ACC}" stroke="#fff" stroke-width="1"/></g>`;
      });
      $("layers").setAttribute("viewBox", `0 0 ${W} ${y(L - 1) + rowH + 24}`);
      $("layers").innerHTML = s;
      const hots = [...$("layers").querySelectorAll(".hot")];
      const tipFor = (i) => `Neuron #${H.units[i][1]}, layer ${H.units[i][0]}: ${H.labels[i]}.`;
      const showTip = (i) => ($("layTip").textContent = tipFor(i));
      hots.forEach((g) => {
        const i = +g.dataset.i;
        g.addEventListener("mouseenter", () => showTip(i));
        g.addEventListener("focus", () => showTip(i));
        g.addEventListener("click", () => showTip(i));
      });
      $("layDial").max = H.strengths.length - 1;
      function layR(step) {
        const s2 = H.strengths[step], g = Math.min(1.5, s2) / 1.5;
        $("layVal").textContent = fmtS(s2);
        hots.forEach((h) => {
          h.firstChild.setAttribute("r", 6 + 16 * g);
          h.firstChild.setAttribute("fill-opacity", 0.14 + 0.3 * g);
          h.lastChild.setAttribute("r", 3.5 + 3.5 * g);
        });
        if (!$("layTip").dataset.touched) $("layTip").textContent = s2 ? "The Eiffel Tower neurons, turned up to " + fmtS(s2) + ". Hover over or tap one to see what it does." : "The six Eiffel Tower neurons are the large dots. Hover over or tap one to see what it does.";
      }
      hots.forEach((g) => g.addEventListener("mouseenter", () => ($("layTip").dataset.touched = "1")));
      autoDial($("layDial"), $("layPlay"), layR, { step: 1800, first: 2700, hold: 3600 });
      layR(0);
    })();
  }

  fetch("data/site.json").then((r) => r.json()).then(init).catch((e) => {
    $("heroReply").textContent = "Couldn't load the page's data (" + e.message + "). Try reloading.";
  });

  /* ---------- plain-English definitions on hover or tap, first mention of each term ---------- */
  const TERMS = [
    ["interpretability", "The study of how AI models work inside: what their parts do and why they give the answers they do."],
    ["neurons", "Small units inside the model. For every word-piece it reads or writes, each neuron produces one number: how strongly it is firing. Qwen3-8B has 442,368, in 36 layers."],
    ["superposition", "The theory that a model packs several unrelated concepts into each neuron, so no single neuron means one thing."],
    ["sparse autoencoder", "A second AI model, trained separately, that translates another model's insides into a larger set of more readable \"features\". Golden Gate Claude used one."],
    ["Golden Gate Claude", "A 2024 demo by Anthropic: Claude with its Golden Gate Bridge feature turned up, so it brought the bridge into almost every answer."],
    ["Qwen3-8B", "A free, open AI model from Alibaba with 8 billion parameters. It runs on one high-end home graphics card."],
    ["prosumer graphics card", "A top-end graphics card sold to consumers (here an Nvidia RTX 5090), not a data-centre machine."],
    ["layers", "The model processes text in 36 stages, one after another. Each stage has 12,288 neurons."],
    ["steer", "Changing what a model says by adjusting its insides directly, rather than by changing the question."],
    ["attention", "The part of each layer that lets the model look back at earlier words. We studied only the other part, the neurons."],
    ["word-piece", "Models read and write text in small chunks called tokens: a whole short word, or part of a longer one."],
    ["firing", "How active a neuron is at that moment: a high number means it is strongly firing."],
    ["attribution", "A score for how much one neuron pushed the model towards an answer: its activity times how sensitive the answer is to it."],
    ["activity", "The number a neuron produces at that moment; high means it is firing strongly."],
    ["post-training", "The last stage of building a chat model, where it is trained to be a helpful, careful assistant."],
    ["odds", "Chances written as a ratio: 95% sure is odds of 19 to 1. A 20-fold fall takes that to about 1 to 1, roughly 49%."],
    ["greedy", "The model always picks its single most likely next word, so the same question always gets the same reply."],
  ];
  function addTerms() {
    const done = new Set();
    document.querySelectorAll("[data-copy]").forEach((el) => {
      const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      nodes.forEach((node) => {
        if (node.parentElement.closest("a, .term, summary, h1, .top")) return;
        for (const [term, tip] of TERMS) {
          if (done.has(term)) continue;
          const m = new RegExp("\\b" + term.replace(/[-]/g, "\\-") + "\\b", "i").exec(node.nodeValue);
          if (!m) continue;
          done.add(term);
          const after = node.splitText(m.index);
          after.splitText(m[0].length);
          const span = document.createElement("span");
          span.className = "term"; span.tabIndex = 0; span.setAttribute("data-tip", tip);
          span.textContent = after.nodeValue;
          after.replaceWith(span);
          return;                                   // one term per text node keeps the walk simple
        }
      });
    });
  }
  // Headline slots: "[Eiffel Tower] [Qwen]" - draw the brackets lighter, to read as swappable parts.
  function decorateTitle() {
    const h = document.querySelector("h1[data-copy]");
    if (h && !h.querySelector(".slot")) h.innerHTML = h.innerHTML.replace(/\[([^\]<]+)\]/g, '<span class="slot"><span class="brk">[</span>$1<span class="brk">]</span></span>');
  }
  function plainTitle() {
    const h = document.querySelector("h1[data-copy]");
    if (h) h.querySelectorAll(".slot").forEach((sl) => sl.replaceWith(document.createTextNode(sl.textContent)));
    if (h) h.normalize();
  }
  function removeTerms() {
    plainTitle();
    document.querySelectorAll("[data-copy] .term").forEach((t) => t.replaceWith(document.createTextNode(t.textContent)));
    document.querySelectorAll("[data-copy]").forEach((el) => el.normalize());
  }

  /* ---------- editable prose ----------
     The prose lives in data/copy.json ({key: inline HTML}) and is baked into index.html at build time.
     Only in the private claude.ai editing copy (where the `artifact` capability resolves) does an "Edit text"
     button appear; Save writes data/copy.json back to that artifact. The public site never shows it. */
  const ALLOWED = new Set(["A", "B", "STRONG", "I", "EM", "CODE", "BR", "SUP", "SUB", "SPAN"]);
  function inline(node) {
    let out = "";
    node.childNodes.forEach((n) => {
      if (n.nodeType === 3) { out += esc(n.nodeValue); return; }
      if (n.nodeType !== 1) return;
      const inner = inline(n), tag = n.tagName;
      if (tag === "DIV" || tag === "P") { out += (out ? "<br>" : "") + inner; return; }   // Enter in some browsers
      if (!ALLOWED.has(tag)) { out += inner; return; }
      const t = tag.toLowerCase();
      if (t === "br") { out += "<br>"; return; }
      if (t === "a") {
        const href = n.getAttribute("href") || "";
        out += /^(https?:|#)/.test(href) ? `<a href="${esc(href)}">${inner}</a>` : inner; return;
      }
      out += `<${t}>${inner}</${t}>`;
    });
    return out;
  }
  const blocks = () => [...document.querySelectorAll("[data-copy]")];
  fetch("data/copy.json").then((r) => (r.ok ? r.json() : null)).then((copy) => {
    if (copy) blocks().forEach((el) => { if (copy[el.dataset.copy] != null) el.innerHTML = copy[el.dataset.copy]; });
  }).catch(() => {}).then(() => { addTerms(); decorateTitle(); });

  (async () => {
    if (!window.claude || typeof window.claude.use !== "function") return;
    let art = null;
    try { art = await window.claude.use("artifact"); } catch (e) { art = null; }
    if (!art) return;
    const bar = document.createElement("div");
    bar.className = "editbar";
    bar.innerHTML = '<span class="editmsg" id="editMsg" aria-live="polite"></span><button type="button" class="chip" id="editSave" hidden>Save</button><button type="button" class="chip" id="editToggle">Edit text</button>';
    document.body.appendChild(bar);
    let editing = false;
    const msg = (t) => ($("editMsg").textContent = t);
    function setEditing(on) {
      editing = on;
      if (on) removeTerms(); else { addTerms(); decorateTitle(); }
      document.body.classList.toggle("editing", on);
      document.querySelectorAll("details.method").forEach((d) => { if (on) d.open = true; });
      blocks().forEach((el) => { if (on) el.setAttribute("contenteditable", "true"); else el.removeAttribute("contenteditable"); });
      $("editToggle").textContent = on ? "Done" : "Edit text";
      $("editSave").hidden = !on;
      msg(on ? "Click any text to change it." : "");
    }
    $("editToggle").addEventListener("click", () => setEditing(!editing));
    document.addEventListener("click", (e) => { if (editing && e.target.closest("[data-copy] a, a[data-copy], summary[data-copy]")) e.preventDefault(); }, true);
    document.addEventListener("keydown", (e) => {
      if (editing && e.key === "Enter" && e.target.closest && e.target.closest("[data-copy]")) { e.preventDefault(); document.execCommand("insertLineBreak"); }
      if (editing && e.key === " " && e.target.closest && e.target.closest("summary[data-copy]")) { e.preventDefault(); document.execCommand("insertText", false, " "); }
    });
    document.addEventListener("paste", (e) => {
      if (!editing || !e.target.closest || !e.target.closest("[data-copy]")) return;
      e.preventDefault();
      document.execCommand("insertText", false, (e.clipboardData || window.clipboardData).getData("text/plain"));
    });
    $("editSave").addEventListener("click", async () => {
      const copy = {};
      blocks().forEach((el) => (copy[el.dataset.copy] = inline(el).trim()));
      const body = JSON.stringify(copy, null, 1);
      try { sessionStorage.setItem("copy-draft", body); } catch (e) { /* storage may be blocked */ }
      $("editSave").disabled = true; msg("Saving…");
      try {
        await art.publish({ "data/copy.json": { content: body, contentType: "application/json" } });
        try { sessionStorage.removeItem("copy-draft"); } catch (e) { /* ignore */ }
        msg("Saved. Claude copies this text into the site before each deploy.");
      } catch (err) {
        const code = err && err.code;
        if (code === "conflict") msg("Someone saved a newer version first. The page is reloading to it.");
        else if (code === "not_writer" || code === "not_granted" || code === "capability_disabled" || code === "not_declared") { msg("This copy is read-only for you."); $("editSave").hidden = true; $("editToggle").hidden = true; }
        else if (code === "rate_limited") msg("Saving too often. Wait a moment, then save again.");
        else msg("Couldn't save (" + (code || "error") + "). Your edits are still on the page; try Save again.");
      } finally { $("editSave").disabled = false; }
    });
  })();
})();
