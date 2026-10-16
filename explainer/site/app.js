/* Eiffel Tower Qwen: renders every figure from precomputed JSON (data/site.json + data/c/<id>.json). No model runs here. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const ACC = "#8a5a2e", COOL = "#8a9aa5", INK = "#1d2226";
  const fmtS = (s) => (s === 0 ? "off" : "×" + s);
  const pct = (p) => (p >= 0.995 ? ">99%" : p < 0.005 ? "<1%" : Math.round(p * 100) + "%");

  function highlight(text, words) {
    const safe = esc(text);
    if (!words || !words.length) return safe;
    const re = new RegExp("\\b(" + words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")\\w*", "gi");
    return safe.replace(re, "<mark>$&</mark>");
  }
  // A reply cut at the token limit can end in half an emoji, which decodes as U+FFFD: show it as a visible marker.
  const clean = (s) => s.replace(/�+$/, "▯");
  function chips(el, labels, onPick) {
    el.innerHTML = "";
    labels.forEach((t, i) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip"; b.textContent = t;
      b.addEventListener("click", () => onPick(i));
      el.appendChild(b);
    });
    return (i) => [...el.children].forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
  }

  /* knockout chart: mean P(answer) for top-k neurons off (accent) vs random (grey dashed) */
  function knockChart(svg, ko, h) {
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
    ks.forEach((k) => (s += `<text x="${x(k)}" y="${h - P.b + 15}" text-anchor="middle" font-size="10">${k}</text>`));
    s += `<text x="${(P.l + W - P.r) / 2}" y="${h - 3}" text-anchor="middle" font-size="10">neurons switched off</text>`;
    s += `<polyline fill="none" stroke="${COOL}" stroke-width="1.6" stroke-dasharray="3 3" points="${rnd.map((r) => x(r[0]) + "," + y(r[1])).join(" ")}"/>`;
    s += `<polyline fill="none" stroke="${ACC}" stroke-width="2.2" points="${top.map((r) => x(r[0]) + "," + y(r[1])).join(" ")}"/>`;
    top.forEach((r) => (s += `<circle cx="${x(r[0])}" cy="${y(r[1])}" r="2.6" fill="${ACC}"/>`));
    const lastR = rnd[rnd.length - 1], lastT = top[top.length - 1];
    s += `<text x="${W - P.r}" y="${y(lastR[1]) - 6}" text-anchor="end" font-size="10" style="fill:${COOL}">random</text>`;
    s += `<text x="${W - P.r}" y="${Math.min(h - P.b - 4, y(lastT[1]) + 14)}" text-anchor="end" font-size="10" style="fill:${ACC}">top neurons</text>`;
    svg.setAttribute("viewBox", `0 0 ${W} ${h}`);
    svg.innerHTML = s;
  }

  function init(D) {
    /* ---------- Figure 1: the Eiffel dial ---------- */
    const H = D.hero, HW = H.highlight;
    let hq = H.featured[0], hs = 0, promptToo = false;
    const heroPress = chips($("heroTabs"), H.featured_labels, (i) => { hq = H.featured[i]; $("heroAll").value = hq; heroR(); });
    H.prompts.forEach((q, i) => { const o = document.createElement("option"); o.value = i; o.textContent = (i + 1) + ". " + q; $("heroAll").appendChild(o); });
    $("heroAll").addEventListener("change", (e) => { hq = +e.target.value; heroR(); });
    $("heroPrompt").addEventListener("change", (e) => { promptToo = e.target.checked; heroR(); });
    const hn = $("heroNeurons");
    H.units.forEach((u, i) => {
      const d = document.createElement("div"); d.className = "neuron";
      d.innerHTML = `<code>layer ${u[0]} · #${u[1]}</code>${esc(H.labels[i])}`;
      hn.appendChild(d);
    });
    $("heroDial").max = H.strengths.length - 1;
    $("heroDial").addEventListener("input", (e) => { hs = +e.target.value; heroR(); });
    function heroR() {
      const s = H.strengths[hs], grid = promptToo ? H.prompt_too : H.reply_only;
      heroPress(H.featured.indexOf(hq));
      $("heroAsk").textContent = H.prompts[hq];
      $("heroReply").innerHTML = (s ? highlight(clean(grid[String(s)][hq]), HW) : esc(clean(grid["0"][hq]))) + ' <span class="cap">[stops at 120 tokens]</span>';
      $("heroVal").textContent = fmtS(s);
      $("heroState").textContent = (promptToo ? H.state_prompt : H.state)[hs] || "";
      const g = Math.min(1, s);
      [...hn.children].forEach((n) => {
        n.style.background = g ? `rgba(138,90,46,${0.14 * g})` : "";
        n.style.borderColor = g ? `rgba(138,90,46,${0.25 + 0.6 * g})` : "";
      });
    }
    $("heroAll").value = hq; heroR();

    /* ---------- Figure 2: switch it off ---------- */
    const K = H.knock, KS = [0].concat(K.curve.map((c) => c.k));
    let kq = 0, ki = 0;
    const offPress = chips($("offTabs"), K.questions.map((_, i) => "Question " + (i + 1)), (i) => { kq = i; offR(); });
    $("offDial").max = KS.length - 1;
    $("offDial").addEventListener("input", (e) => { ki = +e.target.value; offR(); });
    knockChart($("offChart"), K, 210);
    function offR() {
      const k = KS[ki], r = K.replies[String(kq)];
      const pTop = k ? K.curve[ki - 1].p_per_q[kq] : K.baseline_p_per_q[kq];
      const pRnd = k ? K.curve[ki - 1].p_per_q_random[kq] : K.baseline_p_per_q[kq];
      offPress(kq);
      $("offAsk").textContent = K.questions_plain[kq];
      $("offReply").innerHTML = highlight(clean(r.top[String(k)]), HW) + `<span class="cap">\nChance of "${esc(K.answer)}": ${pct(pTop)}</span>`;
      $("offRand").innerHTML = highlight(clean(k ? r.random[String(k)] : r.top["0"]), HW) + `<span class="cap">\nChance of "${esc(K.answer)}": ${pct(pRnd)}</span>`;
      $("offVal").textContent = k;
    }
    offR();

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
    let ci = Math.max(0, D.concepts.findIndex((c) => c.name === D.default_concept)), aq = 0, as = 0, cur = null, token = 0;
    const anyPress = chips($("anyTabs"), D.prompt_labels, (i) => { aq = i; anyR(); });
    $("anyDial").max = D.strengths.length - 1;
    $("anyDial").addEventListener("input", (e) => { as = +e.target.value; anyR(); });
    pick.addEventListener("change", (e) => load(+e.target.value));
    $("rand").addEventListener("click", () => { let j; do { j = Math.floor(Math.random() * D.concepts.length); } while (j === ci && D.concepts.length > 1); pick.value = j; load(j); });
    function load(i) {
      ci = i; const t = ++token, c = D.concepts[i];
      $("anyReply").innerHTML = '<span class="loading">Loading…</span>';
      fetch("data/c/" + c.id + ".json").then((r) => r.json()).then((x) => { if (t !== token) return; cur = x; anyFixed(); anyR(); })
        .catch(() => { $("anyReply").textContent = "Couldn't load this concept's data."; });
    }
    function anyFixed() {
      const c = D.concepts[ci], ko = cur.knockout;
      const kTxt = ko.k_logodds3 ? `<b>${ko.k_logodds3}</b> neuron${ko.k_logodds3 === 1 ? "" : "s"} off cut the odds of the right answer about 20-fold` : "<b>more than 20</b> neurons needed to cut the odds 20-fold";
      $("anyFacts").innerHTML = `Recalled on ${c.recall_kept} of ${c.recall_total} questions · ${kTxt} · steering: ${c.n} neuron${c.n === 1 ? "" : "s"}, chosen by rule ${c.rule} (${esc(D.rules[c.rule])})`;
      knockChart($("anyChart"), ko, 180);
      $("anyWrites").innerHTML = cur.steer.units.map((u, j) =>
        `<li><b>layer ${u[0]} · #${u[1]}</b> pushes towards <code>${cur.steer.writes[j].slice(0, 6).map((w) => esc(JSON.stringify(w).slice(1, -1))).join("</code> <code>")}</code></li>`).join("");
    }
    function anyR() {
      if (!cur) return;
      const s = D.strengths[as], c = D.concepts[ci];
      anyPress(aq);
      const txt = s ? cur.steer.replies[String(s)][aq] : D.unsteered[aq];
      $("anyReply").innerHTML = `<span class="cap">${esc(D.prompts[aq])}\n\n</span>` + (s ? highlight(clean(txt), c.highlight) : esc(clean(txt))) + ' <span class="cap">[stops at 100 tokens]</span>';
      $("anyVal").textContent = fmtS(s);
    }
    pick.value = ci; load(ci);

    /* ---------- Figure 4: shared parts ---------- */
    $("pairs").innerHTML = "<thead><tr><th>Related pair</th><th>Shared specific neurons</th><th></th></tr></thead><tbody>" +
      D.pairs.map((p) => `<tr><td>${esc(p[0])} &amp; ${esc(p[1])}</td><td class="num">${p[3]}</td><td><svg viewBox="0 0 160 10" width="160" height="10" aria-hidden="true"><rect x="0" y="1" width="${p[3] * 18}" height="8" fill="${ACC}" rx="1"/></svg></td></tr>`).join("") +
      `<tr><td><b>Related pairs, average</b></td><td class="num"><b>${D.pairs_mean.related}</b></td><td><svg viewBox="0 0 160 10" width="160" height="10" aria-hidden="true"><rect x="0" y="1" width="${D.pairs_mean.related * 18}" height="8" fill="${INK}" rx="1"/></svg></td></tr>` +
      `<tr><td><b>Random pairs, average</b></td><td class="num"><b>${D.pairs_mean.random}</b></td><td><svg viewBox="0 0 160 10" width="160" height="10" aria-hidden="true"><rect x="0" y="1" width="${Math.max(1.5, D.pairs_mean.random * 18)}" height="8" fill="${COOL}" rx="1"/></svg></td></tr></tbody>`;

    /* ---------- Figure 5: layers ---------- */
    (function () {
      const W = 860, L = 36, x = (l) => 40 + (l * (W - 60)) / L, cw = (W - 60) / L - 2;
      const bands = [[0, 5], [6, 12], [13, 21], [22, 32], [33, 35]];
      const rows = [["Specific to the concept", ACC, [0.6, 2.3, 0.2, 1.4, 0.3]], ["General “say the answer”", "#2a5d77", [0.8, 0.3, 1.4, 0.9, 1.4]], ["Words from the question", COOL, [2.1, 2.0, 0.5, 1.1, 0.3]]];
      let s = "";
      for (let l = 0; l < L; l++) s += `<rect x="${x(l)}" y="8" width="${cw}" height="8" fill="#e2e5e8"/>`;
      s += `<text x="${x(0)}" y="31" font-size="11">layer 0</text><text x="${x(35) + cw}" y="31" font-size="11" text-anchor="end">35</text>`;
      [["recognise", 6, 12], ["concept parts", 22, 32], ["say it", 33, 35]].forEach(([t, a, b]) => (s += `<text x="${(x(a) + x(b) + cw) / 2}" y="31" font-size="12" text-anchor="middle" style="fill:${INK};font-weight:600">${t}</text>`));
      rows.forEach(([name, col, v], r) => {
        const y0 = 54 + r * 34;
        s += `<text x="${x(0)}" y="${y0 - 5}" font-size="11">${name}</text>`;
        bands.forEach(([a, b], i) => {
          const hgt = Math.min(1, v[i] / 2.4);
          s += `<rect x="${x(a)}" y="${y0}" width="${x(b) + cw - x(a)}" height="16" fill="${col}" opacity="${0.12 + 0.88 * hgt}" rx="2"/>`;
          s += `<text x="${(x(a) + x(b) + cw) / 2}" y="${y0 + 12}" font-size="11" text-anchor="middle" style="fill:${hgt > 0.5 ? "#fff" : INK}">×${v[i]}</text>`;
        });
      });
      $("layers").innerHTML = s;
    })();

    /* ---------- timeline ---------- */
    $("timeline").innerHTML = D.timeline.map((t) => `<li><time>${esc(t[0])}</time>${esc(t[1])}</li>`).join("");
  }

  fetch("data/site.json").then((r) => r.json()).then(init).catch((e) => {
    $("heroReply").textContent = "Couldn't load the page's data (" + e.message + "). Try reloading.";
  });

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
  }).catch(() => {});

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
      document.body.classList.toggle("editing", on);
      blocks().forEach((el) => { if (on) el.setAttribute("contenteditable", "true"); else el.removeAttribute("contenteditable"); });
      $("editToggle").textContent = on ? "Done" : "Edit text";
      $("editSave").hidden = !on;
      msg(on ? "Click any text to change it." : "");
    }
    $("editToggle").addEventListener("click", () => setEditing(!editing));
    document.addEventListener("click", (e) => { if (editing && e.target.closest("[data-copy] a")) e.preventDefault(); }, true);
    document.addEventListener("keydown", (e) => {
      if (editing && e.key === "Enter" && e.target.closest && e.target.closest("[data-copy]")) { e.preventDefault(); document.execCommand("insertLineBreak"); }
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
