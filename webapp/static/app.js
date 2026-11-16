/* Glass-Box Insight lab - browser side.
 *
 * Flow:  settings form --POST /api/runs--> server trains in a thread and
 * streams events (/api/runs/<id>/events) --> this file draws them live.
 * After training, sliders call /predict and rule edits call /rules; both
 * are answered by the Python models in milliseconds.
 */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const { esc, fmt } = Charts;
const pct = (v, d = 1) => (v == null ? "–" : (v * 100).toFixed(d) + "%");
const CLASS = ["Non-diabetic", "Diabetic"];
const riskWord = (p) => (p < 0.2 ? "very low risk" : p < 0.4 ? "low risk" : p < 0.6 ? "uncertain"
  : p < 0.8 ? "high risk" : "very high risk");
const STEP = { BMI: 0.1, DiabetesPedigreeFunction: 0.001, Insulin: 0.5, SkinThickness: 0.5 };

const state = {
  defaults: null, runId: null, source: null, runs: [],
  done: null, initialModel: null, model: null, anfisEpochs: 0,
  patient: null, x: null, predictBusy: false, predictAgain: false,
  edit: null,
};
const charts = {};
let mfTrain = [], mfExplore = [];

// ------------------------------------------------------------------ helpers
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

const pending = new Set();
let frame = null;
function redraw(chart) {           // draw at most once per animation frame
  pending.add(chart);
  if (!frame) frame = requestAnimationFrame(() => { pending.forEach((c) => c.draw()); pending.clear(); frame = null; });
}

function setPill(text, kind = "") {
  const pill = $("#run-pill");
  pill.textContent = text;
  pill.className = "pill " + kind;
}

function showView(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  if (name === "runs") refreshRuns();
  requestAnimationFrame(() => Charts.redrawAll());
}

function conditionHTML(model, c, mu) {
  const f = model.features[c.feature];
  return `<span class="cond">${esc(f.name)} is <b>${esc(model.term_names[c.term])}</b>`
    + (mu != null ? ` <span class="mu">${mu.toFixed(2)}</span>` : "") + "</span>";
}
function thenHTML(model, rule) {
  if (model.order === 1) return `<span class="then">risk = ${esc(rule.formula)}</span>`;
  return `<span class="then">risk ${rule.output.toFixed(2)}</span> <span class="mu">${riskWord(rule.output)}</span>`;
}
function ruleSentence(model, rule) {
  const ifs = rule.conditions.map((c) => conditionHTML(model, c)).join('<span class="kw">AND</span>') || "<i>always</i>";
  return `<span class="kw">IF</span> ${ifs} <span class="kw">THEN</span> ${thenHTML(model, rule)}`;
}
function configSummary(c) {
  const parts = [`MLP ${c.hidden_layers} α=${c.alpha}${c.tune ? " (CV)" : ""}`, `${c.n_features} inputs`,
    `${c.n_terms} terms`, `${c.n_rules} clusters`];
  if (c.order === 1) parts.push("linear THEN");
  if (c.target === "truth") parts.push("true labels");
  parts.push(c.oracle_samples ? `${c.oracle_samples} synthetic` : "no synthetic");
  if (!c.constrain) parts.push("no constraints");
  parts.push(c.prune ? `prune ≤${pct(c.tolerance, 1)}` : "no pruning");
  parts.push(`seed ${c.seed}`);
  return parts.join(" · ");
}

// -------------------------------------------------------------- settings
function fillForm(cfg) {
  const form = $("#config");
  for (const [k, v] of Object.entries(cfg)) {
    const el = form.elements[k];
    if (!el) continue;
    if (el.type === "checkbox") el.checked = !!v;
    else el.value = String(v);
  }
  if (form.elements.alpha) form.elements.alpha.value = String(+cfg.alpha);
  updateOutputs();
}
function updateOutputs() {
  const form = $("#config");
  $$("output[data-for]", form).forEach((o) => {
    const v = +form.elements[o.dataset.for].value;
    o.textContent = o.hasAttribute("data-pct") ? pct(v, 1) : v;
  });
  form.elements.tolerance.disabled = !form.elements.prune.checked;
  form.elements.oracle_samples.disabled = form.elements.target.value === "truth";
}
function readForm() {
  const f = $("#config").elements;
  return {
    hidden_layers: f.hidden_layers.value, alpha: +f.alpha.value, tune: f.tune.checked,
    n_features: +f.n_features.value, n_terms: +f.n_terms.value, n_rules: +f.n_rules.value, order: +f.order.value,
    epochs: +f.epochs.value, target: f.target.value, constrain: f.constrain.checked,
    oracle_samples: +f.oracle_samples.value, prune: f.prune.checked, tolerance: +f.tolerance.value,
    seed: +f.seed.value, slow_motion: f.slow_motion.checked,
  };
}

// ------------------------------------------------------------------ train
function resetTrainView() {
  $("#train-empty").classList.add("hidden");
  $("#train-grid").classList.remove("hidden");
  $("#result-card").classList.add("hidden");
  $$("#stepper li").forEach((li) => (li.className = ""));
  ["#nn-kpi", "#anfis-kpi", "#prune-kpi", "#cv-list", "#rules-live", "#prune-log", "#mf-train"].forEach((s) => ($(s).innerHTML = ""));
  $("#cv-box").classList.add("hidden");
  charts.nn.setSeries([{ name: "training loss", color: "--s1", points: [] }]);
  charts.anfis.setSeries([
    { name: "training", color: "--s1", points: [] },
    { name: "validation", color: "--s2", points: [] },
  ]);
  charts.imp.setRows([]);
  mfTrain.forEach((c) => c.destroy());
  mfTrain = [];
  state.done = null; state.initialModel = null; state.model = null; state.anfisEpochs = 0;
  $$('.tab[data-view="explore"], .tab[data-view="rules"]').forEach((t) => (t.disabled = true));
}

function startRun(id, { replay = false } = {}) {
  if (state.source) state.source.close();
  state.runId = id;
  resetTrainView();
  setPill(replay ? "Loading run…" : "Training live…", "live");
  $("#train-btn").disabled = !replay;
  const source = new EventSource(`/api/runs/${id}/events`);
  state.source = source;
  source.onmessage = (msg) => handle(JSON.parse(msg.data));
  source.addEventListener("end", () => { source.close(); $("#train-btn").disabled = false; });
  source.onerror = () => { if (source.readyState === EventSource.CLOSED) $("#train-btn").disabled = false; };
}

function setStage(stage, status) {
  const li = $(`#stepper li[data-stage="${stage}"]`);
  if (li) li.className = status;
}

function renderLiveRules(model, flash = false) {
  const list = $("#rules-live");
  list.innerHTML = model.rules.map((r) => `<li class="${flash ? "flash" : ""}"><b>R${r.id}</b> ${ruleSentence(model, r)}</li>`).join("");
  if (flash) setTimeout(() => $$("#rules-live li").forEach((li) => li.classList.remove("flash")), 400);
}

function updateMFTrain(model) {
  if (mfTrain.length !== model.features.length) {
    mfTrain.forEach((c) => c.destroy());
    $("#mf-train").innerHTML = model.features.map((f, j) =>
      `<div class="mf-cell"><h4>${esc(f.name)}${f.unit ? " (" + esc(f.unit) + ")" : ""}</h4><div class="chart" id="mft-${j}"></div></div>`).join("");
    mfTrain = model.features.map((_, j) => new Charts.MFChart($(`#mft-${j}`)));
  }
  model.features.forEach((f, j) => {
    mfTrain[j].feature = f;
    mfTrain[j].initial = state.initialModel ? state.initialModel.features[j] : null;
    redraw(mfTrain[j]);
  });
}

function handle(ev) {
  switch (ev.type) {
    case "stage":
      setStage(ev.stage, ev.status);
      break;
    case "cv": {
      $("#cv-box").classList.remove("hidden");
      const row = document.createElement("div");
      row.className = "cv-row";
      row.dataset.auc = ev.auc;
      row.innerHTML = `<span>[${ev.hidden.join(",")}] α=${ev.alpha}</span><span><div class="cv-bar" style="width:${Math.max(4, (ev.auc - 0.7) / 0.15 * 100)}%"></div></span><span>${ev.auc.toFixed(3)}</span>`;
      $("#cv-list").appendChild(row);
      $("#nn-kpi").innerHTML = `cross-validating <b>${ev.index + 1}/${ev.total}</b>`;
      break;
    }
    case "cv_done":
      $$(".cv-row").forEach((r) => r.classList.toggle("best", r.firstChild.textContent === `[${ev.hidden.join(",")}] α=${ev.alpha}`));
      break;
    case "nn_epoch": {
      charts.nn.series[0].points.push([ev.epoch, ev.loss]);
      redraw(charts.nn);
      $("#nn-kpi").innerHTML = `epoch <b>${ev.epoch}</b> · loss <b>${ev.loss.toFixed(3)}</b> · validation accuracy <b>${pct(ev.val_accuracy)}</b>`;
      break;
    }
    case "nn_done":
      $("#nn-kpi").innerHTML = `MLP 8 → ${ev.hidden.join(" → ")} → 1 · test accuracy <b>${pct(ev.test.accuracy)}</b> · AUC <b>${ev.test.roc_auc.toFixed(2)}</b>`;
      break;
    case "importance":
      charts.imp.setRows(ev.rows.map((r) => ({
        label: r.feature, value: r.importance, highlight: ev.selected.includes(r.feature),
        tip: `<b>${esc(r.feature)}</b><br>output moves by ${r.importance.toFixed(3)} on average<br>${pct(r.share)} of total importance`,
      })));
      break;
    case "rules_proposed":
      state.initialModel = ev.model;
      renderLiveRules(ev.model);
      updateMFTrain(ev.model);
      $("#prune-kpi").innerHTML = `FCM: <b>${ev.clusters}</b> clusters → <b>${ev.model.rules.length}</b> distinct rules`;
      break;
    case "anfis_epoch": {
      if (ev.phase === "anfis") state.anfisEpochs = ev.epoch;
      const x = ev.phase === "anfis" ? ev.epoch : state.anfisEpochs + ev.epoch;
      charts.anfis.series[0].points.push([x, ev.train_loss]);
      if (ev.val_loss != null) charts.anfis.series[1].points.push([x, ev.val_loss]);
      redraw(charts.anfis);
      $("#anfis-kpi").innerHTML = `${ev.phase === "anfis" ? "epoch" : "fine-tuning epoch"} <b>${ev.epoch}</b> · MSE <b>${ev.train_loss.toFixed(4)}</b>`;
      if (ev.model) {
        state.model = ev.model;
        updateMFTrain(ev.model);
        if (ev.phase === "anfis") renderLiveRules(ev.model);
      }
      break;
    }
    case "prune_step": {
      renderLiveRules(ev.model, true);
      updateMFTrain(ev.model);
      const li = document.createElement("li");
      li.innerHTML = `${esc(ev.description)} → fidelity <b>${pct(ev.fidelity)}</b> · ${ev.rules} rules, <b>${ev.conditions}</b> conditions`;
      $("#prune-log").prepend(li);
      $("#prune-kpi").innerHTML = `<b>${ev.rules}</b> rules · <b>${ev.conditions}</b> conditions`;
      break;
    }
    case "done":
      onDone(ev);
      break;
    case "error":
      setPill("Training failed", "err");
      $("#train-btn").disabled = false;
      $("#prune-log").innerHTML = `<li><b>Error:</b> ${esc(ev.message)}</li>`;
      break;
  }
}

function tile(label, value, note = "", cls = "") {
  return `<div class="tile"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="note">${note}</div></div>`;
}

function onDone(ev) {
  state.done = ev;
  state.model = ev.model;
  renderLiveRules(ev.model);
  updateMFTrain(ev.model);
  const s = ev.summary;
  $("#prune-kpi").innerHTML = `<b>${s.final.rules}</b> rules · <b>${s.final.conditions}</b> conditions (from ${s.full.conditions})`;
  $("#result-tiles").innerHTML = [
    tile("Fidelity to the network", pct(s.final.fidelity), `${s.final.disagreements} of 154 decisions differ`),
    tile("Rule base", `${s.final.rules} rules`, `${s.final.conditions} conditions (before pruning ${s.full.conditions})`),
    tile("Decision tree, same size", pct(s.tree.fidelity), `${s.tree.rules} leaves, ${s.tree.conditions} conditions`),
    tile("Before pruning", pct(s.full.fidelity), `${s.full.rules} rules`),
    tile("Accuracy vs. diagnosis", pct(s.final.accuracy), `network: ${pct(s.nn_accuracy)}`),
    tile("Weak matches", pct(s.final.weak_match_share, 0), "patients no rule describes well"),
  ].join("");
  $("#result-card").classList.remove("hidden");
  setPill(`Run ${state.runId} · ${pct(s.final.fidelity)} fidelity`, "ok");
  $$('.tab[data-view="explore"], .tab[data-view="rules"]').forEach((t) => (t.disabled = false));
  // The server keeps any rule edits made earlier on this run; start from those.
  api(`/api/runs/${state.runId}/model`).then((cur) => {
    state.model = cur.model;
    state.edit = cur.model.rules.map((r) => r.antecedent.slice());
    renderEditor(cur.model);
    editTiles(cur.metrics, cur.trained);
    setupExplore(ev);
  });
}

// ---------------------------------------------------------------- explore
function setupExplore(done) {
  const sel = $("#patient");
  const pts = done.patients;
  sel.innerHTML = `<option value="-1">Typical patient (training medians)</option>` + pts.y.map((y, i) =>
    `<option value="${i}">Test patient #${i} · ${CLASS[y]} · network ${pts.nn_prob[i].toFixed(2)}</option>`).join("");
  sel.value = pts.y.length > 152 ? "152" : "0";
  mfExplore.forEach((c) => c.destroy());
  $("#mf-explore").innerHTML = done.model.features.map((f, j) =>
    `<div class="mf-cell"><h4>${esc(f.name)}${f.unit ? " (" + esc(f.unit) + ")" : ""}</h4><div class="chart" id="mfe-${j}"></div></div>`).join("");
  mfExplore = done.model.features.map((_, j) => new Charts.MFChart($(`#mfe-${j}`)));
  loadPatient();
}

function loadPatient() {
  const done = state.done;
  const i = +$("#patient").value;
  state.patient = i;
  state.x = (i < 0 ? done.typical : done.patients.x[i]).slice();
  const used = new Set(done.selected_index);
  $("#sliders").innerHTML = done.feature_names.map((name, j) => {
    const lo = done.ranges.lo[j], hi = done.ranges.hi[j];
    return `<div class="slider-row ${used.has(j) ? "used" : ""}">
      <div class="top"><span class="name">${esc(name)}</span><span><b id="val-${j}"></b> ${esc(done.units[j])}<span class="term" id="term-${j}"></span></span></div>
      <input type="range" data-j="${j}" min="${lo}" max="${hi}" step="${STEP[name] || 1}" value="${state.x[j]}"></div>`;
  }).join("");
  $$("#sliders input").forEach((el) => el.addEventListener("input", () => {
    state.x[+el.dataset.j] = +el.value;
    updateSliderLabels();
    predict();
  }));
  updateSliderLabels();
  predict();
}

function updateSliderLabels() {
  const done = state.done, model = state.model;
  done.feature_names.forEach((name, j) => {
    $(`#val-${j}`).textContent = fmt(state.x[j]);
    const k = done.selected_index.indexOf(j);
    const termEl = $(`#term-${j}`);
    if (k < 0) { termEl.textContent = ""; return; }
    const f = model.features[k];
    const mu = Charts.memberships(f, state.x[j], f.terms);
    termEl.textContent = f.terms[mu.indexOf(Math.max(...mu))].name;
  });
  const edited = state.patient >= 0 && state.x.some((v, j) => Math.abs(v - done.patients.x[state.patient][j]) > 1e-9);
  $("#truth").innerHTML = state.patient < 0 ? "A made-up patient: no real outcome."
    : `Actual outcome of patient #${state.patient}: <b>${CLASS[done.patients.y[state.patient]]}</b>`
      + (edited ? " (you changed the values, so this no longer applies)" : "");
}

async function predict() {
  if (state.predictBusy) { state.predictAgain = true; return; }
  state.predictBusy = true;
  try {
    renderExplanation(await api(`/api/runs/${state.runId}/predict`, { x: state.x }));
  } catch (err) {
    $("#fired").innerHTML = `<p class="hint">${esc(err.message)}</p>`;
  } finally {
    state.predictBusy = false;
    if (state.predictAgain) { state.predictAgain = false; predict(); }
  }
}

function gauge(el, label, p, decision) {
  el.innerHTML = `<div class="label">${label}</div>
    <div><span class="value">${p.toFixed(2)}</span><span class="decision">${decision}</span></div>
    <div class="meter"><i style="width:${(p * 100).toFixed(1)}%"></i><b title="decision threshold 0.5"></b></div>
    <div class="meter-scale"><span>0 · non-diabetic</span><span>0.5</span><span>diabetic · 1</span></div>`;
}

function renderExplanation(e) {
  const model = state.model;
  gauge($("#gauge-nn"), "Neural network · P(diabetic)", e.nn_prob, e.nn_decision);
  gauge($("#gauge-fz"), "Fuzzy rules · risk", e.fuzzy_output, e.fuzzy_decision);
  const m = e.match_strength;
  const match = m < 0.2 ? ["bad", "⚠", "Weak match", "No rule describes this patient well. Treat the explanation with care."]
    : m < 0.5 ? ["warn", "◐", "Partial match", "The patient sits between rules, so several rules share the decision."]
      : ["good", "✓", "Good match", "One rule describes this patient well."];
  $("#badges").innerHTML = `
    <div class="badge ${e.agree ? "good" : "bad"}"><strong><span class="icon">${e.agree ? "✓" : "✗"}</span>${e.agree ? "Both models agree" : "The models disagree"}</strong>
      ${e.agree ? "The rules reproduce the network's decision." : "The rules give the other decision. This happens most near 0.5."}</div>
    <div class="badge ${match[0]}"><strong><span class="icon">${match[1]}</span>${match[2]} · ${m.toFixed(2)}</strong>${match[3]}</div>`;

  // stacked contributions: segments add up to the fuzzy risk
  $("#stack").innerHTML = e.rules.filter((r) => r.contribution > 0.002).map((r) =>
    `<span data-tip="R${r.rule}: weight ${r.weight.toFixed(2)} × risk ${r.consequent.toFixed(2)} = ${r.contribution.toFixed(3)}"
      style="width:${(r.contribution * 100).toFixed(2)}%;background:var(${r.consequent >= 0.5 ? "--s2" : "--s1"})"></span>`).join("");
  $$("#stack span").forEach((sp) => {
    sp.addEventListener("pointermove", (evt) => {
      const tip = $("#tooltip"); tip.textContent = sp.dataset.tip; tip.classList.add("show");
      tip.style.left = evt.clientX + 12 + "px"; tip.style.top = evt.clientY + 12 + "px";
    });
    sp.addEventListener("pointerleave", () => $("#tooltip").classList.remove("show"));
  });

  const byId = Object.fromEntries(model.rules.map((r) => [r.id, r]));
  $("#fired").innerHTML = e.rules.map((r) => {
    const rule = byId[r.rule];
    const conds = rule ? rule.conditions.map((c, k) => conditionHTML(model, c, r.conditions[k] && r.conditions[k].membership))
      .join('<span class="kw">AND</span>') : "";
    return `<div class="fired-rule ${r.weight < 0.02 ? "faint" : ""}">
      <span class="rid">R${r.rule}</span>
      <span><span class="kw">IF</span> ${conds || "<i>always</i>"} <span class="kw">THEN</span> risk ${r.consequent.toFixed(2)}</span>
      <span><div class="weight" style="width:${Math.max(1, r.weight * 100)}%"></div></span>
      <span class="num">${r.weight.toFixed(2)} × ${r.consequent.toFixed(2)} = <b>${r.contribution.toFixed(3)}</b></span></div>`;
  }).join("");

  model.features.forEach((f, k) => {
    const j = state.done.selected_index[k];
    mfExplore[k].set(f, null, state.x[j]);
  });
}

// ---------------------------------------------------------------- editor
function editTiles(metrics, trained) {
  const delta = (a, b, better, unit = "pp") => {
    const d = (a - b) * (unit === "pp" ? 100 : 1);
    if (Math.abs(d) < 1e-9) return "same as trained";
    const good = better === "up" ? d > 0 : d < 0;
    return `<span class="${good ? "delta-up" : "delta-down"}">${d > 0 ? "+" : ""}${unit === "pp" ? d.toFixed(1) + " pts" : d}</span> vs trained`;
  };
  $("#edit-tiles").innerHTML = [
    tile("Fidelity (test)", pct(metrics.fidelity), delta(metrics.fidelity, trained.fidelity, "up")),
    tile("Conditions", metrics.conditions, delta(metrics.conditions, trained.conditions, "down", "n")),
    tile("Rules", metrics.rules, delta(metrics.rules, trained.rules, "down", "n")),
    tile("Accuracy vs. diagnosis", pct(metrics.accuracy), delta(metrics.accuracy, trained.accuracy, "up")),
    tile("Weak matches", pct(metrics.weak_match_share, 0), "patients no rule fits well"),
  ].join("");
}

function renderEditor(model) {   // one card per rule, built from state.edit
  const names = model.term_names;
  $("#editor").innerHTML = state.edit.map((a, r) => {
    const rule = model.rules[r];
    const chips = a.map((t, j) => t < 0 ? null :
      `<span class="chip" data-r="${r}" data-j="${j}" title="Click to change the term">${esc(model.features[j].name)} is <b>${esc(names[t])}</b>
        <button data-del="${r},${j}" aria-label="Remove condition">×</button></span>`).filter(Boolean).join('<span class="kw">AND</span>');
    const adds = a.map((t, j) => t >= 0 ? "" : `<span class="chip add" data-add="${r},${j}">+ ${esc(model.features[j].name)}</span>`).join("");
    return `<div class="edit-rule">
      <span class="rid">R${r + 1}</span>
      <div class="chips"><span class="kw">IF</span>${chips || "<i>always</i>"}${adds}</div>
      <span class="out"><span class="kw">THEN</span> ${rule ? thenHTML(model, rule) : ""}</span>
      <button class="del" data-delrule="${r}" title="Delete this rule" aria-label="Delete rule">🗑</button></div>`;
  }).join("");
}

async function submitEdit() {
  try {
    const res = await api(`/api/runs/${state.runId}/rules`, { antecedents: state.edit });
    state.model = res.model;
    state.edit = res.model.rules.map((r) => r.antecedent.slice());
    renderEditor(res.model);
    editTiles(res.metrics, res.trained);
    if (state.x) { updateSliderLabels(); predict(); }
  } catch (err) {
    alert(err.message);
  }
}

$("#editor").addEventListener("click", (evt) => {
  const t = evt.target.closest("[data-del],[data-add],[data-delrule],.chip");
  if (!t || !state.edit) return;
  const nTerms = state.model.term_names.length;
  if (t.dataset.del) {
    const [r, j] = t.dataset.del.split(",").map(Number);
    state.edit[r][j] = -1;
  } else if (t.dataset.add) {
    const [r, j] = t.dataset.add.split(",").map(Number);
    state.edit[r][j] = Math.floor(nTerms / 2);
  } else if (t.dataset.delrule) {
    if (state.edit.length === 1) { alert("A rule base needs at least one rule."); return; }
    state.edit.splice(+t.dataset.delrule, 1);
  } else {
    const r = +t.dataset.r, j = +t.dataset.j;
    state.edit[r][j] = (state.edit[r][j] + 1) % nTerms;
  }
  submitEdit();
});
$("#add-rule").addEventListener("click", () => {
  if (!state.edit) return;
  const rule = state.edit[0].map(() => -1);
  rule[0] = Math.floor(state.model.term_names.length / 2);
  state.edit.push(rule);
  submitEdit();
});
$("#reset-rules").addEventListener("click", () => {
  if (!state.done) return;
  state.edit = state.done.model.rules.map((r) => r.antecedent.slice());
  submitEdit();
});

// ------------------------------------------------------------------ runs
async function refreshRuns() {
  const runs = await api("/api/runs");
  state.runs = runs;
  const points = [];
  runs.forEach((r, i) => {
    if (!r.summary) return;
    const f = r.summary.final, tip = `<b>Run ${i + 1}</b> · ${pct(f.fidelity)} fidelity, ${f.conditions} conditions<br>${esc(configSummary(r.config))}`;
    points.push({ x: f.conditions, y: f.fidelity, label: `#${i + 1}`, color: r.config.target === "truth" ? "--s2" : "--s1", tip, id: r.id });
    points.push({ x: r.summary.tree.conditions, y: r.summary.tree.fidelity, color: "--s3", id: r.id,
      tip: `<b>Decision tree</b> for run ${i + 1}: ${r.summary.tree.rules} leaves, ${pct(r.summary.tree.fidelity)}` });
    if (r.edited) points.push({ x: r.edited.conditions, y: r.edited.fidelity, hollow: true, color: "--s1", id: r.id,
      tip: `<b>Run ${i + 1} after your edits</b><br>${r.edited.rules} rules, ${r.edited.conditions} conditions, ${pct(r.edited.fidelity)}` });
  });
  charts.runs.setPoints(points);
  $("#runs-table").innerHTML = `<thead><tr><th>#</th><th>Started</th><th>Settings</th><th>Rules</th><th>Conditions</th>
    <th>Fidelity</th><th>Tree, same size</th><th>Accuracy</th><th>Status</th></tr></thead><tbody>` +
    runs.map((r, i) => {
      const f = r.summary && r.summary.final;
      return `<tr data-id="${r.id}" class="${r.id === state.runId ? "current" : ""}">
        <td>${i + 1}</td><td>${new Date(r.created * 1000).toLocaleTimeString()}</td><td>${esc(configSummary(r.config))}</td>
        <td>${f ? f.rules : "–"}</td><td>${f ? f.conditions : "–"}${r.edited ? ` → ${r.edited.conditions}` : ""}</td>
        <td>${f ? pct(f.fidelity) : "–"}${r.edited ? ` → ${pct(r.edited.fidelity)}` : ""}</td>
        <td>${r.summary ? pct(r.summary.tree.fidelity) : "–"}</td><td>${f ? pct(f.accuracy) : "–"}</td><td>${r.status}</td></tr>`;
    }).join("") + "</tbody>";
}
$("#runs-table").addEventListener("click", (evt) => {
  const tr = evt.target.closest("tr[data-id]");
  if (tr) openRun(tr.dataset.id);
});
function openRun(id) {
  showView("train");
  startRun(id, { replay: true });
}

// ------------------------------------------------------------------- init
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) { /* storage blocked */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
    Charts.redrawAll();
  });
}

async function init() {
  initTheme();
  charts.nn = new Charts.LineChart($("#chart-nn"), { xLabel: "epoch", yLabel: "loss" });
  charts.anfis = new Charts.LineChart($("#chart-anfis"), { xLabel: "epoch", yLabel: "MSE vs network" });
  charts.imp = new Charts.BarChart($("#chart-imp"), { format: (r) => r.value.toFixed(3) });
  charts.runs = new Charts.ScatterChart($("#chart-runs"), {
    xLabel: "total conditions in the rule base", yLabel: "fidelity", onClick: (p) => openRun(p.id),
  });

  $$(".tab").forEach((t) => t.addEventListener("click", () => !t.disabled && showView(t.dataset.view)));
  $$("[data-goto]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.goto)));
  $("#patient").addEventListener("change", loadPatient);
  $("#config").addEventListener("input", updateOutputs);
  $("#config").addEventListener("submit", async (evt) => {
    evt.preventDefault();
    $("#train-btn").disabled = true;
    try {
      const run = await api("/api/runs", readForm());
      startRun(run.id);
    } catch (err) {
      setPill("Could not start: " + err.message, "err");
      $("#train-btn").disabled = false;
    }
  });
  $("#reset-btn").addEventListener("click", () => fillForm(state.defaults));

  const d = await api("/api/defaults");
  state.defaults = d.defaults;
  fillForm(d.defaults);
  const runs = await api("/api/runs");
  if (runs.length) openRun(runs[runs.length - 1].id);
}

init().catch((err) => setPill("Server not reachable: " + err.message, "err"));
