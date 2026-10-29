"use strict";
const $ = selector => document.querySelector(selector);
const state = { token: "", config: null, jobs: [], selected: null, output: null, outputRequest: 0, submitting: false,
  evidence: { job: null, request: 0, revision: null } };
const labels = { queued: "等待启动", running: "正在运行", completed: "已完成", failed: "失败", cancelled: "已停止", interrupted: "已中断" };
const stageOrder = ["prepare", "chunks", "merge", "analysis", "persona", "audit"];
const active = job => job && ["queued", "running"].includes(job.status);
let lastJobsSnapshot = "";

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function api(path, options = {}) {
  let response = await fetch(path, { ...options, headers: { "X-Dialograph-Token": state.token, ...options.headers } });
  if (response.status === 403 && path !== "/api/bootstrap") {
    const bootstrap = await fetch("/api/bootstrap");
    if (bootstrap.ok) {
      state.config = await bootstrap.json(); state.token = state.config.token;
      response = await fetch(path, { ...options, headers: { "X-Dialograph-Token": state.token, ...options.headers } });
    }
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败，请重试。");
  return data;
}
const post = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
function showError(selector, error) {
  $(selector).textContent = error ? error.message || error : "";
  $(selector).hidden = !error;
}
async function loadFiles(selectId) {
  const files = await api("/api/conversations");
  const select = $("#conversation"), previous = selectId || select.value;
  select.replaceChildren(new Option(files.length ? "请选择一份对话" : "暂无文件，请导入对话", ""));
  files.forEach(file => select.add(new Option(`${file.name} · ${(file.bytes / 1024).toFixed(0)} KB`, file.id)));
  if (files.some(file => file.id === previous)) select.value = previous;
  select.disabled = !files.length;
  $("#file-hint").textContent = files.length ? `找到 ${files.length} 份本地对话。分析前会检查格式与发送者归属。` : "支持 WeFlow JSON、统一协议 JSON 和 Alpha Markdown 私聊导出。";
  updateForm();
}
function updateForm() {
  const full = $("input[name=mode]:checked").value === "full", busy = state.jobs.some(active);
  $("#model-notice").hidden = !full;
  $("#model-notice").textContent = state.config?.configured
    ? `完整分析会将对话分块发送至 ${state.config.provider}，使用 ${state.config.model}，可能产生模型费用。`
    : "尚未配置模型密钥，请在项目根目录 .env 中设置 LLM_API_KEY 后重启服务。仍可运行仅预处理。";
  $("#start-button").textContent = busy ? "任务运行中" : state.submitting ? "正在启动…" : full ? "开始完整分析 ↗" : "开始预处理 ↗";
  $("#start-button").disabled = !state.token || !$("#conversation").value || busy || state.submitting || (full && !state.config?.configured);
  $("#mode-description").textContent = full ? "逐块阅读全部有效消息，再生成报告。" : "解析与统计在本机完成，不调用模型。";
  $("#chunk-size").disabled = !full;
  $("#timeout").disabled = !full;
}
function renderHistory() {
  const list = $("#job-history"); list.replaceChildren();
  $("#history-empty").hidden = !!state.jobs.length;
  state.jobs.slice(0, 12).forEach(job => {
    const button = element("button", undefined, "history-item"); button.type = "button";
    button.setAttribute("aria-pressed", String(job.id === state.selected));
    button.append(element("strong", job.name), element("small", `${job.mode === "full" ? "完整分析" : "预处理"} · ${labels[job.status]} · ${job.created.replace("T", " ")}`));
    button.addEventListener("click", () => selectJob(job.id)); list.append(button);
  });
}
function renderStats(job) {
  const stats = job.stats; $("#stats-panel").hidden = !stats;
  if (!stats) return;
  const metrics = $("#metrics"); metrics.replaceChildren();
  [["有效消息", stats.total_messages], ["活跃天数", stats.active_days], ["会话数", stats.session_count], ["对话轮次", stats.conversation_turn_count]].forEach(([name, value]) => {
    const group = element("div"); group.append(element("dt", name), element("dd", Number(value).toLocaleString("zh-CN"))); metrics.append(group);
  });
  const monthly = (stats.monthly_trend || []).slice(-12), max = Math.max(1, ...monthly.map(month => month.total));
  const chart = $("#monthly-chart"); chart.replaceChildren();
  monthly.forEach(month => {
    const row = element("div", undefined, "month-row"), meter = element("meter");
    meter.min = 0; meter.max = max; meter.value = month.total;
    meter.setAttribute("aria-label", `${month.month}，我 ${month.my_count} 条，对方 ${month.other_count} 条，总计 ${month.total} 条`);
    row.append(element("span", month.month), meter, element("span", `${month.my_count} / ${month.other_count}`)); chart.append(row);
  });
  const diagnostics = job.diagnostics || {};
  const redactionCount = Object.values(diagnostics.redaction?.replacement_counts || {}).reduce((sum, value) => sum + value, 0);
  $("#diagnostic-summary").textContent = `${stats.first_date} 至 ${stats.last_date} · 跳过 ${diagnostics.skipped_message_count || 0} 条 · 时间格式异常 ${diagnostics.invalid_time_message_count || 0} 条。会话按 ${stats.session_gap_hours} 小时间隔划分。${diagnostics.redaction?.enabled ? ` 已启用脱敏，识别替换 ${redactionCount} 处。` : ""}`;
}
function renderJob(job) {
  if (!job) return;
  const searchable = job.status === "completed" && job.mode === "full" && !!job.files?.evidence;
  if (state.evidence.job !== job.id || !searchable) resetEvidence(searchable ? job.id : null);
  $("#evidence-panel").hidden = !searchable;
  $("#job-status").textContent = labels[job.status]; $("#job-name").textContent = job.name;
  $("#live-message").textContent = job.message || "正在启动…";
  $("#live-message").classList.toggle("error-text", job.status === "failed");
  $("#cancel-button").hidden = !active(job);
  $("#retry-button").hidden = !["failed", "cancelled", "interrupted"].includes(job.status);
  const currentIndex = stageOrder.indexOf(job.stage);
  document.querySelectorAll("#stages li").forEach((node, index) => {
    const skipped = job.mode === "preprocess" && index > 0;
    const done = !skipped && (job.status === "completed" || index < currentIndex);
    const current = !skipped && index === currentIndex && job.status !== "completed";
    node.className = skipped ? "skipped" : done ? "done" : current ? (job.status === "failed" ? "failed" : "active") : "";
    node.querySelector("small").textContent = skipped ? "无需执行" : done ? "已完成" : current ? (active(job) ? "执行中" : labels[job.status]) : "等待开始";
    if (current) node.setAttribute("aria-current", "step"); else node.removeAttribute("aria-current");
  });
  $("#chunk-progress").hidden = !(job.total > 0);
  const chunksDone = currentIndex > 1 || job.status === "completed" ? job.total : job.completed;
  $("#chunk-label").textContent = job.stage === "chunks" ? `当前分块 ${job.current} / ${job.total}` : "分块处理";
  $("#chunk-count").textContent = `已完成 ${chunksDone || 0} / ${job.total || 0}`;
  $("#chunk-meter").max = job.total || 1; $("#chunk-meter").value = chunksDone || 0;
  const logs = $("#logs"); logs.replaceChildren();
  (job.events || []).forEach(event => { const row = element("li"); row.append(element("time", event.time.slice(11)), document.createTextNode(event.message)); logs.append(row); });
  $("#log-count").textContent = job.events?.length || 0;
  renderStats(job);
  const files = Object.entries(job.files || {});
  $("#result-label").textContent = job.status === "completed" ? `${files.length} 份输出` : job.stats ? "统计已就绪" : "等待分析";
  $("#results-empty").hidden = !!job.stats; $("#report-panel").hidden = !files.length;
  const tabs = $("#report-tabs"); tabs.replaceChildren();
  files.forEach(([key, file]) => {
    const button = element("button", file.label); button.type = "button"; button.dataset.output = key;
    button.setAttribute("aria-pressed", String(state.output === key)); button.addEventListener("click", () => loadOutput(key)); tabs.append(button);
  });
  if (files.length && !state.output) loadOutput(files[0][0]);
}
function selectJob(id) {
  if (state.selected !== id) {
    state.selected = id; state.output = null; state.outputRequest += 1;
    resetEvidence(id);
    $("#report-content").replaceChildren(); showError("#report-error", null);
    try { localStorage.setItem("dialograph-job", id); } catch (_) { /* 私密模式可不保存选择。 */ }
  }
  renderHistory(); renderJob(state.jobs.find(job => job.id === id));
}
function resetEvidence(jobId) {
  state.evidence = { job: jobId, request: state.evidence.request + 1, revision: null };
  $("#evidence-results").replaceChildren(); $("#evidence-query").value = "";
  $("#evidence-status").textContent = "输入关键词，查询当前任务的脱敏证据。首次查询需在本机建立索引。";
  $("#evidence-results").setAttribute("aria-busy", "false");
  showError("#evidence-error", null);
}
function evidenceRequestCurrent(job, request) {
  return state.selected === job && state.evidence.job === job && state.evidence.request === request;
}
async function searchEvidence() {
  const job = state.selected, query = $("#evidence-query").value.trim();
  const request = ++state.evidence.request;
  state.evidence.revision = null;
  $("#evidence-results").replaceChildren(); showError("#evidence-error", null);
  $("#evidence-results").setAttribute("aria-busy", "false");
  if (!query) { $("#evidence-status").textContent = "请输入查询关键词。"; return; }
  $("#evidence-status").textContent = "正在本机检索…";
  $("#evidence-results").setAttribute("aria-busy", "true");
  try {
    const data = await post("/api/evidence/search", { id: job, query });
    if (!evidenceRequestCurrent(job, request)) return;
    if (data.job_id !== job) throw new Error("任务结果不一致，请重新查询。");
    state.evidence.revision = data.ledger_revision;
    renderEvidenceResults(data);
    const count = Object.values(data.matches).reduce((sum, rows) => sum + rows.length, 0);
    $("#evidence-status").textContent = `“${query}” · 共 ${count} 条匹配，每类最多展示 5 条。`;
  } catch (error) {
    if (!evidenceRequestCurrent(job, request)) return;
    $("#evidence-status").textContent = "查询未完成，可重新提交。"; showError("#evidence-error", error);
  } finally {
    if (evidenceRequestCurrent(job, request)) $("#evidence-results").setAttribute("aria-busy", "false");
  }
}
function renderEvidenceResults(data) {
  const container = $("#evidence-results"); container.replaceChildren();
  const categories = { events: "事件", relation_signals: "关系信号", persona_signals: "人物信号", counter_evidence: "反证", uncertainties: "存疑" };
  for (const [role, label] of [["support", "支持候选"], ["counter", "反证"], ["mixed", "混合信号"], ["uncertainty", "存疑"]]) {
    const rows = data.matches[role] || [], group = element("section", undefined, "evidence-group");
    group.append(element("h4", `${label} · ${rows.length}`));
    if (!rows.length) group.append(element("p", `此关键词未匹配到${label}，不代表整段对话中不存在。`, "field-hint"));
    for (const row of rows) {
      const card = element("article", undefined, "evidence-row"), payload = row.payload || {};
      card.append(element("p", [payload.summary, payload.signal, payload.trait, payload.model, payload.against, payload.question].filter(Boolean).join(" · "), "evidence-summary"));
      if (payload.reason) card.append(element("p", payload.reason));
      if (role !== "uncertainty" && row.quote) card.append(element("blockquote", row.quote));
      card.append(element("p", `${categories[row.category] || row.category} · ${row.evidence_id}`, "evidence-meta"));
      if (row.confidence !== undefined && row.confidence !== null) card.append(element("p", `原抽取置信度：${({ high: "高", medium: "中", low: "低" })[row.confidence] || row.confidence}（不代表结论置信度）`, "field-hint"));
      if (row.grounding_status === "matched" && role !== "uncertainty") {
        const button = element("button", "回查原文", "text-button"), source = element("div", undefined, "evidence-source");
        button.type = "button"; button.setAttribute("aria-expanded", "false"); source.hidden = true;
        button.addEventListener("click", async () => {
          source.hidden = !source.hidden; button.setAttribute("aria-expanded", String(!source.hidden));
          if (!source.hidden && !source.dataset.loaded) {
            button.disabled = true;
            try { await loadEvidenceSource(row, source); } finally { button.disabled = false; }
          }
        });
        card.append(button, source);
      } else card.append(element("p", "存疑项没有事实级原文锚点，不能作为支持证据。", "field-hint"));
      group.append(card);
    }
    container.append(group);
  }
}
function renderEvidenceSource(item, container) {
  container.replaceChildren();
  for (const occurrence of item.occurrences || []) {
    for (const ref of occurrence.source_refs || []) {
      const source = element("div"), text = element("p", undefined, "source-message");
      source.append(element("p", `分块 ${occurrence.chunk_index} · 原始第 ${ref.original_index + 1} 条 · ${ref.sender === "me" ? "我" : ref.sender === "other" ? "对方" : ref.sender} · ${ref.time}`, "evidence-meta"));
      // 后端位置以 Unicode 字符计数，避免表情符号导致高亮错位。
      const chars = Array.from(ref.content);
      text.append(document.createTextNode(chars.slice(0, ref.start).join("")), element("mark", chars.slice(ref.start, ref.end).join("")), document.createTextNode(chars.slice(ref.end).join("")));
      source.append(text); container.append(source);
    }
  }
}
async function loadEvidenceSource(row, container) {
  const { job, request, revision } = state.evidence;
  container.textContent = "正在回查脱敏原文…";
  try {
    const data = await post("/api/evidence/show", { id: job, evidence_id: row.evidence_id, revision });
    if (!evidenceRequestCurrent(job, request) || revision !== state.evidence.revision || !container.isConnected) return;
    if (data.job_id !== job || data.ledger_revision !== revision) throw new Error("证据版本已改变，请重新检索。");
    renderEvidenceSource(data.evidence, container); container.dataset.loaded = "true";
  } catch (error) {
    if (evidenceRequestCurrent(job, request) && container.isConnected) container.textContent = error.message || "回查失败，请重试。";
  }
}
// 仅渲染安全文本，不执行报告中的 HTML 或加载外部资源。
function inlineText(node, text) {
  text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).forEach(part => {
    if (part.startsWith("**") && part.endsWith("**")) node.append(element("strong", part.slice(2, -2)));
    else if (part.startsWith("`") && part.endsWith("`")) node.append(element("code", part.slice(1, -1)));
    else node.append(document.createTextNode(part));
  });
}
function renderMarkdown(text, container) {
  const lines = text.replace(/<!--[\s\S]*?-->/g, "").split(/\r?\n/);
  let code = null, list = null, table = null;
  for (const line of lines) {
    if (line.startsWith("```")) { if (code) code = null; else { code = element("pre", ""); container.append(code); } continue; }
    if (code) { code.textContent += line + "\n"; continue; }
    if (!line.trim()) { list = null; table = null; continue; }
    if (/^\s*\|.*\|\s*$/.test(line)) {
      if (/^\s*\|[\s:|\-]+\|\s*$/.test(line)) continue;
      if (!table) { const wrap = element("div", undefined, "table-wrap"); table = element("table"); wrap.append(table); container.append(wrap); }
      const row = element("tr"); line.trim().slice(1, -1).split("|").forEach(cell => { const td = element("td"); inlineText(td, cell.trim()); row.append(td); }); table.append(row); continue;
    }
    table = null;
    const heading = /^(#{1,6})\s+(.*)/.exec(line), item = /^\s*(?:[-*+] |\d+\. )(.*)/.exec(line);
    let node;
    if (heading) { list = null; node = element(`h${heading[1].length}`); inlineText(node, heading[2]); }
    else if (item) { if (!list) { list = element("ul"); container.append(list); } node = element("li"); inlineText(node, item[1]); list.append(node); continue; }
    else { list = null; node = element(line.startsWith(">") ? "blockquote" : "p"); inlineText(node, line.replace(/^>\s?/, "")); }
    container.append(node);
  }
}
async function loadOutput(key) {
  state.output = key;
  const id = state.selected, request = ++state.outputRequest;
  document.querySelectorAll("[data-output]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.output === key)));
  const content = $("#report-content"); content.textContent = "正在读取结果…";
  $("#download-button").disabled = true; showError("#report-error", null);
  try {
    const output = await api(`/api/outputs/${id}/${key}`);
    if (request !== state.outputRequest) return;
    content.replaceChildren();
    if (output.kind === "markdown") renderMarkdown(output.content, content);
    else content.append(element("pre", output.kind === "json" ? JSON.stringify(JSON.parse(output.content), null, 2) : output.content));
    $("#download-button").disabled = false;
  } catch (error) { if (request !== state.outputRequest) return; content.replaceChildren(); showError("#report-error", error); }
}
async function refreshJobs() {
  state.jobs = await api("/api/jobs");
  const snapshot = JSON.stringify(state.jobs);
  if (snapshot === lastJobsSnapshot) { updateForm(); return; }
  lastJobsSnapshot = snapshot;
  if (!state.selected && state.jobs.length) {
    let saved; try { saved = localStorage.getItem("dialograph-job"); } catch (_) { /* 使用最新任务。 */ }
    state.selected = state.jobs.some(job => job.id === saved) ? saved : state.jobs[0].id;
  }
  renderHistory(); renderJob(state.jobs.find(job => job.id === state.selected)); updateForm();
}
async function poll() {
  try { await refreshJobs(); showError("#connection-error", null); }
  catch (_) { showError("#connection-error", "本地服务连接中断，正在重连。连接恢复后将更新任务状态。"); }
  setTimeout(poll, state.jobs.some(active) ? 1200 : 5000);
}
$("#analysis-form").addEventListener("submit", async event => {
  event.preventDefault(); state.submitting = true; updateForm(); showError("#form-error", null);
  try {
    const redactionTerms = $("#redaction-terms").value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
    const job = await post("/api/jobs", { chat_id: $("#conversation").value, mode: $("input[name=mode]:checked").value, chunk_size: Number($("#chunk-size").value), timeout: Number($("#timeout").value), redaction_terms: redactionTerms });
    await refreshJobs(); selectJob(job.id);
  } catch (error) { showError("#form-error", error); }
  finally { state.submitting = false; updateForm(); }
});
$("#conversation").addEventListener("change", updateForm);
document.querySelectorAll("input[name=mode]").forEach(input => input.addEventListener("change", updateForm));
$("#refresh-files").addEventListener("click", () => loadFiles().catch(error => showError("#form-error", error)));
$("#refresh-jobs").addEventListener("click", () => refreshJobs().catch(error => showError("#connection-error", error)));
$("#import-button").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file) return;
  showError("#form-error", null); $("#import-button").disabled = true; $("#import-button").textContent = "正在导入…";
  try {
    if (file.size > 50 * 1024 * 1024) throw new Error("文件超过 50 MB，请先拆分导出。");
    const result = await api(`/api/import?name=${encodeURIComponent(file.name)}`, { method: "POST", headers: { "Content-Type": file.name.toLowerCase().endsWith(".md") ? "text/markdown;charset=utf-8" : "application/json" }, body: file }); await loadFiles(result.id);
  } catch (error) { showError("#form-error", error); }
  finally { $("#import-button").disabled = false; $("#import-button").textContent = "导入对话"; event.target.value = ""; }
});
$("#cancel-button").addEventListener("click", async () => {
  try { await post("/api/cancel", { id: state.selected }); await refreshJobs(); }
  catch (error) { showError("#form-error", error); }
});
$("#retry-button").addEventListener("click", async () => {
  $("#retry-button").disabled = true;
  resetEvidence(null); $("#evidence-panel").hidden = true;
  try { await post("/api/jobs", { retry_of: state.selected }); state.output = null; await refreshJobs(); }
  catch (error) { showError("#form-error", error); }
  finally { $("#retry-button").disabled = false; renderJob(state.jobs.find(job => job.id === state.selected)); }
});
$("#evidence-form").addEventListener("submit", event => { event.preventDefault(); searchEvidence(); });
$("#download-button").addEventListener("click", async () => {
  try {
    const output = await api(`/api/outputs/${state.selected}/${state.output}`);
    const url = URL.createObjectURL(new Blob([output.content], { type: "text/plain;charset=utf-8" }));
    const link = element("a"); link.href = url; link.download = output.name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { showError("#report-error", error); }
});
async function init() {
  try {
    state.config = await api("/api/bootstrap"); state.token = state.config.token;
    $("#model-name").textContent = state.config.model; $("#config-status").textContent = state.config.configured ? "已配置" : "未配置密钥";
    $("#config-hint").textContent = state.config.configured ? `完整分析使用 ${state.config.provider}。密钥仅保存在服务端。` : "仅预处理可直接使用；完整分析需先配置项目 .env。";
    await loadFiles(); await refreshJobs(); poll();
  } catch (_) { showError("#connection-error", "未连接到分析服务。请从项目根目录运行 python agent/web_app.py，再访问此页面。静态预览无法执行分析。"); $("#config-status").textContent = "未连接"; }
}
init();
