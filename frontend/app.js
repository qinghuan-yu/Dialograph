"use strict";
const $ = selector => document.querySelector(selector);
const state = { token: "", config: null, jobs: [], selected: null, output: null, outputRequest: 0, submitting: false };
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
  select.replaceChildren(new Option(files.length ? "请选择一份对话" : "暂无文件，请导入 JSON", ""));
  files.forEach(file => select.add(new Option(`${file.name} · ${(file.bytes / 1024).toFixed(0)} KB`, file.id)));
  if (files.some(file => file.id === previous)) select.value = previous;
  select.disabled = !files.length;
  $("#file-hint").textContent = files.length ? `找到 ${files.length} 份本地 JSON。分析前会检查是否为支持的 WeFlow 格式。` : "读取 talks 文件夹，或导入 WeFlow 原始导出的 JSON。";
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
  $("#diagnostic-summary").textContent = `${stats.first_date} 至 ${stats.last_date} · 跳过 ${diagnostics.skipped_message_count || 0} 条 · 时间格式异常 ${diagnostics.invalid_time_message_count || 0} 条。会话按 ${stats.session_gap_hours} 小时间隔划分。`;
}
function renderJob(job) {
  if (!job) return;
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
    $("#report-content").replaceChildren(); showError("#report-error", null);
    try { localStorage.setItem("dialograph-job", id); } catch (_) { /* 私密模式可不保存选择。 */ }
  }
  renderHistory(); renderJob(state.jobs.find(job => job.id === id));
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
    const job = await post("/api/jobs", { chat_id: $("#conversation").value, mode: $("input[name=mode]:checked").value, chunk_size: Number($("#chunk-size").value), timeout: Number($("#timeout").value) });
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
    const result = await api(`/api/import?name=${encodeURIComponent(file.name)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: file }); await loadFiles(result.id);
  } catch (error) { showError("#form-error", error); }
  finally { $("#import-button").disabled = false; $("#import-button").textContent = "导入 JSON"; event.target.value = ""; }
});
$("#cancel-button").addEventListener("click", async () => {
  try { await post("/api/cancel", { id: state.selected }); await refreshJobs(); }
  catch (error) { showError("#form-error", error); }
});
$("#retry-button").addEventListener("click", async () => {
  $("#retry-button").disabled = true;
  try { await post("/api/jobs", { retry_of: state.selected }); state.output = null; await refreshJobs(); }
  catch (error) { showError("#form-error", error); }
  finally { $("#retry-button").disabled = false; }
});
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
