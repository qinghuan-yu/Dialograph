"use strict";
const $ = selector => document.querySelector(selector);
const state = { token: "", config: null, jobs: [], selected: null, output: null, outputRequest: 0, submitting: false, readingJob: null, progressJob: null, setup: false,
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
  $("#file-hint").textContent = files.length ? `${files.length} 份对话可供选择` : "支持 WeFlow JSON、统一协议 JSON 和 Alpha Markdown。";
  updateForm();
}
function updateForm() {
  const full = $("input[name=mode]:checked").value === "full", busy = state.jobs.some(active);
  $("#model-notice").hidden = !full;
  $("#model-notice").textContent = state.config?.configured
    ? `完整分析会将对话分块发送至 ${state.config.provider}，使用 ${state.config.model}，可能产生模型费用。`
    : "尚未配置模型密钥，请在项目根目录 .env 中设置 LLM_API_KEY 后重启服务。仍可运行仅预处理。";
  $("#start-button").textContent = busy ? "任务运行中" : state.submitting ? "正在启动…" : full ? "开始完整分析" : "开始预处理";
  $("#start-button").disabled = !state.token || !$("#conversation").value || busy || state.submitting || (full && !state.config?.configured);
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
  const stats = job.stats; $("#stats-panel").hidden = !stats; $("#stats-details").hidden = !stats;
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
  const notes = [`${stats.first_date} 至 ${stats.last_date}`];
  if (diagnostics.skipped_message_count) notes.push(`跳过 ${diagnostics.skipped_message_count} 条`);
  if (diagnostics.invalid_time_message_count) notes.push(`时间格式异常 ${diagnostics.invalid_time_message_count} 条`);
  if (diagnostics.invalid_timestamp_message_count) notes.push(`时间戳异常 ${diagnostics.invalid_timestamp_message_count} 条`);
  $("#diagnostic-summary").textContent = notes.join(" · ");
}
function renderJob(job) {
  if (!job) return;
  const reading = job.status === "completed" && job.mode === "full";
  document.body.classList.toggle("reading-mode", reading && !state.setup);
  if (reading && state.readingJob !== job.id) {
    $("#input-panel").open = false; $("#process-panel").open = false; $("#stats-details").open = false; state.readingJob = job.id;
  }
  if (active(job) && state.progressJob !== job.id) {
    $("#process-panel").open = true; $("#input-panel").open = false; state.progressJob = job.id;
  }
  if (!active(job)) state.progressJob = null;
  if (job.mode === "preprocess") $("#stats-details").open = true;
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
    node.hidden = skipped;
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
  const reports = files.filter(([key]) => ["analysis", "persona"].includes(key));
  $("#results-empty").hidden = !!job.stats; $("#report-panel").hidden = !reports.length;
  const tabs = $("#report-tabs"); tabs.replaceChildren();
  reports.forEach(([key, file]) => {
    const button = element("button", file.label); button.type = "button"; button.dataset.output = key;
    button.setAttribute("aria-pressed", String(state.output === key)); button.addEventListener("click", () => loadOutput(key)); tabs.append(button);
  });
  const attachments = $("#attachment-list"); attachments.replaceChildren();
  files.filter(([key]) => !["analysis", "persona"].includes(key)).forEach(([key, file]) => {
    const button = element("button", `下载${file.label}`, "text-button"); button.type = "button";
    button.addEventListener("click", () => downloadOutput(key)); attachments.append(button);
  });
  $("#attachments").hidden = !attachments.children.length;
  if (reports.length && !state.output) loadOutput(reports[0][0]);
}
function selectJob(id) {
  if (state.selected !== id) {
    state.selected = id; state.output = null; state.outputRequest += 1; state.setup = false; state.readingJob = null;
    resetEvidence(id);
    $("#report-content").replaceChildren(); showError("#report-error", null);
    try { localStorage.setItem("dialograph-job", id); } catch (_) { /* 私密模式可不保存选择。 */ }
  }
  renderHistory(); renderJob(state.jobs.find(job => job.id === id));
}
function resetEvidence(jobId) {
  state.evidence = { job: jobId, request: state.evidence.request + 1, revision: null };
  $("#evidence-results").replaceChildren(); $("#evidence-query").value = "";
  $("#evidence-status").textContent = "";
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
  for (const [role, label] of [["support", "相关线索"], ["counter", "反证"], ["mixed", "多种解释"], ["uncertainty", "待了解"]]) {
    const rows = data.matches[role] || [], group = element("section", undefined, "evidence-group");
    if (!rows.length) continue;
    group.append(element("h4", `${label} · ${rows.length}`));
    for (const row of rows) {
      const card = element("article", undefined, "evidence-row"), payload = row.payload || {};
      card.append(element("p", [payload.summary, payload.signal, payload.trait, payload.model, payload.against, payload.question].filter(Boolean).join(" · "), "evidence-summary"));
      if (payload.reason) card.append(element("p", payload.reason));
      if (role !== "uncertainty" && row.quote) card.append(element("blockquote", row.quote));
      card.append(element("p", categories[row.category] || row.category, "evidence-meta"));
      if (row.confidence !== undefined && row.confidence !== null) card.append(element("p", `线索可信度：${({ high: "高", medium: "中", low: "低" })[row.confidence] || row.confidence}`, "field-hint"));
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
      }
      group.append(card);
    }
    container.append(group);
  }
  if (!container.children.length) container.append(element("p", "没有找到相关记录。", "field-hint"));
}
function renderEvidenceSource(item, container) {
  container.replaceChildren();
  for (const occurrence of item.occurrences || []) {
    for (const ref of occurrence.source_refs || []) {
      const source = element("div"), text = element("p", undefined, "source-message");
      source.append(element("p", `${ref.sender === "me" ? "我" : ref.sender === "other" ? "对方" : ref.sender} · ${ref.time}`, "evidence-meta"));
      // 后端位置以 Unicode 字符计数，避免表情符号导致高亮错位。
      const chars = Array.from(ref.content);
      if (ref.end > ref.start) text.append(document.createTextNode(chars.slice(0, ref.start).join("")), element("mark", chars.slice(ref.start, ref.end).join("")), document.createTextNode(chars.slice(ref.end).join("")));
      else text.textContent = ref.content;
      const location = element("details", undefined, "source-location"); location.append(element("summary", "来源位置"), element("p", `原始第 ${ref.original_index + 1} 条`));
      source.append(text, location); container.append(source);
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
function reportBody(text) {
  let introduction = true;
  return text.split(/\n\s*\n/).filter(block => {
    if (/^##\s/.test(block.trim())) introduction = false;
    return !introduction || !/^(?:>\s*)?(?:\*\*)?(?:来源说明|重要声明|方法说明)(?:[：:]|\*\*[：:])/.test(block.trim());
  }).join("\n\n");
}
function reportCitationData(text, metadata = {}) {
  const ids = [], entries = [], numbers = new Map();
  const token = "(?:[EM]_[0-9a-f]{8,32}(?:…|\\.{3})?|chunk_\\d{1,6})";
  const pattern = new RegExp("`(" + token + ")`|(?<!\\w)" + token + "(?!\\w)", "g");
  let fence = null;
  const content = text.split(/(<!--[\s\S]*?-->|\r?\n)/).map(line => {
    if (line.startsWith("<!--")) return line;
    const marker = /^\s*(`{3,}|~{3,})/.exec(line);
    if (marker) { fence = fence === marker[1][0] ? null : fence || marker[1][0]; return line; }
    if (fence) return line;
    return line.replace(/chunk_\d{1,6}(?:\/\d{1,6})+/g, group => group.replaceAll("/", "/chunk_")).replace(pattern, (match, quoted) => {
      const raw = quoted || match;
      const description = metadata.references?.[raw] || { kind: "evidence", canonical: raw, status: /^E_[0-9a-f]{32}$/.test(raw) ? "resolved" : "missing" };
      const id = description.canonical || raw, key = `${description.kind}:${id}`;
      if (!numbers.has(key)) { ids.push(id); entries.push(description); numbers.set(key, ids.length); }
      return `[^e${numbers.get(key)}]`;
    });
  }).join("");
  return { content, ids, entries };
}
function formatReportDownload(text, metadata = {}) {
  const { content, ids, entries } = reportCitationData(reportBody(text), metadata);
  if (!ids.length) return content;
  return content.trimEnd() + "\n\n## 引用证据\n\n" + ids.map((id, index) => {
    const entry = entries[index], label = entry.status !== "resolved" ? `待核对：${id}` : entry.kind === "chunk" ? `对话片段 ${Number(id.slice(6))}` : id;
    return `[^e${index + 1}]: \`${label}\``;
  }).join("\n") + "\n";
}
function createReportCitations(ids, entries, metadata, container) {
  const dialog = element("dialog", undefined, "citation-dialog"), header = element("div", undefined, "citation-header");
  const title = element("h3", "引用原文"), close = element("button", "关闭", "text-button");
  const source = element("div", undefined, "evidence-source");
  dialog.setAttribute("aria-label", "引用证据"); close.type = "button";
  const context = { ids, entries, revision: metadata.ledger_revision, dialog, title, source, job: state.selected, report: state.outputRequest, sequence: 0 };
  close.addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => { context.sequence += 1; });
  header.append(title, close); dialog.append(header, source); container.append(dialog);
  return context;
}
async function openReportCitation(number, context) {
  const { job, report, ids, dialog, source } = context;
  if (job !== state.selected || report !== state.outputRequest) return;
  const reference = ids[number - 1], entry = context.entries[number - 1], sequence = ++context.sequence;
  const current = () => job === state.selected && report === state.outputRequest && sequence === context.sequence && dialog.open && source.isConnected;
  context.title.textContent = `${({ evidence: "证据", message: "消息", chunk: "对话片段" })[entry.kind]} ${number} · 原文`;
  source.textContent = "正在回查原文…";
  if (!dialog.open) dialog.showModal();
  if (entry.status !== "resolved") { source.textContent = "未找到可唯一定位的来源，这条引用待核对。"; return; }
  try {
    const data = await post("/api/evidence/reference", { id: job, reference, revision: context.revision });
    if (!current()) return;
    if (data.job_id !== job || data.ledger_revision !== context.revision || data.reference.canonical !== reference) throw new Error("引用版本已改变，请重新打开报告。");
    const row = data.reference;
    if (row.kind === "evidence" && row.grounding_status === "not_applicable") {
      source.textContent = ["存疑记录", row.payload.question, row.payload.reason].filter(Boolean).join(" · ");
      return;
    }
    renderEvidenceSource(row, source);
  } catch (error) { if (current()) source.textContent = error.message || "回查失败，请重试。"; }
}
function appendReportText(node, text, citations) {
  text.split(/(\[\^e\d+\])/g).forEach(part => {
    const match = /^\[\^e(\d+)\]$/.exec(part), number = match ? Number(match[1]) : 0;
    if (citations && number > 0 && number <= citations.ids.length) {
      const button = element("button", `[${number}]`, "report-citation");
      button.type = "button"; button.setAttribute("aria-label", `查看引用 ${number} 原文`); button.setAttribute("aria-haspopup", "dialog");
      button.addEventListener("click", () => openReportCitation(number, citations)); node.append(button);
    } else node.append(document.createTextNode(part));
  });
}
// 仅渲染安全文本，不执行报告中的 HTML 或加载外部资源。
function inlineText(node, text, citations) {
  text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*|【(?:事实|推断|假设|存疑)(?:[+＋\/／、](?:事实|推断|假设|存疑))*】)/g).forEach(part => {
    if (part.startsWith("**") && part.endsWith("**")) { const strong = element("strong"); inlineText(strong, part.slice(2, -2), citations); node.append(strong); }
    else if (part.startsWith("`") && part.endsWith("`")) { const code = element("code"); appendReportText(code, part.slice(1, -1), citations); node.append(code); }
    else if (part.startsWith("*") && part.endsWith("*")) { const emphasis = element("em"); appendReportText(emphasis, part.slice(1, -1), citations); node.append(emphasis); }
    else if (/^【.+】$/.test(part)) {
      const label = part.slice(1, -1), kind = ({ 事实: "fact", 推断: "inference", 假设: "hypothesis", 存疑: "uncertainty" })[label.split(/[+＋\/／、]/)[0]];
      node.append(element("span", label, `claim-label claim-${kind}`));
    }
    else appendReportText(node, part, citations);
  });
}
function renderMarkdown(text, container, metadata = {}) {
  const { content, ids, entries } = reportCitationData(reportBody(text).replace(/<!--[\s\S]*?-->/g, ""), metadata);
  const citations = ids.length ? createReportCitations(ids, entries, metadata, container) : null;
  const lines = content.split(/\r?\n/);
  let code = null, fence = null, quote = null, paragraph = null, lists = [];
  const tableLine = line => /^\s*\|.*\|\s*$/.test(line);
  const tableRow = (line, heading) => {
    const row = element("tr");
    line.trim().slice(1, -1).split("|").forEach(cell => {
      const node = element(heading ? "th" : "td");
      if (heading) node.setAttribute("scope", "col");
      inlineText(node, cell.trim(), citations); row.append(node);
    });
    return row;
  };
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index], marker = /^\s*(`{3,}|~{3,})/.exec(line);
    if (code) {
      if (marker && marker[1][0] === fence[0] && marker[1].length >= fence.length) { code = null; fence = null; }
      else code.textContent += line + "\n";
      continue;
    }
    if (marker) { code = element("pre", ""); fence = marker[1]; container.append(code); lists = []; quote = paragraph = null; continue; }
    if (!line.trim()) { quote = paragraph = null; continue; }
    if (tableLine(line) && /^\s*\|[\s:|\-]+\|\s*$/.test(lines[index + 1] || "")) {
      const wrap = element("div", undefined, "table-wrap"), table = element("table"), head = element("thead"), body = element("tbody");
      head.append(tableRow(line, true)); index += 1;
      while (index + 1 < lines.length && tableLine(lines[index + 1])) body.append(tableRow(lines[++index], false));
      table.append(head, body); wrap.append(table); container.append(wrap); lists = []; quote = paragraph = null; continue;
    }
    const item = /^(\s*)([-+*]|\d+[.)])\s+(.*)/.exec(line);
    if (item) {
      const indent = item[1].replace(/\t/g, "    ").length, tag = /^\d/.test(item[2]) ? "ol" : "ul";
      while (lists.length && (indent < lists.at(-1).indent || (indent === lists.at(-1).indent && tag !== lists.at(-1).tag))) lists.pop();
      if (!lists.length || indent > lists.at(-1).indent) {
        const list = element(tag);
        if (tag === "ol") list.setAttribute("start", String(parseInt(item[2], 10)));
        (lists.length ? lists.at(-1).last : container).append(list); lists.push({ indent, tag, list, last: null });
      }
      const node = element("li"); inlineText(node, item[3], citations); lists.at(-1).list.append(node); lists.at(-1).last = node; quote = paragraph = null; continue;
    }
    lists = [];
    const heading = /^(#{1,6})\s+(.*)/.exec(line);
    if (heading) { const node = element(`h${heading[1].length}`); inlineText(node, heading[2], citations); container.append(node); quote = paragraph = null; }
    else if (/^\s*>/.test(line)) {
      if (!quote) { quote = element("blockquote"); container.append(quote); }
      else quote.append(element("br"));
      inlineText(quote, line.replace(/^\s*>\s?/, ""), citations); paragraph = null;
    } else {
      if (!paragraph) { paragraph = element("p"); container.append(paragraph); }
      else paragraph.append(document.createTextNode(" "));
      inlineText(paragraph, line, citations); quote = null;
    }
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
    if (output.kind === "markdown") renderMarkdown(output.content, content, output.citations);
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
async function downloadOutput(key) {
  try {
    const output = await api(`/api/outputs/${state.selected}/${key}`);
    const content = output.kind === "markdown" && ["analysis", "persona"].includes(key) ? formatReportDownload(output.content, output.citations) : output.content;
    const url = URL.createObjectURL(new Blob([content], { type: "text/plain;charset=utf-8" }));
    const link = element("a"); link.href = url; link.download = output.name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { showError("#report-error", error); }
}
$("#download-button").addEventListener("click", () => downloadOutput(state.output));
$("#new-analysis").addEventListener("click", () => {
  state.setup = true; document.body.classList.remove("reading-mode"); $("#input-panel").open = true;
  $("#input-panel").scrollIntoView({ block: "start" }); $("#conversation").focus();
});
async function init() {
  try {
    state.config = await api("/api/bootstrap"); state.token = state.config.token;
    $("#model-name").textContent = state.config.model; $("#config-status").textContent = state.config.configured ? "已配置" : "未配置密钥";
    $("#config-hint").textContent = state.config.configured ? state.config.provider : "填写 .env 后可使用完整分析。";
    await loadFiles(); await refreshJobs(); poll();
  } catch (_) { showError("#connection-error", "未连接到分析服务。请从项目根目录运行 python agent/web_app.py，再访问此页面。静态预览无法执行分析。"); $("#config-status").textContent = "未连接"; }
}
init();
