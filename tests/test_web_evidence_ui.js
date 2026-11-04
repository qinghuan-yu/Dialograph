"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function harness() {
  const created = [], nodes = new Map(), downloads = [];
  class Node {
    constructor(tag = "div") { this.tag = tag; this.children = []; this.text = ""; this.dataset = {}; this.attributes = {}; this.style = {}; this.value = ""; this.isConnected = true; this.listeners = {}; this.classList = { toggle() {}, add() {}, remove() {} }; this.scrollTop = 0; this.scrollHeight = 0; this.clientHeight = 0; }
    set textContent(value) { this.text = String(value); this.children = []; }
    get textContent() { return this.text + this.children.map(node => typeof node === "string" ? node : node.textContent).join(""); }
    append(...children) { children.forEach(child => { child.parentNode = this; }); this.children.push(...children); }
    remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(child => child !== this); }
    getBoundingClientRect() { return { left: 20, right: 620, top: 20, bottom: 500 }; }
    focus() { this.focused = true; }
    scrollIntoView() { this.scrolled = true; }
    replaceChildren(...children) { this.text = ""; this.children = children; }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener(name, listener) { this.listeners[name] = listener; }
    showModal() { this.open = true; }
    close() { this.open = false; this.listeners.close?.(); }
    click() { return this.listeners.click?.(); }
    querySelectorAll() { return []; }
  }
  const get = selector => { if (!nodes.has(selector)) nodes.set(selector, new Node()); return nodes.get(selector); };
  const context = vm.createContext({
    document: { body: get("body"), querySelector: get, querySelectorAll: () => [],
      createElement: tag => { const node = new Node(tag); created.push(node); return node; },
      createTextNode: text => { const node = new Node("#text"); node.textContent = text; return node; } },
    fetch: () => new Promise(() => {}), setTimeout: () => 0, Blob,
    URL: { createObjectURL: blob => { downloads.push(blob); return "blob:local-download"; }, revokeObjectURL() {} },
    localStorage: { getItem() {}, setItem() {} }, console,
  });
  for (const file of ["report-markdown.js", "app.js"]) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, "../frontend", file), "utf8"), context);
  }
  return { context, get, created, downloads, Node,
    run: code => vm.runInContext(code, context),
    reply: data => ({ ok: true, status: 200, json: async () => data }) };
}

function result(job, revision = "r1", label = "合成证据") {
  return { job_id: job, ledger_revision: revision, matches: {
    support: [{ evidence_id: "E_" + "1".repeat(32), grounding_status: "matched", category: "events",
      payload: { summary: label }, quote: "原文", occurrences: [] }], counter: [], mixed: [], uncertainty: [] } };
}

test("切换报告读取期间保留原文布局，新报告就绪后再替换", async () => {
  const h = harness(), content = h.get("#report-content"), old = new h.Node("p");
  old.textContent = "原报告正文"; content.append(old);
  h.run("state.selected = 'task'; state.output = 'analysis';");
  let finish;
  h.context.fetch = () => new Promise(resolve => { finish = resolve; });
  const pending = h.context.loadOutput("persona");
  assert.equal(content.children[0], old, "请求等待期间不能清空长报告，避免页面高度骤降");
  assert.equal(content.attributes["aria-busy"], "true");
  finish(h.reply({ kind: "markdown", content: "# 新侧写\n\n新报告正文" }));
  await pending;
  assert.match(content.textContent, /新侧写.*新报告正文/);
  assert.doesNotMatch(content.textContent, /原报告正文/);
  assert.equal(content.attributes["aria-busy"], "false");
});

test("快速切换报告时只展示最后一次结果，旧响应不结束新请求的加载状态", async () => {
  const h = harness(), content = h.get("#report-content"), replies = [];
  content.append(new h.Node("p"));
  h.run("state.selected = 'task'; state.output = 'analysis';");
  h.context.fetch = () => new Promise(resolve => replies.push(resolve));
  const first = h.context.loadOutput("persona"), second = h.context.loadOutput("analysis");
  replies[0](h.reply({ kind: "markdown", content: "旧侧写" })); await first;
  assert.equal(content.attributes["aria-busy"], "true");
  assert.doesNotMatch(content.textContent, /旧侧写/);
  replies[1](h.reply({ kind: "markdown", content: "最新关系报告" })); await second;
  assert.match(content.textContent, /最新关系报告/);
  assert.equal(content.attributes["aria-busy"], "false");
});

test("重复点击当前已显示报告时不重载，失败后可以重试并结束加载状态", async () => {
  const h = harness(), content = h.get("#report-content");
  content.append(new h.Node("p")); h.get("#download-button").disabled = false;
  h.run("state.selected = 'task'; state.output = 'analysis';");
  h.context.fetch = () => { throw new Error("不应重载当前报告"); };
  await h.context.loadOutput("analysis");
  assert.equal(h.get("#report-error").textContent, "");
  h.context.fetch = async () => ({ ok: false, status: 400, json: async () => ({ error: "合成读取错误" }) });
  await h.context.loadOutput("persona");
  assert.equal(content.attributes["aria-busy"], "false");
  assert.match(h.get("#report-error").textContent, /合成读取错误/);
  h.context.fetch = async () => h.reply({ kind: "markdown", content: "重试成功" });
  await h.context.loadOutput("persona");
  assert.match(content.textContent, /重试成功/);
});

test("切换任务后，旧检索响应不能覆盖新任务", async () => {
  const h = harness(), id = "a".repeat(32);
  assert.equal(typeof h.context.searchEvidence, "function");
  h.run(`state.selected = '${id}'; resetEvidence('${id}');`);
  h.get("#evidence-query").value = "修复意愿";
  let finish;
  h.context.fetch = () => new Promise(resolve => { finish = resolve; });
  const pending = h.context.searchEvidence();
  h.run(`state.selected = '${"b".repeat(32)}'; resetEvidence(state.selected);`);
  finish(h.reply(result(id)));
  await pending;
  assert.equal(h.get("#evidence-results").textContent, "");
});

test("较早的查询晚返回时，不能覆盖较新的查询", async () => {
  const h = harness(), id = "a".repeat(32), replies = [];
  assert.equal(typeof h.context.searchEvidence, "function");
  h.run(`state.selected = '${id}'; resetEvidence('${id}');`);
  h.context.fetch = () => new Promise(resolve => replies.push(resolve));
  h.get("#evidence-query").value = "第一次";
  const first = h.context.searchEvidence();
  h.get("#evidence-query").value = "第二次";
  const second = h.context.searchEvidence();
  replies[1](h.reply(result(id, "new", "新查询结果")));
  await second;
  replies[0](h.reply(result(id, "old", "旧查询结果")));
  await first;
  assert.match(h.get("#evidence-results").textContent, /新查询结果/);
  assert.doesNotMatch(h.get("#evidence-results").textContent, /旧查询结果/);
});

test("证据中的 HTML 作为文字展示，不能创建图片或脚本", () => {
  const h = harness(), data = result("a".repeat(32), "r1", '<img src=x onerror="attack()">');
  assert.equal(typeof h.context.renderEvidenceResults, "function");
  h.context.renderEvidenceResults(data);
  assert.match(h.get("#evidence-results").textContent, /<img src=x/);
  assert.equal(h.created.some(node => ["img", "script", "iframe"].includes(node.tag)), false);
  assert.doesNotMatch(h.get("#evidence-results").textContent, /不代表|此关键词未匹配/);
});

test("开始新查询后，旧来源回查不再写入页面", async () => {
  const h = harness(), id = "a".repeat(32), target = new h.Node();
  assert.equal(typeof h.context.loadEvidenceSource, "function");
  h.run(`state.selected = '${id}'; resetEvidence('${id}'); state.evidence.revision = 'r1';`);
  let finish;
  h.context.fetch = () => new Promise(resolve => { finish = resolve; });
  const row = result(id).matches.support[0];
  const pending = h.context.loadEvidenceSource(row, target);
  h.run("state.evidence.request += 1; state.evidence.revision = null;");
  finish(h.reply({ job_id: id, ledger_revision: "r1", evidence: {
    ...row, occurrences: [{ chunk_index: 1, source_refs: [{ content: "旧来源正文", start: 0, end: 2,
      original_index: 0, position: 1, sender: "me", time: "2026-10-29 10:00:00" }] }] } }));
  await pending;
  assert.doesNotMatch(target.textContent, /旧来源正文/);
});

test("原文表情符号不影响引文高亮，消息序号从一开始显示", () => {
  const h = harness(), target = new h.Node();
  h.context.renderEvidenceSource({ occurrences: [{ chunk_index: 2, source_refs: [{
    content: '🙂我愿意和好。', start: 2, end: 6, original_index: 0, sender: 'me', time: '10:00'
  }] }] }, target);
  assert.equal(h.created.find(node => node.tag === 'mark').textContent, '愿意和好');
  assert.match(target.textContent, /原始第 1 条/);
  assert.match(target.textContent, /🙂我愿意和好。/);
});

test("报告引用按首次出现编号，粗体、行内代码和表格共用编号", () => {
  const h = harness(), target = new h.Node(), first = "E_" + "1".repeat(32), second = "E_" + "2".repeat(32);
  h.context.renderMarkdown(`正文 ${first}，**${second}**，再次引用 \`${first}\`。\n\n| 依据 |\n| --- |\n| ${first} |`, target);
  const buttons = h.created.filter(node => node.className === "report-citation");
  assert.deepEqual(buttons.map(node => node.textContent), ["[1]", "[2]", "[1]", "[1]"]);
  assert.doesNotMatch(target.textContent, /E_[0-9a-f]{32}/);
  assert.equal(buttons[0].attributes["aria-haspopup"], "dialog");
  h.context.renderMarkdown(`另一份报告 ${second}`, new h.Node());
  assert.equal(h.created.filter(node => node.className === "report-citation").at(-1).textContent, "[1]");
});

test("Markdown 下载正文使用脚注，完整编号在文末去重保留", () => {
  const h = harness(), first = "E_" + "1".repeat(32), second = "E_" + "2".repeat(32);
  const output = h.context.formatReportDownload(`正文 \`${second}\`，${first}，再次 ${second}\n<!-- END_FINAL_REPORT -->`);
  const [body, notes] = output.split("## 引用证据");
  assert.match(body, /正文 \[\^e1\]，\[\^e2\]，再次 \[\^e1\]/);
  assert.doesNotMatch(body, /E_[0-9a-f]{32}/);
  assert.ok(notes.includes(`[^e1]: \`${second}\``));
  assert.ok(notes.includes(`[^e2]: \`${first}\``));
  assert.equal(output.split(second).length - 1, 1);
  assert.doesNotMatch(output, /END_FINAL_REPORT/);
  assert.equal(h.context.formatReportDownload("普通正文。"), "普通正文。");
});

test("报告中的代码块和相似的非证据标识保持原样", () => {
  const h = harness(), id = "E_" + "1".repeat(32), target = new h.Node();
  const text = `\`\`\`\n${id}\n\`\`\`\n相似标识 ${id}a，prefix_${id}。`;
  assert.equal(h.context.formatReportDownload(text), text);
  h.context.renderMarkdown(text, target);
  assert.equal(h.created.some(node => node.className === "report-citation"), false);
  assert.match(target.textContent, new RegExp(id));
});

test("隐藏注释中的编号不影响网页和下载的可见引用顺序", () => {
  const h = harness(), hidden = "E_" + "1".repeat(32), visible = "E_" + "2".repeat(32);
  const text = `<!-- 隐藏 ${hidden} -->\n正文 ${visible}`;
  const output = h.context.formatReportDownload(text);
  assert.match(output, /正文 \[\^e1\]/);
  assert.ok(output.includes(`[^e1]: \`${visible}\``));
});

test("下载按钮实际生成脚注版 Markdown，JSON 下载保持原内容", async () => {
  const h = harness(), id = "E_" + "1".repeat(32);
  h.run("state.selected = 'synthetic-task'; state.output = 'analysis';");
  h.context.fetch = async () => h.reply({ content: `正文 ${id}`, kind: "markdown", name: "report.md" });
  await h.get("#download-button").click();
  const markdown = await h.downloads[0].text();
  assert.match(markdown, /正文 \[\^e1\]/);
  assert.ok(markdown.includes(`[^e1]: \`${id}\``));
  assert.equal(h.created.find(node => node.tag === "a").download, "report.md");
  const raw = JSON.stringify({ evidence_id: id });
  h.context.fetch = async () => h.reply({ content: raw, kind: "json", name: "evidence.json" });
  await h.get("#download-button").click();
  assert.equal(await h.downloads[1].text(), raw);
});

test("点击简短引用只发一次请求并绑定当前版本", async () => {
  const h = harness(), job = "a".repeat(32), id = "E_" + "1".repeat(32), calls = [];
  h.run(`state.selected = '${job}'; state.output = 'analysis';`);
  h.context.fetch = async (url, options) => {
    const body = JSON.parse(options.body); calls.push({ url, body });
    return h.reply({ job_id: job, ledger_revision: "r1", reference: {
      kind: "evidence", canonical: id,
      ...result(job).matches.support[0], occurrences: [{ chunk_index: 1, source_refs: [{
        content: "原文在这里", start: 0, end: 2, original_index: 0, sender: "me", time: "10:00"
      }] }] } });
  };
  h.context.renderMarkdown(`依据 ${id}`, new h.Node(), { ledger_revision: "r1" });
  const button = h.created.find(node => node.className === "report-citation");
  assert.ok(button, "长编号应成为可点击的引用按钮");
  await button.listeners.click();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/api/evidence/reference");
  assert.deepEqual(calls[0].body, { id: job, reference: id, revision: "r1" });
  assert.equal(h.created.find(node => node.tag === "dialog").open, true);
  assert.equal(h.created.find(node => node.tag === "mark").textContent, "原文");
});

test("切换报告后，旧引用的异步结果不会写入弹窗", async () => {
  const h = harness(), job = "a".repeat(32), id = "E_" + "1".repeat(32);
  h.run(`state.selected = '${job}'; state.output = 'analysis';`);
  let finish;
  h.context.fetch = () => new Promise(resolve => { finish = resolve; });
  h.context.renderMarkdown(`依据 ${id}`, new h.Node());
  const button = h.created.find(node => node.className === "report-citation");
  assert.ok(button, "报告应提供短引用按钮");
  const pending = button.listeners.click();
  h.run("state.outputRequest += 1; state.output = 'persona';");
  finish(h.reply(result(job, "r1", "旧报告证据")));
  await pending;
  assert.equal(h.created.some(node => node.tag === "mark"), false);
});

test("未知引用和存疑项不会伪造事实原文", async () => {
  for (const uncertainty of [false, true]) {
    const h = harness(), job = "a".repeat(32), id = "E_" + "1".repeat(32), calls = [];
    h.run(`state.selected = '${job}'; state.output = 'analysis';`);
    const data = { job_id: job, ledger_revision: "r1", reference: { kind: "evidence", canonical: id,
      grounding_status: "not_applicable", payload: { question: "是否持续", reason: "需要后续对话" } } };
    h.context.fetch = async url => { calls.push(url); return h.reply(data); };
    const target = new h.Node();
    h.context.renderMarkdown(`依据 ${id}`, target, { ledger_revision: "r1", references: uncertainty ? {} : {
      [id]: { kind: "evidence", status: "missing" } } });
    const button = h.created.find(node => node.className === "report-citation");
    assert.ok(button, "报告应提供短引用按钮");
    await button.listeners.click();
    assert.equal(calls.length, uncertainty ? 1 : 0);
    assert.match(target.textContent, uncertainty ? /是否持续.*需要后续对话/s : /未找到/);
    assert.equal(h.created.some(node => node.tag === "mark"), false);
  }
});

test("已唯一定位的 E/M 缩写和片段均变成短引用，同一来源共用编号", () => {
  const h = harness(), e = "E_" + "1".repeat(32), m = "M_" + "2".repeat(32);
  const references = {
    "E_11111111…": { status: "resolved", kind: "evidence", canonical: e },
    "M_22222222…": { status: "resolved", kind: "message", canonical: m },
    chunk_160: { status: "resolved", kind: "chunk", canonical: "chunk_160" }
  };
  const text = `依据 E_11111111…，消息 M_22222222…，片段 chunk_160，再次 ${e}`;
  const target = new h.Node();
  h.context.renderMarkdown(text, target, { ledger_revision: "r1", references });
  assert.deepEqual(h.created.filter(n => n.className === "report-citation").map(n => n.textContent), ["[1]", "[2]", "[3]", "[1]"]);
  assert.doesNotMatch(target.textContent, /E_|M_|chunk_/);
  const download = h.context.formatReportDownload(text, { references });
  assert.ok(download.includes(e)); assert.ok(download.includes(m));
  assert.equal(download.split("## 引用证据")[0].includes("chunk_"), false);
});

test("报告阅读隐藏固定前言，用轻量标签显示判断类别并保留正文", () => {
  const h = harness(), target = new h.Node();
  h.context.renderMarkdown("> 来源说明：内部运行参数\n\n# 报告\n\n**重要声明：** 固定前言\n\n## 互动\n\n【推断】这是需要保留的具体判断。\n\n【事实+推断】综合判断。", target);
  assert.doesNotMatch(target.textContent, /固定前言|内部运行参数|【推断】|【事实\+推断】/);
  assert.match(target.textContent, /推断.*具体判断/);
  assert.ok(h.created.some(n => n.className?.includes("claim-label")));
});

test("紧凑分块组的每一段都有独立引用，不留下内部序号", () => {
  const h = harness(), target = new h.Node(), references = {};
  for (const id of ["chunk_122", "chunk_125", "chunk_128"]) references[id] = { kind: "chunk", canonical: id, status: "resolved" };
  h.context.renderMarkdown("来源 chunk_122/125/128。", target, { references });
  assert.deepEqual(h.created.filter(n => n.className === "report-citation").map(n => n.textContent), ["[1]", "[2]", "[3]"]);
  assert.doesNotMatch(target.textContent, /125|128|chunk/);
});

test("Markdown 有序列表、嵌套列表、表头、斜体与多行引用具有相应结构", () => {
  const h = harness(), target = new h.Node();
  h.context.renderMarkdown("1. 第一点\n   - 子项 *重点*\n2. 第二点\n\n| 模型 | 依据 |\n| --- | --- |\n| 朋友 | 互动 |\n\n> 引用第一行\n> 引用第二行", target);
  assert.ok(h.created.some(n => n.tag === "ol"));
  assert.ok(h.created.find(n => n.tag === "li").children.some(n => n.tag === "ul"));
  assert.equal(h.created.filter(n => n.tag === "th").length, 2);
  assert.ok(h.created.some(n => n.tag === "em" && n.textContent === "重点"));
  assert.equal(h.created.filter(n => n.tag === "blockquote").length, 1);
});

test("零值诊断不显示，实际异常保留", () => {
  const h = harness(), job = { stats: { first_date: "2026-01-01", last_date: "2026-02-01" },
    diagnostics: { skipped_message_count: 0, invalid_time_message_count: 0 } };
  h.context.renderStats(job);
  assert.doesNotMatch(h.get("#diagnostic-summary").textContent, /跳过 0|异常 0|替换 0|undefined/);
  job.diagnostics.invalid_time_message_count = 2;
  h.context.renderStats(job);
  assert.match(h.get("#diagnostic-summary").textContent, /异常 2/);
});

test("完成任务优先阅读，两份报告与附件分开，保留附件下载", () => {
  const h = harness(), job = { id: "a".repeat(32), status: "completed", mode: "full", events: [],
    files: { analysis: { label: "关系分析" }, persona: { label: "人物侧写" }, evidence: { label: "证据附件" } } };
  h.run(`state.selected='${job.id}'; state.output='analysis';`);
  h.context.renderJob(job);
  assert.equal(h.get("#report-tabs").children.length, 2);
  assert.equal(h.get("#attachment-list").children.length, 1);
  assert.equal(h.get("#input-panel").open, false);
  assert.equal(h.get("#process-panel").open, false);
});

test("从阅读模式开始新任务时展开处理进度", () => {
  const h = harness(), job = { id: "b".repeat(32), status: "running", mode: "full", stage: "prepare", events: [], files: {} };
  h.get("#process-panel").open = false;
  h.get("#input-panel").open = true;
  h.context.renderJob(job);
  assert.equal(h.get("#process-panel").open, true);
  assert.equal(h.get("#input-panel").open, false);
});

test("Markdown 分隔线独立成块，段落不跨越列表、引用和标题合并", () => {
  const h = harness(), target = new h.Node();
  h.context.renderMarkdown("前文\n---\n后文\n***\n1. 项目\n___\n> 引用\n# 标题\n结尾", target);
  assert.deepEqual(target.children.map(n => n.tag), ["p", "hr", "p", "hr", "ol", "hr", "blockquote", "h1", "p"]);
});

test("表格中的转义竖线和代码竖线不分列，嵌套强调保留结构", () => {
  const h = harness(), target = new h.Node();
  h.context.renderMarkdown("| 内容 | 依据 |\n| --- | --- |\n| a\\|b | `x|y` |\n\n**粗体 *斜体* 【事实】 `代码`**", target);
  assert.deepEqual(h.created.filter(n => n.tag === "td").map(n => n.textContent), ["a|b", "x|y"]);
  const bold = h.created.find(n => n.tag === "strong");
  assert.ok(bold.children.some(n => n.tag === "em" && n.textContent === "斜体"));
  assert.ok(bold.children.some(n => n.tag === "code"));
  assert.ok(bold.children.some(n => n.className?.includes("claim-label")));
});

test("引用弹窗仅在矩形外点击时关闭，关闭后恢复引用焦点", async () => {
  const h = harness(), id = "E_" + "1".repeat(32);
  h.context.renderMarkdown(`引用 ${id}`, new h.Node(), { references: { [id]: { status: "missing", kind: "evidence" } } });
  const dialog = h.created.find(n => n.tag === "dialog"), button = h.created.find(n => n.className === "report-citation");
  await button.click();
  dialog.listeners.click?.({ target: dialog, clientX: 30, clientY: 30 });
  assert.equal(dialog.open, true, "对话框内部留白不关闭");
  dialog.listeners.click?.({ target: dialog, clientX: 5, clientY: 5 });
  assert.equal(dialog.open, false, "点击背景关闭");
  assert.equal(button.focused, true);
});

test("原文展开和收起按钮同步文案，来源序号直接显示", async () => {
  const h = harness();
  h.context.renderEvidenceResults(result("a".repeat(32)));
  const button = h.created.find(n => n.tag === "button" && n.textContent === "回查原文");
  const source = h.created.find(n => n.className === "evidence-source"); source.dataset.loaded = "true";
  await button.click(); assert.equal(button.textContent, "收起原文");
  await button.click(); assert.equal(button.textContent, "回查原文");
  h.context.renderEvidenceSource({ occurrences: [{ source_refs: [{ sender: "me", time: "10:00", content: "正文", original_index: 3 }] }] }, source);
  assert.match(source.textContent, /原始第 4 条/);
  assert.equal(h.created.some(n => n.className === "source-location"), false);
});

test("运行记录保留已有节点，跟随末尾但不打断向上阅读", () => {
  const h = harness(), job = { id: "logs-task", events: [{ time: "2026-11-04T10:00:00", message: "开始" }] };
  const logs = h.get("#logs"); logs.clientHeight = 100; logs.scrollHeight = 300; logs.scrollTop = 200;
  assert.equal(typeof h.context.renderLogs, "function");
  h.context.renderLogs(job); const first = logs.children[0];
  job.events.push({ time: "2026-11-04T10:00:01", message: "继续" });
  logs.scrollHeight = 400; logs.scrollTop = 300;
  h.context.renderLogs(job); assert.equal(logs.children[0], first); assert.equal(logs.scrollTop, 400);
  logs.scrollTop = 30; job.events.push({ time: "2026-11-04T10:00:02", message: "完成" });
  h.context.renderLogs(job); assert.equal(logs.scrollTop, 30);
  const retained = logs.children[1]; job.events.shift();
  h.context.renderLogs(job); assert.equal(logs.children[0], retained); assert.equal(logs.children.length, 2);
});

test("展开运行记录时可继续跟随末尾，也保留向上阅读的位置", () => {
  const h = harness(), logs = h.get("#logs"), details = h.get("#log-details");
  h.context.renderLogs({ id: "job", events: [{ time: "2026-11-04T10:00:00", message: "开始" }] });
  logs.clientHeight = 100; logs.scrollHeight = 500; details.open = true;
  details.listeners.toggle?.(); assert.equal(logs.scrollTop, 500);
  logs.scrollTop = 50; logs.listeners.scroll?.();
  details.listeners.toggle?.(); assert.equal(logs.scrollTop, 50);
});

test("Markdown 紧邻块边界与双向嵌套强调均正确结束", () => {
  const h = harness(), target = new h.Node();
  h.context.renderMarkdown("前文\n```\n代码\n```\n后文\n| 标题 |\n| --- |\n| 值 |\n尾段\n*斜体 **粗体***\n**粗体 *斜体***", target);
  assert.deepEqual(target.children.map(n => n.tag), ["p", "pre", "p", "div", "p"]);
  const outer = h.created.find(n => n.tag === "em");
  assert.ok(outer.children.some(n => n.tag === "strong" && n.textContent === "粗体"));
  assert.ok(h.created.some(n => n.tag === "strong" && n.children.some(c => c.tag === "em" && c.textContent === "斜体")));
});

test("月度图分开显示双方数量，快捷入口展开对应面板", () => {
  const h = harness();
  h.context.renderStats({ stats: { monthly_trend: [{ month: "2026-11", total: 10, my_count: 3, other_count: 7 }] } });
  assert.equal(h.created.filter(n => n.tag === "meter").length, 0);
  const bar = h.created.find(n => n.attributes.role === "img");
  assert.match(bar?.attributes["aria-label"] || "", /我 3 条.*对方 7 条/);
  assert.equal(bar.children.length, 2);
  h.get("#jump-evidence").click();
  assert.equal(h.get("#evidence-panel").open, true);
  assert.equal(h.get("#evidence-query").focused, true);
});
