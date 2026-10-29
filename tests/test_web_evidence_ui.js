"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function harness() {
  const created = [], nodes = new Map();
  class Node {
    constructor(tag = "div") { this.tag = tag; this.children = []; this.text = ""; this.dataset = {}; this.attributes = {}; this.value = ""; this.isConnected = true; }
    set textContent(value) { this.text = String(value); this.children = []; }
    get textContent() { return this.text + this.children.map(node => typeof node === "string" ? node : node.textContent).join(""); }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.text = ""; this.children = children; }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener() {}
    querySelectorAll() { return []; }
  }
  const get = selector => { if (!nodes.has(selector)) nodes.set(selector, new Node()); return nodes.get(selector); };
  const context = vm.createContext({
    document: { querySelector: get, querySelectorAll: () => [],
      createElement: tag => { const node = new Node(tag); created.push(node); return node; },
      createTextNode: text => { const node = new Node("#text"); node.textContent = text; return node; } },
    fetch: () => new Promise(() => {}), setTimeout: () => 0,
    localStorage: { getItem() {}, setItem() {} }, console,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../frontend/app.js"), "utf8"), context);
  return { context, get, created, Node,
    run: code => vm.runInContext(code, context),
    reply: data => ({ ok: true, status: 200, json: async () => data }) };
}

function result(job, revision = "r1", label = "合成证据") {
  return { job_id: job, ledger_revision: revision, matches: {
    support: [{ evidence_id: "E_" + "1".repeat(32), grounding_status: "matched", category: "events",
      payload: { summary: label }, quote: "原文", occurrences: [] }], counter: [], mixed: [], uncertainty: [] } };
}

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
  assert.match(h.get("#evidence-results").textContent, /反证/);
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
