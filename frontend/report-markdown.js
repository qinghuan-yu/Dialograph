"use strict";
const ReportMarkdown = (() => {
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
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
  const { content, ids, entries } = reportCitationData(reportBody(text.replace(/<!--\s*END_(?:FINAL_REPORT|PERSONA_REPORT)\s*-->/g, "")), metadata);
  if (!ids.length) return content;
  return content.trimEnd() + "\n\n## 引用证据\n\n" + ids.map((id, index) => {
    const entry = entries[index], label = entry.status !== "resolved" ? `待核对：${id}` : entry.kind === "chunk" ? `对话片段 ${Number(id.slice(6))}` : id;
    return `[^e${index + 1}]: \`${label}\``;
  }).join("\n") + "\n";
}
function appendReportText(node, text, citations) {
  text.split(/(\[\^e\d+\])/g).forEach(part => {
    const match = /^\[\^e(\d+)\]$/.exec(part), number = match ? Number(match[1]) : 0;
    if (citations && number > 0 && number <= citations.ids.length) {
      const button = element("button", `[${number}]`, "report-citation");
      button.type = "button"; button.setAttribute("aria-label", `查看引用 ${number} 原文`); button.setAttribute("aria-haspopup", "dialog");
      button.addEventListener("click", () => citations.open(number, button)); node.append(button);
    } else node.append(document.createTextNode(part));
  });
}
// 仅渲染安全文本，不执行报告中的 HTML 或加载外部资源。
function inlineText(node, text, citations) {
  let plain = "";
  const flush = () => { if (plain) appendReportText(node, plain, citations); plain = ""; };
  for (let index = 0; index < text.length;) {
    if (text[index] === "\\" && /[\\`*_{}\[\]()#+\-.!|>]/.test(text[index + 1] || "")) {
      plain += text[index + 1]; index += 2; continue;
    }
    const remaining = text.slice(index), code = /^(`+)([\s\S]*?)\1(?!`)/.exec(remaining);
    if (code) {
      flush(); const span = element("code"); appendReportText(span, code[2], citations); node.append(span); index += code[0].length; continue;
    }
    const label = /^【((?:原话概括|事实|推断|假设|存疑)(?:[+＋\/／、](?:原话概括|事实|推断|假设|存疑))*)】/.exec(remaining);
    if (label) {
      flush(); const kind = ({ 原话概括: "fact", 事实: "fact", 推断: "inference", 假设: "hypothesis", 存疑: "uncertainty" })[label[1].split(/[+＋\/／、]/)[0]];
      node.append(element("span", label[1], `claim-label claim-${kind}`)); index += label[0].length; continue;
    }
    const delimiter = remaining.startsWith("**") ? "**" : remaining.startsWith("*") ? "*" : null;
    if (delimiter) {
      const end = emphasisEnd(text, index + delimiter.length, delimiter);
      if (end > index + delimiter.length) {
        flush(); const span = element(delimiter.length === 2 ? "strong" : "em");
        inlineText(span, text.slice(index + delimiter.length, end), citations); node.append(span); index = end + delimiter.length; continue;
      }
    }
    plain += text[index++];
  }
  flush();
}
function emphasisEnd(text, start, delimiter) {
  let nestedEmphasis = false;
  for (let index = start; index < text.length; index += 1) {
    if (text[index] === "\\") { index += 1; continue; }
    const code = /^(`+)([\s\S]*?)\1(?!`)/.exec(text.slice(index));
    if (code) { index += code[0].length - 1; continue; }
    if (text[index] !== "*") continue;
    const run = /^\*+/.exec(text.slice(index))[0].length;
    if (delimiter === "**" && run === 1) { nestedEmphasis = !nestedEmphasis; continue; }
    if (delimiter === "**" && run >= 2) return index + (nestedEmphasis && run >= 3 ? 1 : 0);
    if (delimiter === "*" && run === 1) return index;
    if (delimiter === "*" && run >= 2) {
      const nested = emphasisEnd(text, index + 2, "**");
      if (nested >= 0) { index = nested + 1; continue; }
      if (run % 2) return index;
    }
  }
  return -1;
}
function tableCells(line) {
  const cells = []; let cell = "", fence = "";
  const text = line.trim().slice(1, -1);
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === "\\" && index + 1 < text.length) { cell += text[index] + text[++index]; continue; }
    if (text[index] === "`") {
      const run = /^`+/.exec(text.slice(index))[0];
      if (run === fence) fence = "";
      else if (!fence && text.indexOf(run, index + run.length) >= 0) fence = run;
      cell += run; index += run.length - 1; continue;
    }
    if (text[index] === "|" && !fence) { cells.push(cell); cell = ""; }
    else cell += text[index];
  }
  cells.push(cell); return cells;
}
function renderMarkdown(text, container, metadata = {}, createCitations) {
  const { content, ids, entries } = reportCitationData(reportBody(text).replace(/<!--[\s\S]*?-->/g, ""), metadata);
  const citations = ids.length ? createCitations(ids, entries, metadata, container) : null;
  const lines = content.split(/\r?\n/);
  let code = null, fence = null, quote = null, paragraph = null, lists = [], target = container;
  const tableLine = line => /^\s*\|.*\|\s*$/.test(line);
  const tableRow = (line, heading) => {
    const row = element("tr");
    tableCells(line).forEach(cell => {
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
    if (marker) { code = element("pre", ""); fence = marker[1]; target.append(code); lists = []; quote = paragraph = null; continue; }
    if (!line.trim()) { quote = paragraph = null; continue; }
    if (/^\s{0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$/.test(line)) {
      target.append(element("hr")); lists = []; quote = paragraph = null; continue;
    }
    if (tableLine(line) && /^\s*\|[\s:|\-]+\|\s*$/.test(lines[index + 1] || "")) {
      const wrap = element("div", undefined, "table-wrap"), table = element("table"), head = element("thead"), body = element("tbody");
      head.append(tableRow(line, true)); index += 1;
      while (index + 1 < lines.length && tableLine(lines[index + 1])) body.append(tableRow(lines[++index], false));
      table.append(head, body); wrap.append(table); target.append(wrap); lists = []; quote = paragraph = null; continue;
    }
    const item = /^(\s*)([-+*]|\d+[.)])\s+(.*)/.exec(line);
    if (item) {
      const indent = item[1].replace(/\t/g, "    ").length, tag = /^\d/.test(item[2]) ? "ol" : "ul";
      while (lists.length && (indent < lists.at(-1).indent || (indent === lists.at(-1).indent && tag !== lists.at(-1).tag))) lists.pop();
      if (!lists.length || indent > lists.at(-1).indent) {
        const list = element(tag);
        if (tag === "ol") list.setAttribute("start", String(parseInt(item[2], 10)));
        (lists.length ? lists.at(-1).last : target).append(list); lists.push({ indent, tag, list, last: null });
      }
      const node = element("li"); inlineText(node, item[3], citations); lists.at(-1).list.append(node); lists.at(-1).last = node; quote = paragraph = null; continue;
    }
    lists = [];
    const heading = /^(#{1,6})\s+(.*)/.exec(line);
    if (heading) {
      const level = heading[1].length;
      if (level <= 2) target = container;
      if (level === 2 && heading[2].trim() === "原文事实") {
        const details = element("details", undefined, "report-facts"), summary = element("summary");
        summary.append(element("h2", "原文事实"), element("span", "查看原话与出处", "summary-meta"));
        target = element("div", undefined, "report-facts-body");
        details.append(summary, target); container.append(details);
      } else {
        const node = element(`h${level}`); inlineText(node, heading[2], citations); target.append(node);
      }
      quote = paragraph = null;
    }
    else if (/^\s*>/.test(line)) {
      if (!quote) { quote = element("blockquote"); target.append(quote); }
      else quote.append(element("br"));
      inlineText(quote, line.replace(/^\s*>\s?/, ""), citations); paragraph = null;
    } else {
      if (!paragraph) { paragraph = element("p"); target.append(paragraph); }
      else paragraph.append(document.createTextNode(" "));
      inlineText(paragraph, line, citations); quote = null;
    }
  }
}
return { render: renderMarkdown, formatDownload: formatReportDownload };
})();
