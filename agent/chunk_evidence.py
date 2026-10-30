"""模型证据规范化、校验、账本与分块缓存。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from llm_client import call_llm_with_retry_result
from evidence_validation import (validate_evidence, balanced_items, SUMMARY_LIMIT,
                                 EXTRACTION_SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION)
from evidence_grounding import (grounding_issues, validate_grounded_document, GROUNDING_VERSION, LEDGER_VERSION)
from privacy_redaction import current_redactor
from analysis_types import ChunkMetadata, RunPaths


CHUNK_OUTPUT_TOKENS = 1200


EVIDENCE_SUMMARY_MAX_ITEMS_PER_TYPE = SUMMARY_LIMIT


CHUNK_EVIDENCE_SYSTEM_PROMPT = """You extract structured evidence from one chat chunk.
Return only one valid JSON object. Do not write Markdown, code fences, prose, or explanations.
Use short evidence quotes. Do not assume gender, romance, ambiguity, or relationship goals.
For analytic descriptions, refer to the other person as "对方/此人" only; do not use 他/她/ta unless it appears inside a direct quote."""


class IncompleteChunkOutput(RuntimeError):
    """Raised when a chunk response cannot be used as complete evidence."""


def iter_balanced_json_objects(text: str) -> list[str]:
    objects = []
    start = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start:index + 1])
                start = None

    return objects


def extract_json_block(text: str) -> dict:
    """Extract a structured JSON object from an LLM response."""
    matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    candidates = [match.strip() for match in reversed(matches) if match.strip().startswith("{")]
    if not candidates:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)
        candidates.extend(reversed(iter_balanced_json_objects(text)))

    errors = []
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            errors.append(str(exc))

    detail = "; ".join(errors) if errors else "未找到 JSON 对象"
    raise ValueError(f"无法解析结构化证据 JSON: {detail}")


def ensure_list(value) -> list:
    return value if isinstance(value, list) else []


def build_fallback_evidence(chunk_meta: dict, summary: str, error: str | None = None) -> dict:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "chunk_index": chunk_meta["chunk_index"],
        "message_range": {
            "start": chunk_meta["start_message"],
            "end": chunk_meta["end_message"],
        },
        "time_range": {
            "start": chunk_meta["start_time"],
            "end": chunk_meta["end_time"],
        },
        "events": [],
        "relation_signals": [],
        "persona_signals": [],
        "counter_evidence": [],
        "uncertainties": [{
            "question": "结构化证据未能从模型输出中可靠解析",
            "reason": error or "缺少合法 JSON 证据块",
        }],
        "raw_markdown_summary": summary,
        "parse_status": "fallback",
    }


def normalize_chunk_evidence(summary: str, chunk_meta: dict) -> dict:
    try:
        data = extract_json_block(summary)
        data = current_redactor().tree(data)
        validate_evidence(data)
    except ValueError as exc:
        return build_fallback_evidence(chunk_meta, summary, str(exc))

    normalized = {
        "schema_version": data.get("schema_version") or EVIDENCE_SCHEMA_VERSION,
        "chunk_index": chunk_meta["chunk_index"],
        "message_range": {
            "start": chunk_meta["start_message"],
            "end": chunk_meta["end_message"],
        },
        "time_range": {
            "start": chunk_meta["start_time"],
            "end": chunk_meta["end_time"],
        },
        "events": data["events"],
        "relation_signals": data["relation_signals"],
        "persona_signals": data["persona_signals"],
        "counter_evidence": data["counter_evidence"],
        "uncertainties": data["uncertainties"],
        "parse_status": "ok",
        "evidence_status": "signals" if any(data[key] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')) else "insufficient",
    }
    return normalized


def is_complete_chunk_evidence(evidence: dict) -> bool:
    if not isinstance(evidence, dict) or evidence.get("parse_status") != "ok":
        return False
    try:
        validate_evidence(evidence)
    except ValueError:
        return False
    return True


def generate_chunk_analysis(
    config,
    prompt: str,
    timeout: int,
    chunk_meta: dict,
    initial_max_tokens: int = CHUNK_OUTPUT_TOKENS,
) -> tuple[str, dict, int]:
    max_tokens = initial_max_tokens
    last_error = ""
    for attempt in range(1, 3):
        result = call_llm_with_retry_result(
            config,
            CHUNK_EVIDENCE_SYSTEM_PROMPT,
            prompt,
            timeout,
            max_tokens=max_tokens,
        )
        summary = result.content
        evidence = normalize_chunk_evidence(summary, chunk_meta)
        if is_complete_chunk_evidence(evidence):
            return summary, evidence, result.max_tokens_used
        last_error = "缺少合法结构化 JSON"
        print(f"- 分块输出不完整，准备重试: {last_error}")
        max_tokens = min(max_tokens * 2, 8000)
    raise IncompleteChunkOutput(f"分块 {chunk_meta['chunk_index']} 输出不完整: {last_error}")


def write_chunk_evidence(evidence_dir: Path, evidence: dict) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_file = evidence_dir / f"evidence_{evidence['chunk_index']:03d}.json"
    evidence_file.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence_file


def compact_text(value, max_chars: int = 240) -> str:
    text = str(value or "").replace("\r\n", " ").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def format_evidence_summary(evidence_docs: list[dict]) -> str:
    schema = LEDGER_VERSION if evidence_docs and all(doc.get('grounding_version') == GROUNDING_VERSION for doc in evidence_docs) else EVIDENCE_SCHEMA_VERSION
    sections = [
        ("events", "事件证据"),
        ("relation_signals", "关系建模证据"),
        ("persona_signals", "人物建模证据"),
        ("counter_evidence", "反证与边界"),
        ("uncertainties", "存疑问题"),
    ]
    lines = [
        "## 结构化证据摘要",
        f"- schema: {schema}",
        f"- 分块数: {len(evidence_docs)}",
        f"- 解析失败分块数: {sum(1 for doc in evidence_docs if doc.get('parse_status') != 'ok')}",
        "- 说明: 以下摘要来自每个分块末尾的结构化 JSON 证据块，用于最终关系建模、人物侧写和后续 RAG。",
        "- 原文绑定仅核对文字出处；存疑不是已证实事实，不能据此证明解释正确。",
    ]
    for key, title in sections:
        qualified = [{**doc, key: [item for item in ensure_list(doc.get(key))
                                  if doc.get('grounding_version') != GROUNDING_VERSION
                                  or item.get('grounding', {}).get('status') in ('matched', 'not_applicable')]}
                     for doc in evidence_docs]
        items = balanced_items(qualified, key, EVIDENCE_SUMMARY_MAX_ITEMS_PER_TYPE)
        lines.append(f"\n### {title}")
        total = sum(len(ensure_list(doc.get(key))) for doc in evidence_docs)
        usable = sum(len(doc[key]) for doc in qualified)
        lines.append(f'- 总数 {total}；选取 {len(items)}；省略 {total - len(items)}。完整条目保留于结构化证据总表。')
        if usable < total:
            lines.append(f'- 未匹配或歧义条目 {total - usable} 条已隔离，不作为报告证据。')
        if not items:
            lines.append("- 暂无结构化条目。")
            continue
        for item in items:
            chunk = item.get("chunk_index", "?")
            chunk_label = f"chunk_{chunk:03d}" if isinstance(chunk, int) else f"chunk_{chunk}"
            time_range = item.get("time_range", {})
            time_label = item.get("time") or f"{time_range.get('start', '?')} ~ {time_range.get('end', '?')}"
            refs = item.get('grounding', {}).get('source_refs', [])
            if refs:
                time_label = refs[0]['time']
            if key == "relation_signals":
                desc = f"{item.get('model', '未标注模型')} / {item.get('direction', 'mixed')}: {compact_text(item.get('signal', ''))}"
            elif key == "persona_signals":
                desc = f"{item.get('trait', '未标注维度')}: {compact_text(item.get('signal', ''))}"
            elif key == "counter_evidence":
                desc = f"反驳 {item.get('against', '未标注判断')}: {compact_text(item.get('evidence', ''))}"
            elif key == "uncertainties":
                desc = f"{compact_text(item.get('question', ''))}；原因：{compact_text(item.get('reason', ''))}"
            else:
                desc = compact_text(item.get("summary", ""))
            evidence = compact_text(item.get("evidence", ""))
            confidence = item.get("confidence", "")
            suffix = f"；证据：{evidence}" if evidence else ""
            conf = f"；可信度：{confidence}" if confidence else ""
            evidence_id = item.get('evidence_id', '未核验旧条目')
            location = f"；消息 {refs[0]['message_id']} / 原始序号 {refs[0]['original_index']}" if refs else ""
            lines.append(f"- {evidence_id} / {chunk_label} / {time_label}: {desc}{suffix}{conf}{location}")
    return "\n".join(lines)


def write_evidence_ledger(
    evidence_docs: list[dict],
    ledger_file: Path,
    summary_file: Path,
    messages: list[dict],
    redaction: dict | None = None,
) -> str:
    ledger = {"schema_version": LEDGER_VERSION, "grounding_version": GROUNDING_VERSION,
              "chunk_count": len(evidence_docs), "documents": evidence_docs,
              "source_id": messages[0]['source_id'], "source_messages": messages,
              "grounding_issues": grounding_issues(evidence_docs)}
    if redaction is not None:
        ledger['redaction'] = {**redaction, 'source_id': ledger['source_id']}
    ledger_file.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = format_evidence_summary(evidence_docs)
    summary_file.write_text(summary + "\n", encoding="utf-8")
    return summary


def load_existing_chunk_result(run_paths: RunPaths, chunk_index: int, expected_meta: ChunkMetadata | None = None,
                               expected_messages: list[dict] | None = None) -> tuple[str, dict] | None:
    chunk_file = run_paths["chunk_dir"] / f"chunk_{chunk_index:03d}.md"
    evidence_file = run_paths["evidence_dir"] / f"evidence_{chunk_index:03d}.json"
    if not chunk_file.exists() or not evidence_file.exists():
        return None
    summary = chunk_file.read_text(encoding="utf-8").strip()
    evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    if not is_complete_chunk_evidence(evidence):
        print(f"- 已有分块 {chunk_index} 输出不完整，将重新生成")
        return None
    if expected_meta is not None:
        expected_range = {'start': expected_meta['start_message'], 'end': expected_meta['end_message']}
        expected_time = {'start': expected_meta['start_time'], 'end': expected_meta['end_time']}
        if evidence.get('chunk_index') != chunk_index or evidence.get('message_range') != expected_range or evidence.get('time_range') != expected_time:
            raise ValueError(f'缓存分块 {chunk_index} 的消息范围或时间范围与当前输入不一致，拒绝续跑。')
    if expected_messages is not None:
        validate_grounded_document(evidence, expected_messages)
    return summary, evidence
