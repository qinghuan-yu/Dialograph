"""连续分块、证据抽取、检查点与关系报告运行。"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from artifact_audit import audit_artifacts
from llm_client import (call_llm_with_retry_result, generate_marked_report,
                        should_retry_with_smaller_prompt, require_completion_marker,
                        IncompleteReportOutput)
from report_generation import build_final_prompt, FINAL_REPORT_MAX_OUTPUT_TOKENS
from report_output import write_report
from evidence_validation import validate_evidence, balanced_items, SUMMARY_LIMIT
from workflow_state import run_identity, resume_metadata, LIMIT_KEYS, report_identity
from run_promotion import promote_successful_run
from evidence_grounding import (ground_document, grounding_issues, validate_grounded_document,
                                GROUNDING_VERSION, LEDGER_VERSION)
from privacy_redaction import current_redactor, redact_artifact_call
from prompt_budget import text_bytes


CHUNK_MESSAGE_COUNT = 1200


CHUNK_MAX_TRANSCRIPT_BYTES = 18000


CHUNK_MIN_TRANSCRIPT_BYTES = 12000


CHUNK_AUTO_SHRINK_FACTOR = 0.55


CHUNK_AUTO_SHRINK_MAX_RESTARTS = 3


CHUNK_MAX_PROMPT_BYTES = 32000


CHUNK_MIN_PROMPT_BYTES = 18000


CHUNK_REQUEST_TIMEOUT = 180


CHUNK_REQUEST_SLEEP_SECONDS = 1.0


SUMMARY_MAX_BYTES = 120000


MERGE_MAX_PROMPT_BYTES = 30000


MERGE_MAX_OUTPUT_TOKENS = 1600


CHUNK_OUTPUT_TOKENS = 1200


EVIDENCE_SCHEMA_VERSION = "chat-evidence-v1"


EVIDENCE_SUMMARY_MAX_ITEMS_PER_TYPE = SUMMARY_LIMIT


CHUNK_EVIDENCE_SYSTEM_PROMPT = """You extract structured evidence from one chat chunk.
Return only one valid JSON object. Do not write Markdown, code fences, prose, or explanations.
Use short evidence quotes. Do not assume gender, romance, ambiguity, or relationship goals.
For analytic descriptions, refer to the other person as "对方/此人" only; do not use 他/她/ta unless it appears inside a direct quote."""


class IncompleteChunkOutput(RuntimeError):
    """Raised when a chunk response cannot be used as complete evidence."""


class RetryableChunkStageFailure(RuntimeError):
    """Raised when chunk analysis can be retried with smaller chunks."""


def chunk_messages(
    messages: list[dict],
    chunk_size: int = CHUNK_MESSAGE_COUNT,
    max_transcript_bytes: int = CHUNK_MAX_TRANSCRIPT_BYTES,
) -> list[list[dict]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if max_transcript_bytes <= 0:
        raise ValueError("chunk_max_bytes 必须大于 0")

    chunks = []
    current = []
    current_bytes = 0
    for message in messages:
        line_bytes = len(format_message_line(message, "对方").encode("utf-8")) + 1
        if current and (len(current) >= chunk_size or current_bytes + line_bytes > max_transcript_bytes):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(message)
        current_bytes += line_bytes

    if current:
        chunks.append(current)
    return chunks


def format_message_line(message: dict, other_name: str) -> str:
    prefix = "【我】" if message["sender"] == "me" else f"【{other_name}】"
    content = str(message.get("content", "")).replace("\r\n", "\\n").replace("\n", "\\n")
    return f"{message['time_str']} {prefix} {content}"


def build_chunk_manifest(chunk_sets: list[list[dict]]) -> list[dict]:
    manifest = []
    offset = 0
    for index, chunk in enumerate(chunk_sets, start=1):
        if not chunk:
            continue
        start_position = offset + 1
        end_position = offset + len(chunk)
        transcript_bytes = len(
            "\n".join(format_message_line(message, "对方") for message in chunk).encode("utf-8")
        )
        manifest.append({
            "chunk_index": index,
            "start_message": start_position,
            "end_message": end_position,
            "message_count": len(chunk),
            "start_time": chunk[0]["time_str"],
            "end_time": chunk[-1]["time_str"],
            "transcript_bytes": transcript_bytes,
        })
        offset = end_position
    return manifest


def validate_chunk_coverage(messages: list[dict], chunk_sets: list[list[dict]]) -> None:
    covered = sum(len(chunk) for chunk in chunk_sets)
    if covered != len(messages):
        raise ValueError(f"分块覆盖消息数不一致: covered={covered}, total={len(messages)}")
    if not messages:
        return
    if not chunk_sets or chunk_sets[0][0] != messages[0] or chunk_sets[-1][-1] != messages[-1]:
        raise ValueError("分块没有覆盖原始消息首尾，终止以避免生成不完整报告")


def split_chunks_by_prompt_budget(
    stats: dict,
    chunk_sets: list[list[dict]],
    max_prompt_bytes: int,
) -> list[list[dict]]:
    if max_prompt_bytes <= 0:
        raise ValueError("chunk_max_prompt_bytes 必须大于 0")

    result = []
    queue = list(chunk_sets)
    while queue:
        chunk = queue.pop(0)
        prompt_bytes = len(build_chunk_prompt(stats, 1, 1, chunk).encode("utf-8"))
        if prompt_bytes <= max_prompt_bytes or len(chunk) <= 1:
            result.append(chunk)
            continue

        midpoint = len(chunk) // 2
        left = chunk[:midpoint]
        right = chunk[midpoint:]
        queue.insert(0, right)
        queue.insert(0, left)
    return result


def format_coverage_summary(stats: dict, chunk_manifest: list[dict]) -> str:
    total_chunks = len(chunk_manifest)
    total_messages = sum(item["message_count"] for item in chunk_manifest)
    lines = [
        "## 全量覆盖说明",
        "- 本报告采用全量分块阅读：每个分块都包含连续原文消息，最终报告基于全部分块/阶段总结综合生成。",
        "- 预处理样本只用于辅助定位代表性片段，不作为唯一证据来源。",
        f"- 有效消息总数: {stats['total_messages']}",
        f"- 分块覆盖消息数: {total_messages}",
        f"- 分块数量: {total_chunks}",
    ]
    if chunk_manifest:
        lines.append(f"- 覆盖时间范围: {chunk_manifest[0]['start_time']} ~ {chunk_manifest[-1]['end_time']}")
        lines.append("- 分块索引:")
        for item in chunk_manifest:
            lines.append(
                f"  - chunk_{item['chunk_index']:03d}: "
                f"第{item['start_message']}-{item['end_message']}条，"
                f"{item['message_count']}条，"
                f"{item['start_time']} ~ {item['end_time']}"
            )
    return "\n".join(lines)


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
        "events": ensure_list(data.get("events")),
        "relation_signals": ensure_list(data.get("relation_signals")),
        "persona_signals": ensure_list(data.get("persona_signals")),
        "counter_evidence": ensure_list(data.get("counter_evidence")),
        "uncertainties": ensure_list(data.get("uncertainties")),
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
) -> str:
    ledger = {"schema_version": LEDGER_VERSION, "grounding_version": GROUNDING_VERSION,
              "chunk_count": len(evidence_docs), "documents": evidence_docs,
              "source_id": messages[0]['source_id'], "source_messages": messages,
              "grounding_issues": grounding_issues(evidence_docs)}
    ledger_file.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = format_evidence_summary(evidence_docs)
    summary_file.write_text(summary + "\n", encoding="utf-8")
    return summary


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + uuid.uuid4().hex[:8]


def build_run_paths(artifacts: dict, run_id: str | None = None) -> dict:
    run_id = run_id or make_run_id()
    run_dir = artifacts["runs_dir"] / run_id
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "merge_dir": run_dir / "merge_batches",
        "chunk_dir": run_dir / "分块分析",
        "evidence_dir": run_dir / "结构化证据",
        "chunk_manifest_file": run_dir / f"分块覆盖清单_{artifacts['safe_name']}.json",
        "coverage_summary_file": run_dir / f"全量覆盖说明_{artifacts['safe_name']}.md",
        "evidence_ledger_file": run_dir / f"结构化证据总表_{artifacts['safe_name']}.json",
        "evidence_summary_file": run_dir / f"结构化证据摘要_{artifacts['safe_name']}.md",
        "phase_summary_file": run_dir / f"阶段总结_{artifacts['safe_name']}.md",
        "analysis_file": run_dir / f"分析_{artifacts['safe_name']}.md",
        "analysis_meta_file": run_dir / f"报告来源_分析_{artifacts['safe_name']}.json",
        "run_metadata_file": run_dir / "run_metadata.json",
    }


def write_run_metadata(artifacts: dict, run_paths: dict, status: str, extra: dict | None = None) -> None:
    run_paths["run_dir"].mkdir(parents=True, exist_ok=True)
    previous = {}
    if run_paths['run_metadata_file'].exists():
        previous = json.loads(run_paths['run_metadata_file'].read_text(encoding='utf-8'))
    metadata = {
        **previous,
        "run_id": run_paths["run_id"],
        "status": status,
        "name": artifacts["name"],
        "safe_name": artifacts["safe_name"],
        "chat_file": 'input' + artifacts["chat_file"].suffix.lower(),
        "message_count": artifacts["message_count"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        metadata.update(extra)
    run_paths["run_metadata_file"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_existing_chunk_result(run_paths: dict, chunk_index: int, expected_meta: dict | None = None,
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


def build_chunk_prompt(stats: dict, chunk_index: int, total_chunks: int, chunk: list[dict]) -> str:
    other_name = stats["other_name"]
    transcript = "\n".join(f"[{message.get('message_id', '未编号')}] {format_message_line(message, other_name)}" for message in chunk)
    return f"""只阅读下面这个连续聊天分块，并抽取紧凑结构化证据。

分块: {chunk_index}/{total_chunks}
时间: {chunk[0]['time_str']} ~ {chunk[-1]['time_str']}
消息数: {len(chunk)}

硬性输出规则:
1. 只输出一个合法 JSON 对象。
2. 不要 Markdown，不要代码围栏，不要解释，不要分析正文。
3. 字符串尽量短；证据摘录不超过 60 个中文字符。
4. 最多输出: events 3 条, relation_signals 4 条, persona_signals 4 条, counter_evidence 3 条, uncertainties 3 条。
5. 不预设性别、恋爱、暧昧或关系目标；只写本分块证据。
6. 分析性表述统一使用“对方/此人/该对象/对象昵称”，禁止使用“他/她/ta”代称；只有直接引用原文时可以保留原文中的代词。
7. evidence 必须是某一条文本消息中连续、逐字一致的原话，不加引号、不改写、不拼接多条消息；展示的 \\n 在 JSON 字符串内用换行转义表示。
8. 各引文条目可增加 source_message_ids 字段，值为唯一来源消息编号的数组（如 ["M_..."]）；重复原话必须提供该编号。不要生成 evidence_id 或伪造来源。

JSON 对象格式:
{{
  "schema_version": "{EVIDENCE_SCHEMA_VERSION}",
  "chunk_index": {chunk_index},
  "time_range": {{"start": "{chunk[0]['time_str']}", "end": "{chunk[-1]['time_str']}"}},
  "events": [
    {{"time": "时间或范围", "summary": "事实/推断/假设/存疑 + 简短概括", "evidence": "短证据", "confidence": "high|medium|low"}}
  ],
  "relation_signals": [
    {{"model": "普通熟人|普通朋友|较亲近朋友|工具性/事务性关系|社群/同学/同事式关系|情绪支持或依赖|礼貌维持|低成本社交|亲密或暧昧可能|其他", "direction": "support|against|mixed", "signal": "简短信号", "evidence": "短证据", "confidence": "high|medium|low"}}
  ],
  "persona_signals": [
    {{"trait": "特征维度", "signal": "简短信号", "evidence": "短证据", "stability": "local|repeated|unclear", "confidence": "high|medium|low"}}
  ],
  "counter_evidence": [
    {{"against": "被反驳的判断", "evidence": "短反证", "confidence": "high|medium|low"}}
  ],
  "uncertainties": [
    {{"question": "不能判断的问题", "reason": "证据不足原因"}}
  ]
}}

聊天原文:
{transcript}
"""


def build_merge_prompt(name: str, batch_index: int, total_batches: int, summaries: list[str]) -> str:
    joined = "\n\n".join(summaries)
    return f"""请把以下分块分析合并成一份阶段总结，服务于最终关系分析。

对象：{name}
阶段批次：{batch_index}/{total_batches}

要求：
1. 保留每个分块里真正有价值的证据点，不要被表面热闹带偏。
2. 标出哪些模式在多个分块中重复出现，哪些只是一时情境。
3. 对每个重要判断给出支持证据和反证。
4. 不要下最终结论，只产出中间证据总结。
5. 这份阶段总结必须同时为后续“关系分析”和“人物侧写”服务。
6. 不能只合并事件，要把关系信号和人物信号都合并出来。

输出结构：
## 阶段总结 {batch_index}
### 重复出现的模式
### 只在局部出现的信号
### 对关系类型和互动角色的支持与反证
### 对人物建模的支持与反证
### 仍需在后续阶段验证的问题
最后一行必须是：<!-- END_MERGE_SUMMARY -->

以下是要合并的分块分析：

{joined}
"""


def build_merge_batches(name: str, summaries: list[str], max_prompt_bytes: int = MERGE_MAX_PROMPT_BYTES) -> list[list[str]]:
    if max_prompt_bytes <= 0:
        raise ValueError("max_prompt_bytes 必须大于 0")

    batches = []
    current = []
    for summary in summaries:
        candidate = [*current, summary]
        candidate_prompt = build_merge_prompt(name, 1, 1, candidate)
        if current and text_bytes(candidate_prompt) > max_prompt_bytes:
            batches.append(current)
            current = [summary]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def merge_batch_file(merge_dir: Path, round_index: int, batch_index: int) -> Path:
    return merge_dir / f"round_{round_index:03d}" / f"batch_{batch_index:03d}.md"


def load_cached_merge_batch(merge_dir: Path | None, round_index: int, batch_index: int) -> str | None:
    if merge_dir is None:
        return None
    batch_file = merge_batch_file(merge_dir, round_index, batch_index)
    if not batch_file.exists():
        return None
    text = batch_file.read_text(encoding="utf-8").strip()
    try:
        require_completion_marker(text, "<!-- END_MERGE_SUMMARY -->", "阶段总结合并批次")
    except IncompleteReportOutput:
        print(f"- 已有合并批次缓存不完整，将重新生成: {batch_file.name}", flush=True)
        return None
    return text


def write_cached_merge_batch(merge_dir: Path | None, round_index: int, batch_index: int, text: str) -> None:
    if merge_dir is None:
        return
    batch_file = merge_batch_file(merge_dir, round_index, batch_index)
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    batch_file.write_text(text.strip() + "\n", encoding="utf-8")


def merge_summaries_if_needed(
    name: str,
    summaries: list[str],
    skill_prompt: str,
    config,
    timeout: int,
    merge_dir: Path | None = None,
) -> list[str]:
    current = summaries
    merge_round = 1
    while text_bytes("\n\n".join(current)) > SUMMARY_MAX_BYTES and len(current) > 1:
        before_bytes = text_bytes("\n\n".join(current))
        merged = []
        batches = build_merge_batches(name, current)
        total_batches = len(batches)
        print(
            f"- 阶段总结过长，执行第 {merge_round} 轮压缩合并，共 {total_batches} 批，"
            f"合并前 {before_bytes} bytes",
            flush=True,
        )
        merge_max_tokens = MERGE_MAX_OUTPUT_TOKENS
        for batch_index, batch in enumerate(batches, start=1):
            prompt = build_merge_prompt(name, batch_index, total_batches, batch)
            prompt_bytes = text_bytes(prompt)
            cached_summary = load_cached_merge_batch(merge_dir, merge_round, batch_index)
            if cached_summary is not None:
                print(
                    f"- 复用已完成合并批次 {batch_index}/{total_batches}: "
                    f"{text_bytes(cached_summary)} bytes",
                    flush=True,
                )
                merged.append(cached_summary)
                continue
            print(
                f"- 合并批次 {batch_index}/{total_batches}: "
                f"{len(batch)} 个分块总结，prompt {prompt_bytes} bytes",
                flush=True,
            )
            result = call_llm_with_retry_result(
                config,
                skill_prompt,
                prompt,
                timeout,
                max_tokens=merge_max_tokens,
            )
            merged_summary = result.content
            require_completion_marker(merged_summary, "<!-- END_MERGE_SUMMARY -->", "阶段总结合并批次")
            if result.max_tokens_used > merge_max_tokens:
                merge_max_tokens = result.max_tokens_used
                print(f"- 后续合并批次起始输出上限调整为: {merge_max_tokens}", flush=True)
            write_cached_merge_batch(merge_dir, merge_round, batch_index, merged_summary)
            print(f"- 合并批次 {batch_index}/{total_batches} 完成: {text_bytes(merged_summary)} bytes", flush=True)
            merged.append(merged_summary)
        current = merged
        after_bytes = text_bytes("\n\n".join(current))
        print(f"- 第 {merge_round} 轮压缩完成: {before_bytes} -> {after_bytes} bytes", flush=True)
        if len(current) >= len(batches) and after_bytes >= before_bytes:
            print("- 阶段总结压缩未变短，停止继续合并以避免死循环", flush=True)
            break
        merge_round += 1
    return current


def _extract_chunk_evidence(artifacts, run_paths, chunk_sets, chunk_manifest, cached,
                            config, timeout, chunk_sleep):
    chunk_dir, evidence_dir = run_paths["chunk_dir"], run_paths["evidence_dir"]
    evidence_docs = []
    chunk_max_tokens = CHUNK_OUTPUT_TOKENS
    for chunk_index, chunk in enumerate(chunk_sets, start=1):
        existing = cached.get(chunk_index)
        if existing:
            print(f"- 复用已完成分块 {chunk_index}/{len(chunk_sets)}")
            summary, evidence = existing
        else:
            print(f"- 分析分块 {chunk_index}/{len(chunk_sets)}")
            prompt = build_chunk_prompt(artifacts["stats"], chunk_index, len(chunk_sets), chunk)
            print(f"- 分块提示词大小: {len(prompt.encode('utf-8'))} bytes")
            try:
                summary, evidence, used_max_tokens = generate_chunk_analysis(
                    config,
                    prompt,
                    timeout,
                    chunk_manifest[chunk_index - 1],
                    initial_max_tokens=chunk_max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                if should_retry_with_smaller_prompt(exc):
                    raise RetryableChunkStageFailure(
                        f"分块 {chunk_index} 请求失败，可尝试缩小分块: {exc}"
                    ) from exc
                raise
            if used_max_tokens > chunk_max_tokens:
                chunk_max_tokens = used_max_tokens
                print(f"- 后续分块起始输出上限调整为: {chunk_max_tokens}")
            chunk_file = chunk_dir / f"chunk_{chunk_index:03d}.md"
            chunk_file.write_text(json.dumps(current_redactor().tree(evidence), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            evidence = ground_document(evidence, chunk)
            write_chunk_evidence(evidence_dir, evidence)
            if chunk_sleep > 0 and chunk_index < len(chunk_sets):
                print(f"- 分块请求冷却: {chunk_sleep:g}s")
                time.sleep(chunk_sleep)
        evidence_docs.append(evidence)

    return evidence_docs


@redact_artifact_call()
def run_single_workflow(
    artifacts: dict,
    skill_prompt: str,
    config,
    timeout: int,
    chunk_timeout: int = CHUNK_REQUEST_TIMEOUT,
    chunk_sleep: float = CHUNK_REQUEST_SLEEP_SECONDS,
    chunk_size: int = CHUNK_MESSAGE_COUNT,
    chunk_max_bytes: int = CHUNK_MAX_TRANSCRIPT_BYTES,
    chunk_max_prompt_bytes: int = CHUNK_MAX_PROMPT_BYTES,
    resume_run_id: str | None = None,
) -> dict:
    identity = run_identity(artifacts, skill_prompt, config)
    if resume_run_id:
        saved = resume_metadata(artifacts, resume_run_id, identity)
        chunk_size, chunk_max_bytes, chunk_max_prompt_bytes = (saved[key] for key in LIMIT_KEYS)
    run_paths = build_run_paths(artifacts, resume_run_id)
    chunk_dir = run_paths["chunk_dir"]
    evidence_dir = run_paths["evidence_dir"]
    # 续跑预检全部在写入元数据和覆盖清单之前，拒绝修改已有的不一致 run。
    chunk_sets = chunk_messages(artifacts["messages"], chunk_size, chunk_max_bytes)
    chunk_sets = split_chunks_by_prompt_budget(artifacts["stats"], chunk_sets, chunk_max_prompt_bytes)
    validate_chunk_coverage(artifacts["messages"], chunk_sets)
    chunk_manifest = build_chunk_manifest(chunk_sets)
    cached = {}
    if resume_run_id:
        manifest_path = run_paths['chunk_manifest_file']
        if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding='utf-8')) != chunk_manifest:
            raise ValueError('缓存覆盖清单与当前分块不一致，拒绝续跑。')
        cached = {index: load_existing_chunk_result(run_paths, index, meta, chunk_sets[index - 1])
                  for index, meta in enumerate(chunk_manifest, start=1)}
        if run_paths['phase_summary_file'].exists() and not all(cached.values()):
            raise ValueError('缓存阶段总结存在但分块证据不完整，请重新运行。')
    write_run_metadata(artifacts, run_paths, "running", {
        **identity,
        "chunk_size": chunk_size,
        "chunk_max_bytes": chunk_max_bytes,
        "chunk_max_prompt_bytes": chunk_max_prompt_bytes,
        "resumed": resume_run_id is not None,
    })
    chunk_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    coverage_summary = format_coverage_summary(artifacts["stats"], chunk_manifest)
    run_paths["chunk_manifest_file"].write_text(
        json.dumps(chunk_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_paths["coverage_summary_file"].write_text(coverage_summary + "\n", encoding="utf-8")
    print(f"- 分块数: {len(chunk_sets)}")
    print(f"- 分块覆盖: {sum(item['message_count'] for item in chunk_manifest)}/{artifacts['message_count']} 条")

    evidence_docs = _extract_chunk_evidence(
        artifacts, run_paths, chunk_sets, chunk_manifest, cached, config,
        min(timeout, chunk_timeout), chunk_sleep)
    chunk_summaries = [format_evidence_summary([doc]) for doc in evidence_docs]

    evidence_summary = write_evidence_ledger(
        evidence_docs,
        run_paths["evidence_ledger_file"],
        run_paths["evidence_summary_file"],
        artifacts["messages"],
    )

    if resume_run_id and run_paths["phase_summary_file"].exists():
        print(f"- 复用已完成阶段总结: {run_paths['phase_summary_file'].name}")
        phase_summaries = [run_paths["phase_summary_file"].read_text(encoding="utf-8").strip()]
    else:
        phase_summaries = merge_summaries_if_needed(
            artifacts["name"],
            chunk_summaries,
            skill_prompt,
            config,
            timeout,
            merge_dir=run_paths["merge_dir"],
        )
        run_paths["phase_summary_file"].write_text("\n\n".join(phase_summaries).strip() + "\n", encoding="utf-8")

    final_prompt = build_final_prompt(
        artifacts["name"],
        artifacts["stats_summary"],
        coverage_summary,
        evidence_summary,
        phase_summaries,
    )
    print(f"- 综合全部分块，生成最终报告... prompt {text_bytes(final_prompt)} bytes")
    report = generate_marked_report(
        config,
        skill_prompt,
        final_prompt,
        timeout,
        marker="<!-- END_FINAL_REPORT -->",
        label="最终关系报告",
        initial_max_tokens=FINAL_REPORT_MAX_OUTPUT_TOKENS,
    )
    write_report(
        run_paths["analysis_file"],
        run_paths["analysis_meta_file"],
        report,
        source="api",
        script_name="agent/run_workflow.py",
        model=config.model,
        note=f"全量分块覆盖 {artifacts['message_count']} 条有效消息，共 {len(chunk_sets)} 个分块，run_id={run_paths['run_id']}",
        extra_metadata={
            "run_id": run_paths["run_id"],
            "reuse_identity": report_identity(artifacts, skill_prompt, config),
            "message_count": artifacts["message_count"],
            "chunk_count": len(chunk_sets),
            "chunk_manifest_file": str(run_paths["chunk_manifest_file"]),
            "coverage_summary_file": str(run_paths["coverage_summary_file"]),
            "evidence_ledger_file": str(run_paths["evidence_ledger_file"]),
            "evidence_summary_file": str(run_paths["evidence_summary_file"]),
            "phase_summary_file": str(run_paths["phase_summary_file"]),
            "full_transcript_file": str(artifacts["full_transcript_file"]),
            "diagnostics_file": str(artifacts["diagnostics_file"]),
        },
    )
    run_audit = audit_artifacts({**artifacts, **run_paths}, require_persona=False)
    if not run_audit.ok:
        raise ValueError('本次运行产物审计失败，未替换既有报告：' + '；'.join(run_audit.errors))
    for warning in run_audit.warnings:
        print(f'- 审计提示: {warning}')
    promote_successful_run(artifacts, run_paths)
    write_run_metadata(artifacts, run_paths, "success", {
        "chunk_count": len(chunk_sets),
        "analysis_file": str(artifacts["analysis_file"]),
        "evidence_ledger_file": str(artifacts["evidence_ledger_file"]),
    })
    return {
        "name": artifacts["name"],
        "status": "success",
        "run_id": run_paths["run_id"],
        "analysis_file": str(artifacts["analysis_file"]),
        "evidence_ledger_file": str(artifacts["evidence_ledger_file"]),
        "chunk_count": len(chunk_sets),
        "sample_count": artifacts["sample_count"],
        "segment_count": artifacts["segment_count"],
    }


def shrink_chunk_limits(chunk_size: int, chunk_max_bytes: int) -> tuple[int, int]:
    next_chunk_size = max(1, int(chunk_size * CHUNK_AUTO_SHRINK_FACTOR))
    next_chunk_max_bytes = max(CHUNK_MIN_TRANSCRIPT_BYTES, int(chunk_max_bytes * CHUNK_AUTO_SHRINK_FACTOR))
    if next_chunk_size == chunk_size and chunk_size > 1:
        next_chunk_size = chunk_size - 1
    if next_chunk_max_bytes == chunk_max_bytes and chunk_max_bytes > CHUNK_MIN_TRANSCRIPT_BYTES:
        next_chunk_max_bytes = max(CHUNK_MIN_TRANSCRIPT_BYTES, chunk_max_bytes - 1)
    return next_chunk_size, next_chunk_max_bytes


def shrink_prompt_limit(chunk_max_prompt_bytes: int) -> int:
    next_limit = max(CHUNK_MIN_PROMPT_BYTES, int(chunk_max_prompt_bytes * CHUNK_AUTO_SHRINK_FACTOR))
    if next_limit == chunk_max_prompt_bytes and chunk_max_prompt_bytes > CHUNK_MIN_PROMPT_BYTES:
        next_limit = max(CHUNK_MIN_PROMPT_BYTES, chunk_max_prompt_bytes - 1)
    return next_limit


def run_single_workflow_with_auto_shrink(
    artifacts: dict,
    skill_prompt: str,
    config,
    timeout: int,
    chunk_timeout: int = CHUNK_REQUEST_TIMEOUT,
    chunk_sleep: float = CHUNK_REQUEST_SLEEP_SECONDS,
    chunk_size: int = CHUNK_MESSAGE_COUNT,
    chunk_max_bytes: int = CHUNK_MAX_TRANSCRIPT_BYTES,
    chunk_max_prompt_bytes: int = CHUNK_MAX_PROMPT_BYTES,
    resume_run_id: str | None = None,
    auto_shrink: bool = True,
) -> dict:
    current_chunk_size = chunk_size
    current_chunk_max_bytes = chunk_max_bytes
    current_chunk_max_prompt_bytes = chunk_max_prompt_bytes
    restart_count = 0

    while True:
        try:
            return run_single_workflow(
                artifacts=artifacts,
                skill_prompt=skill_prompt,
                config=config,
                timeout=timeout,
                chunk_timeout=chunk_timeout,
                chunk_sleep=chunk_sleep,
                chunk_size=current_chunk_size,
                chunk_max_bytes=current_chunk_max_bytes,
                chunk_max_prompt_bytes=current_chunk_max_prompt_bytes,
                resume_run_id=resume_run_id if restart_count == 0 else None,
            )
        except Exception as exc:  # noqa: BLE001
            can_shrink = (
                auto_shrink
                and resume_run_id is None
                and isinstance(exc, RetryableChunkStageFailure)
                and restart_count < CHUNK_AUTO_SHRINK_MAX_RESTARTS
                and (
                    current_chunk_max_bytes > CHUNK_MIN_TRANSCRIPT_BYTES
                    or current_chunk_max_prompt_bytes > CHUNK_MIN_PROMPT_BYTES
                )
            )
            if not can_shrink:
                raise

            next_chunk_size, next_chunk_max_bytes = shrink_chunk_limits(
                current_chunk_size,
                current_chunk_max_bytes,
            )
            next_chunk_max_prompt_bytes = shrink_prompt_limit(current_chunk_max_prompt_bytes)
            if (
                next_chunk_size == current_chunk_size
                and next_chunk_max_bytes == current_chunk_max_bytes
                and next_chunk_max_prompt_bytes == current_chunk_max_prompt_bytes
            ):
                raise

            restart_count += 1
            print(
                "- 检测到可恢复的 LLM 请求失败，自动缩小分块后重跑本对象: "
                f"chunk_size {current_chunk_size}->{next_chunk_size}, "
                f"chunk_max_bytes {current_chunk_max_bytes}->{next_chunk_max_bytes}, "
                f"chunk_max_prompt_bytes {current_chunk_max_prompt_bytes}->{next_chunk_max_prompt_bytes}"
            )
            current_chunk_size = next_chunk_size
            current_chunk_max_bytes = next_chunk_max_bytes
            current_chunk_max_prompt_bytes = next_chunk_max_prompt_bytes
