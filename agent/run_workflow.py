"""批量运行聊天记录分析工作流。"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import error, request

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from analyze import (  # noqa: E402
    compute_statistics,
    extract_conversation_segments,
    extract_text_samples,
    format_full_transcript,
    format_statistics_report,
    load_chat,
    parse_chat_messages,
)
from config import load_llm_config  # noqa: E402
from output_paths import get_support_output_dir, sanitize_filename  # noqa: E402
from report_output import write_report  # noqa: E402
from run_analysis import generate_analysis_prompt  # noqa: E402


CHUNK_MESSAGE_COUNT = 1200
CHUNK_MAX_TRANSCRIPT_BYTES = 18000
CHUNK_MIN_TRANSCRIPT_BYTES = 12000
CHUNK_AUTO_SHRINK_FACTOR = 0.55
CHUNK_AUTO_SHRINK_MAX_RESTARTS = 3
CHUNK_MAX_PROMPT_BYTES = 32000
CHUNK_MIN_PROMPT_BYTES = 18000
CHUNK_BATCH_SIZE = 8
SUMMARY_MAX_BYTES = 120000
MERGE_MAX_PROMPT_BYTES = 30000
MERGE_MAX_OUTPUT_TOKENS = 1600
FINAL_EVIDENCE_SUMMARY_MAX_BYTES = 9000
FINAL_COVERAGE_SUMMARY_MAX_BYTES = 5000
FINAL_PHASE_SUMMARY_MAX_BYTES = 26000
FINAL_REPORT_MAX_OUTPUT_TOKENS = 3200
CHUNK_OUTPUT_TOKENS = 3600
EVIDENCE_SCHEMA_VERSION = "chat-evidence-v1"
EVIDENCE_SUMMARY_MAX_ITEMS_PER_TYPE = 60


class LLMOutputTruncated(RuntimeError):
    """Raised when the provider reports that output stopped because of token limits."""


class IncompleteChunkOutput(RuntimeError):
    """Raised when a chunk response cannot be used as complete evidence."""


class IncompleteReportOutput(RuntimeError):
    """Raised when a final report response does not include its completion marker."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行聊天记录分析工作流")
    parser.add_argument(
        "targets",
        nargs="*",
        help="可选：只处理名称或文件名包含这些关键词的聊天记录",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 analysis/分析_{对象名}.md 已存在，也重新生成",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 个匹配到的聊天记录，便于小范围试跑",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="单个 LLM 请求超时时间，单位秒，默认 600",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_MESSAGE_COUNT,
        help=f"每个分块最多包含的消息数，默认 {CHUNK_MESSAGE_COUNT}",
    )
    parser.add_argument(
        "--chunk-max-bytes",
        type=int,
        default=CHUNK_MAX_TRANSCRIPT_BYTES,
        help=f"每个分块原文的近似最大字节数，默认 {CHUNK_MAX_TRANSCRIPT_BYTES}",
    )
    parser.add_argument(
        "--chunk-max-prompt-bytes",
        type=int,
        default=CHUNK_MAX_PROMPT_BYTES,
        help=f"每个分块实际提示词的近似最大字节数，默认 {CHUNK_MAX_PROMPT_BYTES}",
    )
    parser.add_argument(
        "--no-auto-shrink",
        action="store_true",
        help="关闭 LLM 连接断开/413/超时后的自动缩小分块重跑",
    )
    parser.add_argument(
        "--force-persona",
        action="store_true",
        help="即使人物侧写已存在，也重新生成人物侧写",
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        help="从 analysis/临时文件/{对象名}/runs/{run_id} 继续未完成的完整工作流；通常配合目标名称使用",
    )
    return parser.parse_args()


def load_skill_prompt(skill_path: Path) -> str:
    return skill_path.read_text(encoding="utf-8")


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


def extract_json_block(text: str) -> dict:
    """Extract the last fenced JSON object from an LLM response."""
    matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    candidates = [match.strip() for match in reversed(matches) if match.strip().startswith("{")]
    if not candidates:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start:end + 1])

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
    }
    return normalized


def is_complete_chunk_output(summary: str, evidence: dict) -> bool:
    return evidence.get("parse_status") == "ok" and "<!-- END_CHUNK_ANALYSIS -->" in summary


def generate_chunk_analysis(
    config,
    skill_prompt: str,
    prompt: str,
    timeout: int,
    chunk_meta: dict,
) -> tuple[str, dict]:
    max_tokens = CHUNK_OUTPUT_TOKENS
    last_error = ""
    for attempt in range(1, 3):
        summary = call_llm_with_retry(
            config,
            skill_prompt,
            prompt,
            timeout,
            max_tokens=max_tokens,
        )
        evidence = normalize_chunk_evidence(summary, chunk_meta)
        if is_complete_chunk_output(summary, evidence):
            return summary, evidence
        last_error = (
            "缺少合法结构化 JSON"
            if evidence.get("parse_status") != "ok"
            else "缺少 END_CHUNK_ANALYSIS 完成标记"
        )
        print(f"- 分块输出不完整，准备重试: {last_error}")
        max_tokens = min(max_tokens * 2, 8000)
    raise IncompleteChunkOutput(f"分块 {chunk_meta['chunk_index']} 输出不完整: {last_error}")


def write_chunk_evidence(evidence_dir: Path, evidence: dict) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_file = evidence_dir / f"evidence_{evidence['chunk_index']:03d}.json"
    evidence_file.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence_file


def flatten_evidence_items(evidence_docs: list[dict], key: str, max_items: int) -> list[dict]:
    items = []
    for doc in evidence_docs:
        chunk_index = doc.get("chunk_index")
        time_range = doc.get("time_range", {})
        message_range = doc.get("message_range", {})
        for item in ensure_list(doc.get(key)):
            if not isinstance(item, dict):
                continue
            enriched = {
                "chunk_index": chunk_index,
                "time_range": time_range,
                "message_range": message_range,
                **item,
            }
            items.append(enriched)
            if len(items) >= max_items:
                return items
    return items


def compact_text(value, max_chars: int = 240) -> str:
    text = str(value or "").replace("\r\n", " ").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def format_evidence_summary(evidence_docs: list[dict]) -> str:
    sections = [
        ("events", "事件证据"),
        ("relation_signals", "关系建模证据"),
        ("persona_signals", "人物建模证据"),
        ("counter_evidence", "反证与边界"),
        ("uncertainties", "存疑问题"),
    ]
    lines = [
        "## 结构化证据摘要",
        f"- schema: {EVIDENCE_SCHEMA_VERSION}",
        f"- 分块数: {len(evidence_docs)}",
        f"- 解析失败分块数: {sum(1 for doc in evidence_docs if doc.get('parse_status') != 'ok')}",
        "- 说明: 以下摘要来自每个分块末尾的结构化 JSON 证据块，用于最终关系建模、人物侧写和后续 RAG。",
    ]
    for key, title in sections:
        items = flatten_evidence_items(evidence_docs, key, EVIDENCE_SUMMARY_MAX_ITEMS_PER_TYPE)
        lines.append(f"\n### {title}")
        if not items:
            lines.append("- 暂无结构化条目。")
            continue
        for item in items:
            chunk = item.get("chunk_index", "?")
            chunk_label = f"chunk_{chunk:03d}" if isinstance(chunk, int) else f"chunk_{chunk}"
            time_range = item.get("time_range", {})
            time_label = item.get("time") or f"{time_range.get('start', '?')} ~ {time_range.get('end', '?')}"
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
            lines.append(f"- {chunk_label} / {time_label}: {desc}{suffix}{conf}")
    return "\n".join(lines)


def write_evidence_ledger(
    evidence_docs: list[dict],
    ledger_file: Path,
    summary_file: Path,
) -> str:
    ledger_file.write_text(
        json.dumps({
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "chunk_count": len(evidence_docs),
            "documents": evidence_docs,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = format_evidence_summary(evidence_docs)
    summary_file.write_text(summary + "\n", encoding="utf-8")
    return summary


def load_evidence_summary(artifacts: dict) -> str:
    summary_file = artifacts.get("evidence_summary_file")
    if summary_file and summary_file.exists():
        return summary_file.read_text(encoding="utf-8").strip()
    return "未找到结构化证据摘要；请主要依据阶段总结和覆盖说明，并降低细粒度结论的确定性。"


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def build_run_paths(artifacts: dict, run_id: str | None = None) -> dict:
    run_id = run_id or make_run_id()
    run_dir = artifacts["runs_dir"] / run_id
    return {
        "run_id": run_id,
        "run_dir": run_dir,
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
    metadata = {
        "run_id": run_paths["run_id"],
        "status": status,
        "name": artifacts["name"],
        "safe_name": artifacts["safe_name"],
        "chat_file": str(artifacts["chat_file"]),
        "message_count": artifacts["message_count"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        metadata.update(extra)
    run_paths["run_metadata_file"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def promote_successful_run(artifacts: dict, run_paths: dict) -> None:
    copy_dir(run_paths["chunk_dir"], artifacts["chunk_dir"])
    copy_dir(run_paths["evidence_dir"], artifacts["evidence_dir"])
    copy_file(run_paths["chunk_manifest_file"], artifacts["chunk_manifest_file"])
    copy_file(run_paths["coverage_summary_file"], artifacts["coverage_summary_file"])
    copy_file(run_paths["evidence_ledger_file"], artifacts["evidence_ledger_file"])
    copy_file(run_paths["evidence_summary_file"], artifacts["evidence_summary_file"])
    copy_file(run_paths["phase_summary_file"], artifacts["phase_summary_file"])
    copy_file(run_paths["analysis_file"], artifacts["analysis_file"])
    copy_file(run_paths["analysis_meta_file"], artifacts["analysis_meta_file"])
    artifacts["latest_run_file"].write_text(
        json.dumps({
            "run_id": run_paths["run_id"],
            "run_dir": str(run_paths["run_dir"]),
            "analysis_file": str(artifacts["analysis_file"]),
            "promoted_at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_existing_chunk_result(run_paths: dict, chunk_index: int) -> tuple[str, dict] | None:
    chunk_file = run_paths["chunk_dir"] / f"chunk_{chunk_index:03d}.md"
    evidence_file = run_paths["evidence_dir"] / f"evidence_{chunk_index:03d}.json"
    if not chunk_file.exists() or not evidence_file.exists():
        return None
    summary = chunk_file.read_text(encoding="utf-8").strip()
    evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    if not is_complete_chunk_output(summary, evidence):
        print(f"- 已有分块 {chunk_index} 输出不完整，将重新生成")
        return None
    return summary, evidence


def format_stats_summary(stats: dict) -> str:
    lines = [
        "## 全局统计摘要",
        f"- 对方: {stats['other_name']}",
        f"- 时间跨度: {stats['first_date']} ~ {stats['last_date']}",
        f"- 活跃天数: {stats['active_days']}",
        f"- 总消息数: {stats['total_messages']}",
        f"- 我发送: {stats['my_message_count']} ({stats['my_message_count'] / stats['total_messages'] * 100:.1f}%)",
        f"- 对方发送: {stats['other_message_count']} ({stats['other_message_count'] / stats['total_messages'] * 100:.1f}%)",
        f"- 每日首条消息: 我={stats['daily_first_msg'].get('me', 0)}天, 对方={stats['daily_first_msg'].get('other', 0)}天",
        f"- 每日末条消息: 我={stats['daily_last_msg'].get('me', 0)}天, 对方={stats['daily_last_msg'].get('other', 0)}天",
        f"- 会话切分间隔: {stats.get('session_gap_hours', 4)}小时",
        f"- 总会话数: {stats.get('session_count', 0)}",
        f"- 会话发起者: 我={stats.get('session_initiators', {}).get('me', 0)}次, 对方={stats.get('session_initiators', {}).get('other', 0)}次",
        f"- 会话收尾者: 我={stats.get('session_closers', {}).get('me', 0)}次, 对方={stats.get('session_closers', {}).get('other', 0)}次",
        f"- 平均每会话消息数: {stats.get('avg_session_message_count')} 条, 中位 {stats.get('median_session_message_count')} 条",
        f"- 平均会话时长: {stats.get('avg_session_duration_minutes')} 分钟, 中位 {stats.get('median_session_duration_minutes')} 分钟",
        f"- 我的平均回复间隔: {stats['reply_time_my_avg_seconds']} 秒, 中位 {stats['reply_time_my_median_seconds']} 秒",
        f"- 对方平均回复间隔: {stats['reply_time_other_avg_seconds']} 秒, 中位 {stats['reply_time_other_median_seconds']} 秒",
        f"- 会话内我的平均回复: {stats.get('session_reply_time_my_avg_seconds')} 秒",
        f"- 会话内对方平均回复: {stats.get('session_reply_time_other_avg_seconds')} 秒",
        f"- 我的平均文本长度: {stats['avg_my_msg_length']} 字",
        f"- 对方平均文本长度: {stats['avg_other_msg_length']} 字",
        f"- 深夜消息数: {stats['late_night_message_count']}",
        f"- 时间格式异常消息: {stats.get('invalid_time_message_count', 0)}",
        f"- 时间戳异常消息: {stats.get('invalid_timestamp_message_count', 0)}",
        "- 月度趋势:",
    ]
    for item in stats["monthly_trend"]:
        lines.append(f"  - {item['month']}: 总{item['total']}条, 我{item['my_count']}条, 对方{item['other_count']}条")
    return "\n".join(lines)


def format_parse_diagnostics_report(diagnostics: dict) -> str:
    lines = [
        "## 解析诊断",
        f"- 原始消息数: {diagnostics.get('raw_message_count', 0)}",
        f"- 有效解析消息数: {diagnostics.get('parsed_message_count', 0)}",
        f"- 跳过消息数: {diagnostics.get('skipped_message_count', 0)}",
        f"- 是否重新排序: {diagnostics.get('sort_changed', False)}",
        f"- 时间格式异常消息: {diagnostics.get('invalid_time_message_count', 0)}",
        f"- 时间戳异常消息: {diagnostics.get('invalid_timestamp_message_count', 0)}",
        "",
        "### 跳过原因",
    ]
    skip_reasons = diagnostics.get("skip_reasons", {})
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- 无")

    lines.append("\n### 原始消息类型")
    message_types = diagnostics.get("message_types", {})
    if message_types:
        for msg_type, count in sorted(message_types.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {msg_type}: {count}")
    else:
        lines.append("- 无")
    return "\n".join(lines)


def build_chunk_prompt(stats: dict, chunk_index: int, total_chunks: int, chunk: list[dict]) -> str:
    other_name = stats["other_name"]
    transcript = "\n".join(format_message_line(message, other_name) for message in chunk)
    return f"""请阅读下面这个时间顺序连续的聊天分块，并输出一份高密度证据分析。

你正在分析第 {chunk_index}/{total_chunks} 个分块。
本分块时间范围：{chunk[0]['time_str']} ~ {chunk[-1]['time_str']}
消息数：{len(chunk)}

要求：
1. 必须把这个分块里的聊天逐行看完再下结论。
2. 只分析这个分块，不要替整个关系下最终定论。
3. 严格区分【事实】【推断】【假设】【存疑】。
4. 每个判断尽量附时间点、原话或明确上下文。
5. 优先指出反证、边界、语气变化、谁主动推进或收束。
6. 不要泛泛而谈，不要只复述统计数字。
7. 这一份分块分析必须同时覆盖三个层次：
     - 事件层：这段时间发生了什么
     - 关系层：这些互动对关系类型、互动角色和边界意味着什么
     - 人物层：对方在这段时间呈现出怎样的人物特征
8. 不允许只写事件总结，必须明确提炼“关系建模信号”和“人物建模信号”。

请按以下结构输出 Markdown：

## 分块 {chunk_index}
### 时段概览
- 用 3 到 6 条 bullet 说明这段时间主要在聊什么、互动强度如何、是否有转折。

### 关键证据
- 至少列出 8 条证据，格式类似：
  - 【事实】时间/上下文：……
  - 【推断】基于以上可推断……
  - 【假设】可能存在……但证据不足，因为……
  - 【存疑】这里无法判断……

### 互动模式观察
- 分析这个分块里：
  - 谁更主动开启、追问、延展、收尾
  - 对方对求助、吐槽、脆弱表达、玩笑、任务型话题的反应
  - 是否出现边界、回避、低成本维持、情绪支持、工具性协作信号

### 关系建模信号
- 必须明确写出这个分块对关系类型、互动角色和边界的局部支持与反证：
    - 更像普通熟人、普通朋友、较亲近朋友、工具性/事务性关系、社群/同学/同事式关系、情绪支持或依赖、礼貌维持、低成本社交、亲密或暧昧可能中的哪几类
    - 为什么支持，为什么不支持
    - 哪些只是情境性互动，哪些更像稳定模式

### 人物建模信号
- 必须明确提炼对方在这个分块中呈现出的人物特征：
    - 表达风格
    - 情绪处理方式
    - 对求助、合作、冲突、支持、边界的反应方式
    - 主动性、共情方式、控制感、回避方式、投入方式
- 每一项都尽量落到具体证据，不要写抽象形容词列表。

### 仍需后文验证的问题
- 列 2 到 4 条。

### 结构化证据 JSON
在 Markdown 分析正文之后，必须追加一个 fenced JSON 代码块，供程序抽取证据。
JSON 必须是合法 JSON，不要写注释，不要使用 Markdown 表格。
字段结构如下，数组可为空，但键必须存在：

```json
{{
  "schema_version": "{EVIDENCE_SCHEMA_VERSION}",
  "chunk_index": {chunk_index},
  "time_range": {{"start": "{chunk[0]['time_str']}", "end": "{chunk[-1]['time_str']}"}},
  "events": [
    {{"time": "时间或时间范围", "summary": "事件概括", "evidence": "原话或上下文摘录", "confidence": "high|medium|low"}}
  ],
  "relation_signals": [
    {{"model": "普通熟人|普通朋友|较亲近朋友|工具性/事务性关系|社群/同学/同事式关系|情绪支持或依赖|礼貌维持|低成本社交|亲密或暧昧可能|其他", "direction": "support|against|mixed", "signal": "关系信号", "evidence": "原话或上下文摘录", "confidence": "high|medium|low"}}
  ],
  "persona_signals": [
    {{"trait": "人物特征维度", "signal": "人物建模信号", "evidence": "原话或上下文摘录", "stability": "local|repeated|unclear", "confidence": "high|medium|low"}}
  ],
  "counter_evidence": [
    {{"against": "反驳哪个判断", "evidence": "反证原话或上下文", "confidence": "high|medium|low"}}
  ],
  "uncertainties": [
    {{"question": "仍不能判断的问题", "reason": "证据不足的原因"}}
  ]
}}
```

下面是聊天分块原文：

{transcript}
"""


def build_chunk_prompt(stats: dict, chunk_index: int, total_chunks: int, chunk: list[dict]) -> str:
    other_name = stats["other_name"]
    transcript = "\n".join(format_message_line(message, other_name) for message in chunk)
    return f"""请阅读下面这个按时间顺序连续截取的聊天分块，只分析本分块，不下全局定论。

分块: {chunk_index}/{total_chunks}
时间: {chunk[0]['time_str']} ~ {chunk[-1]['time_str']}
消息数: {len(chunk)}

输出要求：
1. 先给 Markdown 分析，严格区分【事实】【推断】【假设】【存疑】。
2. 覆盖三层：事件、关系建模信号、人物建模信号。
3. 每个重要判断尽量带时间点、原话或上下文；优先写反证、边界、语气变化、主动/收束。
4. 不预设对方性别、关系目标或恋爱/暧昧前提；亲密或暧昧只在证据支持时作为可选模型。
5. 控制篇幅，避免复述全部聊天；聚焦最能支撑或反驳模型的证据。

Markdown 结构：
## 分块 {chunk_index}
### 时段概览
### 关键证据
### 互动模式观察
### 关系建模信号
### 人物建模信号
### 仍需后文验证的问题

输出顺序必须是：
1. 先输出一个合法 fenced JSON，字段必须齐全，数组可为空。
2. 再输出简短 Markdown 分析。
3. 最后一行必须是：<!-- END_CHUNK_ANALYSIS -->

JSON schema：
```json
{{
  "schema_version": "{EVIDENCE_SCHEMA_VERSION}",
  "chunk_index": {chunk_index},
  "time_range": {{"start": "{chunk[0]['time_str']}", "end": "{chunk[-1]['time_str']}"}},
  "events": [
    {{"time": "时间或范围", "summary": "事件概括", "evidence": "原话或上下文", "confidence": "high|medium|low"}}
  ],
  "relation_signals": [
    {{"model": "普通熟人|普通朋友|较亲近朋友|工具性/事务性关系|社群/同学/同事式关系|情绪支持或依赖|礼貌维持|低成本社交|亲密或暧昧可能|其他", "direction": "support|against|mixed", "signal": "关系信号", "evidence": "原话或上下文", "confidence": "high|medium|low"}}
  ],
  "persona_signals": [
    {{"trait": "人物特征维度", "signal": "人物建模信号", "evidence": "原话或上下文", "stability": "local|repeated|unclear", "confidence": "high|medium|low"}}
  ],
  "counter_evidence": [
    {{"against": "反驳哪个判断", "evidence": "反证原话或上下文", "confidence": "high|medium|low"}}
  ],
  "uncertainties": [
    {{"question": "仍不能判断的问题", "reason": "证据不足的原因"}}
  ]
}}
```

聊天分块原文：
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


def build_final_prompt(
    name: str,
    stats_summary: str,
    coverage_summary: str,
    evidence_summary: str,
    phase_summaries: list[str],
) -> str:
    joined = limit_text_bytes("\n\n".join(phase_summaries), FINAL_PHASE_SUMMARY_MAX_BYTES)
    evidence_summary = limit_text_bytes(evidence_summary, FINAL_EVIDENCE_SUMMARY_MAX_BYTES)
    coverage_summary = limit_text_bytes(coverage_summary, FINAL_COVERAGE_SUMMARY_MAX_BYTES)
    return f"""请基于以下全局统计和全部阶段总结，生成一份细致、克制、证据导向的最终分析报告。

对象：{name}

写作标准：
1. 参考法证式分析，不要写空话套话。
2. 每个大结论都要写支持证据、反证、边界条件。
3. 要明确区分长期模式、阶段性模式、一次性事件。
4. 可以细，但不能越界读心。
5. 若证据不足，明确写【存疑】。
6. 最终输出要明显比普通总结更细，尽量像“逐项举证的分析报告”。
7. 你看到的阶段总结来自对完整聊天原文的连续分块阅读；请把“全量覆盖说明”作为证据链范围，不要误认为只看了采样。
8. “结构化证据摘要”是从每个分块的 JSON 证据块汇总而来。最终结论应优先绑定这些证据条目，并同时检查反证与存疑。
9. 不预设对方性别、性取向、关系目标或恋爱/暧昧前提；亲密或暧昧只作为证据充分时的可选模型。

请严格按以下结构输出 Markdown：

# {name} 聊天关系分析报告

**重要声明：** 先写一段 2 到 4 句声明，说明本报告如何区分事实、推断、假设、不能判断的部分，并说明不会进行读心或迎合式结论。

## 一、互动概况
- 聊天频率变化
- 主动性与话题控制
- 关系升温/降温节点与重要事件

## 二、对方的表达风格
- 语言风格与情绪表达
- 玩笑、吐槽、反问、表情、严肃模式切换

## 三、对方的互动模式
- 主动关心、追问、记忆、回应脆弱表达
- 面对任务、求助、冲突、边界变化、支持话题时的不同反应
- 是否存在回避、低成本维持、工具性协作等模式

## 四、对方在这段关系中的可能定位
- 用表格评估：普通熟人、普通朋友、较亲近朋友、工具性/事务性关系、社群/同学/同事式关系、情绪支持或依赖、礼貌维持、低成本社交、亲密或暧昧可能
- 每行必须包含：支持证据、反证、可信度

## 五、对方可能的动机逻辑
- 给出 3 到 5 种解释
- 每种都包含：支持证据、反证、可信度、还需要什么信息验证

## 六、对方在你面前呈现出的性格倾向
- 主动性、共情能力、边界感、情绪稳定性、关系投入方式、回避倾向、责任感、表达成熟度

## 七、你自己的误判风险
- 稀缺性滤镜、投射、特殊关系放大、情境性互动误判、选择性注意等
- 要直接指出哪些判断可能站不住脚

## 八、总结
### 1. 最稳妥的结论
### 2. 中等可信的判断
### 3. 不能下结论的部分
### 4. 低风险验证关系类型或互动边界的建议

格式要求：
1. 全文使用【事实】【推断】【假设】【存疑】四类标记。
2. 每个章节都尽量给出时间点、具体片段或概括性证据。
3. 写得具体，允许较长，但不要灌水。
4. 最后一行必须是：<!-- END_FINAL_REPORT -->

以下是全局统计：

{stats_summary}

以下是全量覆盖说明：

{coverage_summary}

以下是结构化证据摘要：

{evidence_summary}

以下是覆盖全部聊天的阶段总结：

{joined}
"""


def build_persona_prompt(
    name: str,
    stats_summary: str,
    coverage_summary: str,
    evidence_summary: str,
    phase_summaries: list[str],
    relation_report: str,
) -> str:
    joined = "\n\n".join(phase_summaries)
    return f"""请基于以下统计摘要、阶段总结和已经完成的关系分析报告，进一步建立对方的人物侧写。

对象：{name}

任务目标：
1. 输出一份独立的人物形象建模报告，而不是重复关系分析报告。
2. 重点回答“这个人，在与你互动时，呈现出怎样稳定的人物形象”。
3. 要有证据支撑，要区分【事实】【推断】【假设】【存疑】。
4. 要写出边界、互动风格、情绪表达、投入方式、回避方式、关系需求等层面。
5. 如果关系分析报告里的某个判断证据不够，要主动降级，不要照搬。
6. 阶段总结来自完整聊天原文的连续分块阅读；请结合“全量覆盖说明”判断哪些人物信号是长期模式，哪些只是局部信号。
7. 必须优先参考“结构化证据摘要”中的人物建模证据、反证与存疑；只有跨分块重复出现的信号才可写成稳定倾向。
8. 不预设对方性别；全文使用“对方/此人”。

请按以下结构输出 Markdown：

# 人物侧写：{name}

## 一、基本画像
## 二、性格与表达风格
## 三、互动模式与边界感
## 四、动机与需求
## 五、对方在你面前的倾向性总结
## 六、不能判断的部分
## 七、对用户最重要的认知

补充要求：
1. 基本画像和倾向性总结尽量用表格。
2. 动机与需求部分用“动机假设 / 支持证据 / 反证 / 可信度”的表格。
3. 语言要像“证据支撑的人物侧写”，不是普通读后感。
4. 最后一行必须是：<!-- END_PERSONA_REPORT -->

以下是统计摘要：

{stats_summary}

以下是全量覆盖说明：

{coverage_summary}

以下是结构化证据摘要：

{evidence_summary}

以下是阶段总结：

{joined}

以下是已经完成的关系分析报告：

{relation_report}
"""


def build_artifacts(
    chat_file: Path,
    output_dir: Path,
    write_supporting_files: bool = True,
) -> dict:
    data = load_chat(str(chat_file))
    session = data.get("session", {})
    raw_messages = data.get("messages", [])
    parse_result = parse_chat_messages(raw_messages)
    messages = parse_result["messages"]
    parse_diagnostics = parse_result["diagnostics"]
    if not messages:
        raise ValueError(f"{chat_file.name} 没有可分析的有效消息，解析诊断: {parse_diagnostics}")
    stats = compute_statistics(messages, session)
    sample_count = min(max(len(messages) // 120, 150), 260)
    segment_count = min(max(len(messages) // 1800, 15), 28)
    samples = extract_text_samples(messages, n_samples=sample_count)
    segments = extract_conversation_segments(messages, n_segments=segment_count)
    report_text = format_statistics_report(stats, samples, segments)
    full_transcript_text = format_full_transcript(messages, session)
    analysis_prompt = generate_analysis_prompt(stats, samples, segments, session)

    name = session.get("displayName", session.get("nickname", chat_file.stem))
    safe_name = sanitize_filename(name)
    support_dir = get_support_output_dir(output_dir, safe_name)
    preproc_file = support_dir / f"预处理_{safe_name}.txt"
    full_transcript_file = support_dir / f"全量解析_{safe_name}.txt"
    stats_file = support_dir / f"统计_{safe_name}.json"
    diagnostics_file = support_dir / f"解析诊断_{safe_name}.json"
    diagnostics_report_file = support_dir / f"解析诊断_{safe_name}.md"
    prompt_file = support_dir / f"分析prompt_{safe_name}.md"
    analysis_file = output_dir / f"分析_{safe_name}.md"
    persona_file = output_dir / f"人物侧写_{safe_name}.md"
    analysis_meta_file = support_dir / f"报告来源_分析_{safe_name}.json"
    persona_meta_file = support_dir / f"报告来源_人物侧写_{safe_name}.json"
    chunk_dir = support_dir / "分块分析"
    evidence_dir = support_dir / "结构化证据"
    runs_dir = support_dir / "runs"
    latest_run_file = support_dir / "latest_run.json"
    phase_summary_file = support_dir / f"阶段总结_{safe_name}.md"
    chunk_manifest_file = support_dir / f"分块覆盖清单_{safe_name}.json"
    coverage_summary_file = support_dir / f"全量覆盖说明_{safe_name}.md"
    evidence_ledger_file = support_dir / f"结构化证据总表_{safe_name}.json"
    evidence_summary_file = support_dir / f"结构化证据摘要_{safe_name}.md"

    if write_supporting_files:
        support_dir.mkdir(parents=True, exist_ok=True)
        preproc_file.write_text(report_text, encoding="utf-8")
        full_transcript_file.write_text(full_transcript_text, encoding="utf-8")
        stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        diagnostics_file.write_text(json.dumps(parse_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        diagnostics_report_file.write_text(format_parse_diagnostics_report(parse_diagnostics) + "\n", encoding="utf-8")
    else:
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(analysis_prompt, encoding="utf-8")

    return {
        "chat_file": chat_file,
        "name": name,
        "safe_name": safe_name,
        "session": session,
        "messages": messages,
        "stats": stats,
        "parse_diagnostics": parse_diagnostics,
        "stats_summary": format_stats_summary(stats),
        "message_count": len(messages),
        "sample_count": len(samples),
        "segment_count": len(segments),
        "prompt_bytes": len(analysis_prompt.encode("utf-8")),
        "prompt": analysis_prompt,
        "support_dir": support_dir,
        "preproc_file": preproc_file,
        "full_transcript_file": full_transcript_file,
        "stats_file": stats_file,
        "diagnostics_file": diagnostics_file,
        "diagnostics_report_file": diagnostics_report_file,
        "prompt_file": prompt_file,
        "analysis_file": analysis_file,
        "persona_file": persona_file,
        "analysis_meta_file": analysis_meta_file,
        "persona_meta_file": persona_meta_file,
        "chunk_dir": chunk_dir,
        "evidence_dir": evidence_dir,
        "runs_dir": runs_dir,
        "latest_run_file": latest_run_file,
        "phase_summary_file": phase_summary_file,
        "chunk_manifest_file": chunk_manifest_file,
        "coverage_summary_file": coverage_summary_file,
        "evidence_ledger_file": evidence_ledger_file,
        "evidence_summary_file": evidence_summary_file,
    }


def call_llm(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    max_tokens: int = 2600,
) -> str:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"LLM 返回缺少 choices: {data}")

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message") or {}
    content = message.get("content")
    if finish_reason == "length":
        raise LLMOutputTruncated("LLM response was truncated by max_tokens")
    if isinstance(content, str) and content.strip():
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        text = "\n".join(part for part in parts if part).strip()
        if text:
            return text

    reasoning_content = message.get("reasoning_content")
    if finish_reason == "length":
        raise LLMOutputTruncated("LLM response was truncated by max_tokens")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content

    raise ValueError(f"LLM 返回缺少可用内容: {data}")


def select_chat_files(talks_dir: Path, targets: list[str], limit: int | None) -> list[Path]:
    files = sorted(talks_dir.glob("*.json"))
    if targets:
        lowered = [target.lower() for target in targets]
        files = [
            file_path
            for file_path in files
            if any(target in file_path.stem.lower() for target in lowered)
        ]
    if limit is not None:
        files = files[:limit]
    return files


def should_retry_with_smaller_prompt(exc: Exception) -> bool:
    if isinstance(exc, http.client.RemoteDisconnected):
        return True
    if isinstance(exc, error.HTTPError):
        return exc.code in {400, 408, 413, 429, 500, 502, 503, 504}
    if isinstance(exc, error.URLError):
        reason = str(exc.reason).lower()
        return "closed" in reason or "reset" in reason or "timed out" in reason
    return False


def call_llm_with_retry(
    config,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    max_tokens: int,
    max_attempts: int = 3,
    retry_sleep_seconds: float = 3.0,
) -> str:
    last_error: Exception | None = None
    current_max_tokens = max_tokens
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(f"- 同请求重试: 第 {attempt} 次")
            return call_llm(
                api_base=config.api_base,
                api_key=config.api_key,
                model=config.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout=timeout,
                max_tokens=current_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if isinstance(exc, LLMOutputTruncated) and attempt < max_attempts:
                current_max_tokens = min(current_max_tokens * 2, 8000)
                print(f"- 输出被 max_tokens 截断，提升输出上限后重试: {current_max_tokens}")
                time.sleep(retry_sleep_seconds)
                continue
            if should_retry_with_smaller_prompt(exc) and attempt < max_attempts:
                print(f"- 请求失败，准备重试: {exc}")
                time.sleep(retry_sleep_seconds)
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM 调用失败")


def text_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def limit_text_bytes(text: str, max_bytes: int) -> str:
    if text_bytes(text) <= max_bytes:
        return text
    encoded = text.encode("utf-8")[:max_bytes]
    truncated = encoded.decode("utf-8", errors="ignore").rstrip()
    return truncated + "\n\n[内容已按提示词预算截断，请结合结构化证据摘要与阶段总结文件解读。]"


def require_completion_marker(text: str, marker: str, label: str) -> None:
    if marker not in text:
        raise IncompleteReportOutput(f"{label} 缺少完成标记 {marker}，疑似输出被截断")


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


def merge_summaries_if_needed(name: str, summaries: list[str], skill_prompt: str, config, timeout: int) -> list[str]:
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
        for batch_index, batch in enumerate(batches, start=1):
            prompt = build_merge_prompt(name, batch_index, total_batches, batch)
            prompt_bytes = text_bytes(prompt)
            print(
                f"- 合并批次 {batch_index}/{total_batches}: "
                f"{len(batch)} 个分块总结，prompt {prompt_bytes} bytes",
                flush=True,
            )
            merged_summary = call_llm_with_retry(
                config,
                skill_prompt,
                prompt,
                timeout,
                max_tokens=MERGE_MAX_OUTPUT_TOKENS,
            )
            require_completion_marker(merged_summary, "<!-- END_MERGE_SUMMARY -->", "阶段总结合并批次")
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


def run_single_workflow(
    artifacts: dict,
    skill_prompt: str,
    config,
    timeout: int,
    chunk_size: int = CHUNK_MESSAGE_COUNT,
    chunk_max_bytes: int = CHUNK_MAX_TRANSCRIPT_BYTES,
    chunk_max_prompt_bytes: int = CHUNK_MAX_PROMPT_BYTES,
    resume_run_id: str | None = None,
) -> dict:
    run_paths = build_run_paths(artifacts, resume_run_id)
    chunk_dir = run_paths["chunk_dir"]
    evidence_dir = run_paths["evidence_dir"]
    write_run_metadata(artifacts, run_paths, "running", {
        "chunk_size": chunk_size,
        "chunk_max_bytes": chunk_max_bytes,
        "chunk_max_prompt_bytes": chunk_max_prompt_bytes,
        "resumed": resume_run_id is not None,
    })
    chunk_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    chunk_sets = chunk_messages(artifacts["messages"], chunk_size, chunk_max_bytes)
    chunk_sets = split_chunks_by_prompt_budget(artifacts["stats"], chunk_sets, chunk_max_prompt_bytes)
    validate_chunk_coverage(artifacts["messages"], chunk_sets)
    chunk_manifest = build_chunk_manifest(chunk_sets)
    coverage_summary = format_coverage_summary(artifacts["stats"], chunk_manifest)
    run_paths["chunk_manifest_file"].write_text(
        json.dumps(chunk_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_paths["coverage_summary_file"].write_text(coverage_summary + "\n", encoding="utf-8")
    print(f"- 分块数: {len(chunk_sets)}")
    print(f"- 分块覆盖: {sum(item['message_count'] for item in chunk_manifest)}/{artifacts['message_count']} 条")

    chunk_summaries = []
    evidence_docs = []
    for chunk_index, chunk in enumerate(chunk_sets, start=1):
        existing = load_existing_chunk_result(run_paths, chunk_index) if resume_run_id else None
        if existing:
            print(f"- 复用已完成分块 {chunk_index}/{len(chunk_sets)}")
            summary, evidence = existing
        else:
            print(f"- 分析分块 {chunk_index}/{len(chunk_sets)}")
            prompt = build_chunk_prompt(artifacts["stats"], chunk_index, len(chunk_sets), chunk)
            print(f"- 分块提示词大小: {len(prompt.encode('utf-8'))} bytes")
            summary, evidence = generate_chunk_analysis(
                config,
                skill_prompt,
                prompt,
                timeout,
                chunk_manifest[chunk_index - 1],
            )
            chunk_file = chunk_dir / f"chunk_{chunk_index:03d}.md"
            chunk_file.write_text(summary.strip() + "\n", encoding="utf-8")
            write_chunk_evidence(evidence_dir, evidence)
        chunk_summaries.append(summary.strip())
        evidence_docs.append(evidence)

    evidence_summary = write_evidence_ledger(
        evidence_docs,
        run_paths["evidence_ledger_file"],
        run_paths["evidence_summary_file"],
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
    report = call_llm_with_retry(config, skill_prompt, final_prompt, timeout, max_tokens=FINAL_REPORT_MAX_OUTPUT_TOKENS)
    require_completion_marker(report, "<!-- END_FINAL_REPORT -->", "最终关系报告")
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
                chunk_size=current_chunk_size,
                chunk_max_bytes=current_chunk_max_bytes,
                chunk_max_prompt_bytes=current_chunk_max_prompt_bytes,
                resume_run_id=resume_run_id if restart_count == 0 else None,
            )
        except Exception as exc:  # noqa: BLE001
            can_shrink = (
                auto_shrink
                and resume_run_id is None
                and should_retry_with_smaller_prompt(exc)
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


def load_phase_summaries(artifacts: dict) -> list[str]:
    text = artifacts["phase_summary_file"].read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"阶段总结为空: {artifacts['phase_summary_file']}")
    return [text]


def generate_persona_report(
    artifacts: dict,
    persona_skill_prompt: str,
    config,
    timeout: int,
) -> str:
    if not artifacts["analysis_file"].exists():
        raise FileNotFoundError(f"缺少关系分析文件: {artifacts['analysis_file']}")
    if not artifacts["phase_summary_file"].exists():
        raise FileNotFoundError(f"缺少阶段总结文件: {artifacts['phase_summary_file']}")

    relation_report = artifacts["analysis_file"].read_text(encoding="utf-8")
    coverage_summary = (
        artifacts["coverage_summary_file"].read_text(encoding="utf-8").strip()
        if artifacts["coverage_summary_file"].exists()
        else "未找到覆盖说明文件；请优先依据阶段总结中的分块编号和统计摘要判断证据范围。"
    )
    persona_prompt = build_persona_prompt(
        artifacts["name"],
        artifacts["stats_summary"],
        coverage_summary,
        load_evidence_summary(artifacts),
        load_phase_summaries(artifacts),
        relation_report,
    )
    print("- 生成对方人物侧写...")
    persona_report = call_llm_with_retry(
        config,
        persona_skill_prompt,
        persona_prompt,
        timeout,
        max_tokens=3200,
    )
    require_completion_marker(persona_report, "<!-- END_PERSONA_REPORT -->", "人物侧写")
    write_report(
        artifacts["persona_file"],
        artifacts["persona_meta_file"],
        persona_report,
        source="api",
        script_name="agent/run_workflow.py",
        model=config.model,
        note=f"基于全量分块阶段总结与关系建模报告生成，覆盖 {artifacts['message_count']} 条有效消息",
        extra_metadata={
            "message_count": artifacts["message_count"],
            "analysis_file": str(artifacts["analysis_file"]),
            "phase_summary_file": str(artifacts["phase_summary_file"]),
            "coverage_summary_file": str(artifacts["coverage_summary_file"]),
            "evidence_ledger_file": str(artifacts["evidence_ledger_file"]),
            "evidence_summary_file": str(artifacts["evidence_summary_file"]),
            "full_transcript_file": str(artifacts["full_transcript_file"]),
            "diagnostics_file": str(artifacts["diagnostics_file"]),
        },
    )
    return str(artifacts["persona_file"])


def main() -> int:
    args = parse_args()
    workspace = Path(__file__).parent.parent
    talks_dir = workspace / "talks"
    output_dir = workspace / "analysis"
    output_dir.mkdir(exist_ok=True)

    config = load_llm_config(workspace)
    if not config.api_key:
        print("LLM_API_KEY 未设置，无法执行工作流分析。请先填写工作区根目录 .env")
        return 1

    skill_prompt = load_skill_prompt(Path(__file__).parent / "SKILL.md")
    persona_skill_prompt = load_skill_prompt(Path(__file__).parent / "PERSONA_SKILL.md")
    chat_files = select_chat_files(talks_dir, args.targets, args.limit)
    if not chat_files:
        print("没有匹配到要处理的聊天记录。")
        return 1

    print("=" * 60)
    print("聊天记录批量工作流分析")
    print("=" * 60)
    print(f"LLM_API_BASE: {config.api_base}")
    print(f"LLM_MODEL: {config.model}")
    print(f"LLM 配置来源: {config.source}")
    print(f"待处理聊天数: {len(chat_files)}")

    results = []
    for index, chat_file in enumerate(chat_files, start=1):
        print(f"\n[{index}/{len(chat_files)}] 处理 {chat_file.name}")
        try:
            initial_artifacts = build_artifacts(
                chat_file,
                output_dir,
                write_supporting_files=True,
            )
            need_analysis = (
                args.force
                or args.resume_run is not None
                or not initial_artifacts["analysis_file"].exists()
                or not initial_artifacts["phase_summary_file"].exists()
                or not initial_artifacts["coverage_summary_file"].exists()
                or not initial_artifacts["evidence_ledger_file"].exists()
                or not initial_artifacts["evidence_summary_file"].exists()
            )
            need_persona = (
                args.force
                or args.force_persona
                or not initial_artifacts["persona_file"].exists()
            )
            if not need_analysis and not need_persona:
                print("- 跳过：关系分析、阶段总结、覆盖说明、结构化证据与人物侧写均已存在")
                results.append({
                    "name": initial_artifacts["name"],
                    "status": "skipped",
                    "analysis_file": str(initial_artifacts["analysis_file"]),
                    "persona_file": str(initial_artifacts["persona_file"]),
                    "evidence_ledger_file": str(initial_artifacts["evidence_ledger_file"]),
                })
                continue

            print(f"- 有效消息数: {initial_artifacts['message_count']}")
            diagnostics = initial_artifacts["parse_diagnostics"]
            if diagnostics.get("sort_changed"):
                print("- 解析诊断: 原始消息顺序已按时间戳重新排序")
            if diagnostics.get("skipped_message_count") or diagnostics.get("invalid_time_message_count"):
                print(
                    "- 解析诊断: "
                    f"跳过{diagnostics.get('skipped_message_count', 0)}条, "
                    f"时间异常{diagnostics.get('invalid_time_message_count', 0)}条"
                )
            if need_analysis:
                print("- 生成完整预处理文件、全量分块分析材料与关系建模报告...")
                result = run_single_workflow_with_auto_shrink(
                    artifacts=initial_artifacts,
                    skill_prompt=skill_prompt,
                    config=config,
                    timeout=args.timeout,
                    chunk_size=args.chunk_size,
                    chunk_max_bytes=args.chunk_max_bytes,
                    chunk_max_prompt_bytes=args.chunk_max_prompt_bytes,
                    resume_run_id=args.resume_run,
                    auto_shrink=not args.no_auto_shrink,
                )
            else:
                print("- 复用已存在的关系分析、阶段总结与覆盖说明")
                result = {
                    "name": initial_artifacts["name"],
                    "status": "success",
                    "analysis_file": str(initial_artifacts["analysis_file"]),
                    "evidence_ledger_file": str(initial_artifacts["evidence_ledger_file"]),
                    "reused_analysis": True,
                    "sample_count": initial_artifacts["sample_count"],
                    "segment_count": initial_artifacts["segment_count"],
                }

            if need_persona:
                result["persona_file"] = generate_persona_report(
                    initial_artifacts,
                    persona_skill_prompt,
                    config,
                    args.timeout,
                )
            else:
                print(f"- 跳过人物侧写：{initial_artifacts['persona_file'].name} 已存在")
                result["persona_file"] = str(initial_artifacts["persona_file"])

            print(f"- 已生成: {result['analysis_file']}")
            print(f"- 已生成: {result['persona_file']}")
            results.append(result)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"- HTTP 错误: {exc.code} {detail}")
            results.append({
                "name": chat_file.stem,
                "status": "failed",
                "error": f"HTTP {exc.code}: {detail}",
            })
        except Exception as exc:  # noqa: BLE001
            print(f"- 失败: {exc}")
            results.append({
                "name": chat_file.stem,
                "status": "failed",
                "error": str(exc),
            })

    summary_dir = output_dir / "临时文件"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "工作流批处理结果.json"
    summary_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    success_count = sum(1 for item in results if item["status"] == "success")
    skipped_count = sum(1 for item in results if item["status"] == "skipped")
    failed_count = sum(1 for item in results if item["status"] == "failed")

    print("\n" + "=" * 60)
    print("批量分析完成")
    print("=" * 60)
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {failed_count}")
    print(f"结果汇总: {summary_file}")
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
