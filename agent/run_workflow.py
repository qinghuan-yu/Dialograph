"""批量运行聊天记录分析工作流。"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
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
    parse_message,
)
from config import load_llm_config  # noqa: E402
from output_paths import get_support_output_dir, sanitize_filename  # noqa: E402
from report_output import write_report  # noqa: E402
from run_analysis import generate_analysis_prompt  # noqa: E402


CHUNK_MESSAGE_COUNT = 1200
CHUNK_MAX_TRANSCRIPT_BYTES = 90000
CHUNK_BATCH_SIZE = 8
SUMMARY_MAX_BYTES = 120000
EVIDENCE_SCHEMA_VERSION = "chat-evidence-v1"
EVIDENCE_SUMMARY_MAX_ITEMS_PER_TYPE = 60


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
        "--force-persona",
        action="store_true",
        help="即使人物侧写已存在，也重新生成人物侧写",
    )
    return parser.parse_args()


def load_skill_prompt(skill_path: Path) -> str:
    return skill_path.read_text(encoding="utf-8")


def collect_messages(raw_messages: list[dict]) -> list[dict]:
    messages = []
    for msg in raw_messages:
        parsed = parse_message(msg)
        if parsed:
            messages.append(parsed)
    return messages


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
        f"- 我的平均回复间隔: {stats['reply_time_my_avg_seconds']} 秒, 中位 {stats['reply_time_my_median_seconds']} 秒",
        f"- 对方平均回复间隔: {stats['reply_time_other_avg_seconds']} 秒, 中位 {stats['reply_time_other_median_seconds']} 秒",
        f"- 我的平均文本长度: {stats['avg_my_msg_length']} 字",
        f"- 对方平均文本长度: {stats['avg_other_msg_length']} 字",
        f"- 深夜消息数: {stats['late_night_message_count']}",
        "- 月度趋势:",
    ]
    for item in stats["monthly_trend"]:
        lines.append(f"  - {item['month']}: 总{item['total']}条, 我{item['my_count']}条, 对方{item['other_count']}条")
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
     - 关系层：这些互动对关系定位意味着什么
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
- 必须明确写出这个分块对关系定位的局部支持与反证：
    - 更像普通朋友、较亲近朋友、工具性/事务性关系、暧昧试探、情绪依赖、低成本社交中的哪几类
    - 为什么支持，为什么不支持
    - 哪些只是情境性亲密，哪些更像稳定模式

### 人物建模信号
- 必须明确提炼对方在这个分块中呈现出的人物特征：
    - 表达风格
    - 情绪处理方式
    - 对亲密、求助、合作、边界的反应方式
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
    {{"model": "普通朋友|较亲近朋友|工具性/事务性关系|暧昧试探|情绪依赖|礼貌维持|低成本社交|其他", "direction": "support|against|mixed", "signal": "关系信号", "evidence": "原话或上下文摘录", "confidence": "high|medium|low"}}
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
### 对关系定位的支持与反证
### 对人物建模的支持与反证
### 仍需在后续阶段验证的问题

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
    joined = "\n\n".join(phase_summaries)
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
- 面对任务、求助、冲突、亲密话题时的不同反应
- 是否存在回避、低成本维持、工具性协作等模式

## 四、对方在这段关系中的可能定位
- 用表格评估：普通朋友、较亲近朋友、工具性/事务性关系、暧昧试探、情绪依赖、礼貌维持、低成本社交
- 每行必须包含：支持证据、反证、可信度

## 五、对方可能的动机逻辑
- 给出 3 到 5 种解释
- 每种都包含：支持证据、反证、可信度、还需要什么信息验证

## 六、对方在你面前呈现出的性格倾向
- 主动性、共情能力、边界感、情绪稳定性、关系推进意愿、回避倾向、责任感、表达成熟度

## 七、你自己的误判风险
- 稀缺性滤镜、投射、暧昧放大、情境性亲密误判、选择性注意等
- 要直接指出哪些判断可能站不住脚

## 八、总结
### 1. 最稳妥的结论
### 2. 中等可信的判断
### 3. 不能下结论的部分
### 4. 低风险验证建议

格式要求：
1. 全文使用【事实】【推断】【假设】【存疑】四类标记。
2. 每个章节都尽量给出时间点、具体片段或概括性证据。
3. 写得具体，允许较长，但不要灌水。

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

请按以下结构输出 Markdown：

# 人物侧写：{name}

## 一、基本画像
## 二、性格与表达风格
## 三、互动模式与边界感
## 四、动机与需求
## 五、她在你面前的倾向性总结
## 六、不能判断的部分
## 七、对用户最重要的认知

补充要求：
1. 基本画像和倾向性总结尽量用表格。
2. 动机与需求部分用“动机假设 / 支持证据 / 反证 / 可信度”的表格。
3. 语言要像“证据支撑的人物侧写”，不是普通读后感。

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
    messages = collect_messages(raw_messages)
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
    prompt_file = support_dir / f"分析prompt_{safe_name}.md"
    analysis_file = output_dir / f"分析_{safe_name}.md"
    persona_file = output_dir / f"人物侧写_{safe_name}.md"
    analysis_meta_file = support_dir / f"报告来源_分析_{safe_name}.json"
    persona_meta_file = support_dir / f"报告来源_人物侧写_{safe_name}.json"
    chunk_dir = support_dir / "分块分析"
    evidence_dir = support_dir / "结构化证据"
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
        "prompt_file": prompt_file,
        "analysis_file": analysis_file,
        "persona_file": persona_file,
        "analysis_meta_file": analysis_meta_file,
        "persona_meta_file": persona_meta_file,
        "chunk_dir": chunk_dir,
        "evidence_dir": evidence_dir,
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

    message = choices[0].get("message") or {}
    content = message.get("content")
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
) -> str:
    last_error: Exception | None = None
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
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if should_retry_with_smaller_prompt(exc) and attempt < max_attempts:
                print(f"- 请求失败，准备重试: {exc}")
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM 调用失败")


def merge_summaries_if_needed(name: str, summaries: list[str], skill_prompt: str, config, timeout: int) -> list[str]:
    current = summaries
    merge_round = 1
    while len("\n\n".join(current).encode("utf-8")) > SUMMARY_MAX_BYTES and len(current) > 1:
        merged = []
        total_batches = (len(current) + CHUNK_BATCH_SIZE - 1) // CHUNK_BATCH_SIZE
        print(f"- 阶段总结过长，执行第 {merge_round} 轮压缩合并，共 {total_batches} 批")
        for batch_index in range(total_batches):
            batch = current[batch_index * CHUNK_BATCH_SIZE:(batch_index + 1) * CHUNK_BATCH_SIZE]
            prompt = build_merge_prompt(name, batch_index + 1, total_batches, batch)
            merged.append(call_llm_with_retry(config, skill_prompt, prompt, timeout, max_tokens=2200))
        current = merged
        merge_round += 1
    return current


def run_single_workflow(
    artifacts: dict,
    skill_prompt: str,
    config,
    timeout: int,
    chunk_size: int = CHUNK_MESSAGE_COUNT,
    chunk_max_bytes: int = CHUNK_MAX_TRANSCRIPT_BYTES,
) -> dict:
    chunk_dir = artifacts["chunk_dir"]
    evidence_dir = artifacts["evidence_dir"]
    chunk_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for old_chunk in chunk_dir.glob("chunk_*.md"):
        old_chunk.unlink()
    for old_evidence in evidence_dir.glob("evidence_*.json"):
        old_evidence.unlink()

    chunk_sets = chunk_messages(artifacts["messages"], chunk_size, chunk_max_bytes)
    validate_chunk_coverage(artifacts["messages"], chunk_sets)
    chunk_manifest = build_chunk_manifest(chunk_sets)
    coverage_summary = format_coverage_summary(artifacts["stats"], chunk_manifest)
    artifacts["chunk_manifest_file"].write_text(
        json.dumps(chunk_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts["coverage_summary_file"].write_text(coverage_summary + "\n", encoding="utf-8")
    print(f"- 分块数: {len(chunk_sets)}")
    print(f"- 分块覆盖: {sum(item['message_count'] for item in chunk_manifest)}/{artifacts['message_count']} 条")

    chunk_summaries = []
    evidence_docs = []
    for chunk_index, chunk in enumerate(chunk_sets, start=1):
        print(f"- 分析分块 {chunk_index}/{len(chunk_sets)}")
        prompt = build_chunk_prompt(artifacts["stats"], chunk_index, len(chunk_sets), chunk)
        summary = call_llm_with_retry(config, skill_prompt, prompt, timeout, max_tokens=3000)
        chunk_file = chunk_dir / f"chunk_{chunk_index:03d}.md"
        chunk_file.write_text(summary.strip() + "\n", encoding="utf-8")
        chunk_summaries.append(summary.strip())
        evidence = normalize_chunk_evidence(summary, chunk_manifest[chunk_index - 1])
        write_chunk_evidence(evidence_dir, evidence)
        evidence_docs.append(evidence)

    evidence_summary = write_evidence_ledger(
        evidence_docs,
        artifacts["evidence_ledger_file"],
        artifacts["evidence_summary_file"],
    )

    phase_summaries = merge_summaries_if_needed(
        artifacts["name"],
        chunk_summaries,
        skill_prompt,
        config,
        timeout,
    )
    artifacts["phase_summary_file"].write_text("\n\n".join(phase_summaries).strip() + "\n", encoding="utf-8")

    final_prompt = build_final_prompt(
        artifacts["name"],
        artifacts["stats_summary"],
        coverage_summary,
        evidence_summary,
        phase_summaries,
    )
    print("- 综合全部分块，生成最终报告...")
    report = call_llm_with_retry(config, skill_prompt, final_prompt, timeout, max_tokens=4200)
    write_report(
        artifacts["analysis_file"],
        artifacts["analysis_meta_file"],
        report,
        source="api",
        script_name="agent/run_workflow.py",
        model=config.model,
        note=f"全量分块覆盖 {artifacts['message_count']} 条有效消息，共 {len(chunk_sets)} 个分块",
        extra_metadata={
            "message_count": artifacts["message_count"],
            "chunk_count": len(chunk_sets),
            "chunk_manifest_file": str(artifacts["chunk_manifest_file"]),
            "coverage_summary_file": str(artifacts["coverage_summary_file"]),
            "evidence_ledger_file": str(artifacts["evidence_ledger_file"]),
            "evidence_summary_file": str(artifacts["evidence_summary_file"]),
            "phase_summary_file": str(artifacts["phase_summary_file"]),
            "full_transcript_file": str(artifacts["full_transcript_file"]),
        },
    )
    return {
        "name": artifacts["name"],
        "status": "success",
        "analysis_file": str(artifacts["analysis_file"]),
        "evidence_ledger_file": str(artifacts["evidence_ledger_file"]),
        "chunk_count": len(chunk_sets),
        "sample_count": artifacts["sample_count"],
        "segment_count": artifacts["segment_count"],
    }


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
            if need_analysis:
                print("- 生成完整预处理文件、全量分块分析材料与关系建模报告...")
                result = run_single_workflow(
                    artifacts=initial_artifacts,
                    skill_prompt=skill_prompt,
                    config=config,
                    timeout=args.timeout,
                    chunk_size=args.chunk_size,
                    chunk_max_bytes=args.chunk_max_bytes,
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
