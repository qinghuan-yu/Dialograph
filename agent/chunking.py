"""连续分块、覆盖说明与分块提示词预算。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from evidence_validation import EXTRACTION_SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION
from analysis_types import ChunkMetadata


CHUNK_MESSAGE_COUNT = 1200


CHUNK_MAX_TRANSCRIPT_BYTES = 18000


CHUNK_MIN_TRANSCRIPT_BYTES = 12000


CHUNK_AUTO_SHRINK_FACTOR = 0.55


CHUNK_AUTO_SHRINK_MAX_RESTARTS = 3


CHUNK_MAX_PROMPT_BYTES = 32000


CHUNK_MIN_PROMPT_BYTES = 18000


CHUNK_REQUEST_TIMEOUT = 180


CHUNK_REQUEST_SLEEP_SECONDS = 1.0


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


def build_chunk_manifest(chunk_sets: list[list[dict]]) -> list[ChunkMetadata]:
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


@dataclass(frozen=True)
class ChunkSettings:
    """一次运行使用的分块与请求参数；缩块返回新配置，不改写原配置。"""

    chunk_size: int = CHUNK_MESSAGE_COUNT
    chunk_max_bytes: int = CHUNK_MAX_TRANSCRIPT_BYTES
    chunk_max_prompt_bytes: int = CHUNK_MAX_PROMPT_BYTES
    chunk_timeout: int = CHUNK_REQUEST_TIMEOUT
    chunk_sleep: float = CHUNK_REQUEST_SLEEP_SECONDS

    def smaller(self) -> ChunkSettings | None:
        if (self.chunk_max_bytes <= CHUNK_MIN_TRANSCRIPT_BYTES
                and self.chunk_max_prompt_bytes <= CHUNK_MIN_PROMPT_BYTES):
            return None
        size, transcript_bytes = shrink_chunk_limits(self.chunk_size, self.chunk_max_bytes)
        return replace(self, chunk_size=size, chunk_max_bytes=transcript_bytes,
                       chunk_max_prompt_bytes=shrink_prompt_limit(self.chunk_max_prompt_bytes))
