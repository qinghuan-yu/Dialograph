"""采样预览的材料压缩与请求回退；不参与完整报告生成。"""
from __future__ import annotations

from collections.abc import Callable
from analyze import extract_conversation_segments, extract_text_samples
from llm_client import call_llm_with_retry, should_retry_with_smaller_prompt
from analysis_types import AnalysisArtifacts


def compact_segments(segments: list[list[dict]], max_messages_per_segment: int = 18) -> str:
    lines = []
    for index, segment in enumerate(segments, start=1):
        lines.append(f"--- 片段 {index} ({segment[0]['time_str'][:10]}) ---")
        if len(segment) <= max_messages_per_segment:
            selected = segment
        else:
            head = max_messages_per_segment // 2
            tail = max_messages_per_segment - head
            selected = segment[:head] + segment[-tail:]

        for message in selected:
            prefix = "【我】" if message["sender"] == "me" else "【对方】"
            lines.append(f"{message['time_str'][11:16]} {prefix} {message['content']}")
    return "\n".join(lines)


def compact_samples(samples: list[dict], max_count: int = 80) -> str:
    selected = samples[:max_count]
    lines = []
    for message in selected:
        prefix = "【我】" if message["sender"] == "me" else "【对方】"
        lines.append(f"{message['time_str']} {prefix} {message['content']}")
    return "\n".join(lines)


def preview_materials(artifacts: AnalysisArtifacts, *, compact: bool = False) -> tuple[str, str]:
    segments = extract_conversation_segments(artifacts['messages'], n_segments=8 if compact else 12)
    sample_count = 30 if compact else 60
    return (compact_segments(segments, max_messages_per_segment=12 if compact else 16),
            compact_samples(extract_text_samples(artifacts['messages'], n_samples=sample_count), max_count=sample_count))


def generate_preview(config, skill_prompt: str, primary_prompt: str,
                     fallback_prompt: Callable[[], str]) -> str:
    try:
        return call_llm_with_retry(config, skill_prompt, primary_prompt,
                                   timeout=1800, max_tokens=4200, max_attempts=1)
    except Exception as exc:
        # 仅传输/尺寸类错误才切换采样材料，程序错误和鉴权失败直接传播。
        if not should_retry_with_smaller_prompt(exc):
            raise
        fallback = fallback_prompt()
        print(f'- 主请求失败，切换紧凑回退: {exc}')
        print(f"- 回退提示词大小: {len(fallback.encode('utf-8'))} bytes")
        return call_llm_with_retry(config, skill_prompt, fallback,
                                   timeout=1800, max_tokens=3200, max_attempts=2)
