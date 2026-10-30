"""阶段总结压缩合并及已完成批次的缓存。"""
from __future__ import annotations

from pathlib import Path
from llm_client import call_llm_with_retry_result, require_completion_marker, IncompleteReportOutput
from prompt_budget import text_bytes


SUMMARY_MAX_BYTES = 120000


MERGE_MAX_PROMPT_BYTES = 30000


MERGE_MAX_OUTPUT_TOKENS = 1600


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
