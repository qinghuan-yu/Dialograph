"""阶段总结压缩合并、确定性跨阶段本地聚合及已完成批次的缓存。"""
from __future__ import annotations

from collections import defaultdict
import re
from pathlib import Path
from evidence_grounding import GROUNDING_VERSION
from llm_client import call_llm_with_retry_result, require_completion_marker, IncompleteReportOutput
from prompt_budget import text_bytes, budget_phase_summaries
from llm_usage import usage_operation


SUMMARY_MAX_BYTES = 120000


MERGE_MAX_PROMPT_BYTES = 30000


MERGE_MAX_OUTPUT_TOKENS = 1600


def build_merge_prompt(name: str, batch_index: int, total_batches: int, summaries: list[str]) -> str:
    joined = "\n\n".join(summaries)
    return f"""请把以下分块分析合并成一份阶段总结，服务于最终关系分析。

要求：
1. 保留每个分块里真正有价值的证据点，不要被表面热闹带偏。
2. 标出哪些模式在多个分块中重复出现，哪些只是一时情境。
3. 对每个重要判断给出支持证据和反证。
4. 不要下最终结论，只产出中间证据总结。
5. 这份阶段总结必须同时为后续“关系分析”和“人物侧写”服务。
6. 不能只合并事件，要把关系信号和人物信号都合并出来。
7. 保留引文发送方“我/对方”；用户自述不能变成对方说过的话或对方的人物特征。

输出结构：
## 阶段总结（标题追加下方批次编号）
### 重复出现的模式
### 只在局部出现的信号
### 对关系类型和互动角色的支持与反证
### 对人物建模的支持与反证
### 仍需在后续阶段验证的问题
最后一行必须是：<!-- END_MERGE_SUMMARY -->

对象：{name}
阶段批次：{batch_index}/{total_batches}

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
            with usage_operation('merge', round=merge_round, batch=batch_index):
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


def _item_ref_info(item: dict, doc: dict) -> tuple[str, str, str, str]:
    refs = item.get("grounding", {}).get("source_refs", [])
    eid = item.get("evidence_id", "未核验条目")
    if refs:
        time_label = refs[0].get("time", "")
        date_label = time_label[:10] if len(time_label) >= 10 else ""
        speaker = {"me": "我", "other": "对方"}.get(refs[0].get("sender"), "发言者未核验")
    else:
        tr = doc.get("time_range", {})
        time_label = f"{tr.get('start', '?')}~{tr.get('end', '?')}"
        date_label = str(tr.get("start", ""))[:10]
        speaker = "发言者未核验"
    return eid, speaker, time_label, date_label


def build_local_phase_summaries(
    name: str,
    evidence_docs: list[dict],
    *,
    batch_size: int = 8,
) -> list[str]:
    """零 LLM 调用的跨时间确定性阶段聚合，保留完整 E 编号、发言者、日期跨度与共现频次。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if not evidence_docs:
        return []

    model_chunks: dict[str, set[int]] = defaultdict(set)
    model_dates: dict[str, set[str]] = defaultdict(set)
    trait_chunks: dict[str, set[int]] = defaultdict(set)
    trait_dates: dict[str, set[str]] = defaultdict(set)

    for doc in evidence_docs:
        cidx = int(doc.get("chunk_index") or 0)
        grounded = doc.get("grounding_version") == GROUNDING_VERSION
        for item in doc.get("relation_signals", []):
            if grounded and item.get("grounding", {}).get("status") != "matched":
                continue
            model = str(item.get("model") or "未标注模型").strip()
            _, _, _, d = _item_ref_info(item, doc)
            model_chunks[model].add(cidx)
            if d:
                model_dates[model].add(d)
        for item in doc.get("persona_signals", []):
            if grounded:
                refs = item.get("grounding", {}).get("source_refs", [])
                if item.get("grounding", {}).get("status") != "matched" or not (
                    refs and all(r.get("sender") == "other" for r in refs)
                ):
                    continue
            trait = str(item.get("trait") or "未标注维度").strip()
            _, _, _, d = _item_ref_info(item, doc)
            trait_chunks[trait].add(cidx)
            if d:
                trait_dates[trait].add(d)

    batches = [evidence_docs[i:i + batch_size] for i in range(0, len(evidence_docs), batch_size)]
    total_batches = len(batches)
    results = []

    for batch_idx, batch in enumerate(batches, start=1):
        start_chunk = batch[0].get("chunk_index", 1)
        end_chunk = batch[-1].get("chunk_index", len(evidence_docs))
        start_time = batch[0].get("time_range", {}).get("start", "?")
        end_time = batch[-1].get("time_range", {}).get("end", "?")
        lines = [
            f"## 阶段总结（批次 {batch_idx}/{total_batches}：chunk_{start_chunk:03d}~chunk_{end_chunk:03d}，{start_time} ~ {end_time}）",
            f"对象：{name}；以下为本地确定性聚合的候选阶段线索（保留引文发送方与跨期共现计数，均待连续原文复核）。",
        ]

        local_rows = []
        rel_rows = []
        persona_rows = []
        counter_rows = []
        uncertainty_rows = []

        for doc in batch:
            cidx = int(doc.get("chunk_index") or 0)
            grounded = doc.get("grounding_version") == GROUNDING_VERSION
            for item in doc.get("events", []):
                if grounded and item.get("grounding", {}).get("status") != "matched":
                    continue
                eid, speaker, t_label, _ = _item_ref_info(item, doc)
                local_rows.append(
                    f"- {eid} / {speaker} / {t_label} [chunk_{cidx:03d} 事件]: {item.get('summary', '')}；证据：{item.get('evidence', '')}"
                )
            for item in doc.get("relation_signals", []):
                if grounded and item.get("grounding", {}).get("status") != "matched":
                    continue
                eid, speaker, t_label, _ = _item_ref_info(item, doc)
                model = str(item.get("model") or "未标注模型").strip()
                direction = item.get("direction", "mixed")
                span_info = f"跨 {len(model_chunks[model])} 个分块/{len(model_dates[model])} 个日期"
                row = (
                    f"- {eid} / {speaker} / {t_label} [{model} / {direction}，{span_info}]: "
                    f"{item.get('signal', '')}；证据：{item.get('evidence', '')}"
                )
                rel_rows.append(row)
            for item in doc.get("persona_signals", []):
                if grounded:
                    refs = item.get("grounding", {}).get("source_refs", [])
                    if item.get("grounding", {}).get("status") != "matched" or not (
                        refs and all(r.get("sender") == "other" for r in refs)
                    ):
                        continue
                eid, speaker, t_label, _ = _item_ref_info(item, doc)
                trait = str(item.get("trait") or "未标注维度").strip()
                stability = item.get("stability", "unclear")
                span_info = f"跨 {len(trait_chunks[trait])} 个分块/{len(trait_dates[trait])} 个日期"
                row = (
                    f"- {eid} / {speaker} / {t_label} [{trait} / {stability}，{span_info}]: "
                    f"{item.get('signal', '')}；证据：{item.get('evidence', '')}"
                )
                persona_rows.append(row)
            for item in doc.get("counter_evidence", []):
                if grounded and item.get("grounding", {}).get("status") != "matched":
                    continue
                eid, speaker, t_label, _ = _item_ref_info(item, doc)
                c_row = (
                    f"- 反证: {eid} / {speaker} / {t_label} [反驳 {item.get('against', '未标注判断')}]: "
                    f"{item.get('evidence', '')}"
                )
                counter_rows.append(c_row)
            for item in doc.get("uncertainties", []):
                eid = item.get("evidence_id", "存疑条目")
                uncertainty_rows.append(
                    f"- {eid} [chunk_{cidx:03d} 存疑]: {item.get('question', '')}；原因：{item.get('reason', '')}"
                )

        sections = (
            ("### 只在局部出现的信号", local_rows),
            ("### 关系候选及跨期共现计数", rel_rows),
            ("### 人物候选及跨期共现计数", persona_rows),
            ("### 反证候选（相关性待原文复核）", counter_rows),
            ("### 仍需在后续阶段验证的问题", uncertainty_rows),
        )
        for heading, rows in sections:
            lines.append(heading)
            lines.extend(rows if rows else ["- 本批次暂无对应条目。"])
        lines.append("<!-- END_MERGE_SUMMARY -->")
        results.append("\n".join(lines))

    return results


def compare_phase_summaries(
    llm_summaries: list[str],
    local_summaries: list[str],
    evidence_docs: list[dict],
    *,
    budget_bytes: int = 12000,
) -> dict:
    """只记录两种格式的可观察字段；编号/日期出现数不能证明语义保留优劣。"""
    llm_budgeted = budget_phase_summaries(llm_summaries, budget_bytes)
    local_budgeted = budget_phase_summaries(local_summaries, budget_bytes)

    all_dates = sorted({
        str(doc.get("time_range", {}).get("start", ""))[:10]
        for doc in evidence_docs
        if len(str(doc.get("time_range", {}).get("start", ""))) >= 10
    })

    def metrics(text: str) -> dict:
        dates = sorted(set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)))
        eids = set(re.findall(r"\bE_[0-9a-f]{32}\b", text))
        quarters = [0, 0, 0, 0]
        if all_dates:
            for d in dates:
                for q in range(4):
                    q_slice = all_dates[q * len(all_dates) // 4:(q + 1) * len(all_dates) // 4 or len(all_dates)]
                    if q_slice and q_slice[0] <= d <= q_slice[-1]:
                        quarters[q] += 1
                        break
        return {
            "bytes": text_bytes(text),
            "evidence_id_count": len(eids),
            "date_count": len(dates),
            "quarter_date_counts": quarters,
            "has_speaker_labels": ("【我】" in text or " / 我 / " in text) and ("【对方】" in text or " / 对方 / " in text),
            "has_counter_section": "支持与反证" in text or "反证" in text,
            "has_uncertainty_section": "验证的问题" in text or "存疑" in text,
        }

    return {
        "comparison_scope": "format_observability_only",
        "semantic_preservation": "not_evaluated",
        "llm_merge": metrics(llm_budgeted),
        "local_aggregation": metrics(local_budgeted),
    }
