"""共享本地预处理、统计摘要和产物路径。"""
from __future__ import annotations

import json
from pathlib import Path
from analyze import (load_chat, parse_chat_messages, compute_statistics,
                     extract_text_samples, extract_conversation_segments,
                     format_statistics_report, format_full_transcript)
from privacy_redaction import redact_conversation
from output_paths import get_support_output_dir, sanitize_filename
from preview_prompts import generate_analysis_prompt
from analysis_types import AnalysisArtifacts

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
        f"- 会话切分间隔: {stats['session_gap_hours']}小时",
        f"- 总会话数: {stats['session_count']}",
        f"- 会话发起者: 我={stats['session_initiators'].get('me', 0)}次, 对方={stats['session_initiators'].get('other', 0)}次",
        f"- 会话收尾者: 我={stats['session_closers'].get('me', 0)}次, 对方={stats['session_closers'].get('other', 0)}次",
        f"- 平均每会话消息数: {stats['avg_session_message_count']} 条, 中位 {stats['median_session_message_count']} 条",
        f"- 平均会话时长: {stats['avg_session_duration_minutes']} 分钟, 中位 {stats['median_session_duration_minutes']} 分钟",
        f"- 我的平均回复间隔: {stats['reply_time_my_avg_seconds']} 秒, 中位 {stats['reply_time_my_median_seconds']} 秒",
        f"- 对方平均回复间隔: {stats['reply_time_other_avg_seconds']} 秒, 中位 {stats['reply_time_other_median_seconds']} 秒",
        f"- 会话内我的平均回复: {stats['session_reply_time_my_avg_seconds']} 秒",
        f"- 会话内对方平均回复: {stats['session_reply_time_other_avg_seconds']} 秒",
        f"- 我的平均文本长度: {stats['avg_my_msg_length']} 字",
        f"- 对方平均文本长度: {stats['avg_other_msg_length']} 字",
        f"- 深夜消息数: {stats['late_night_message_count']}",
        f"- 时间格式异常消息: {stats['invalid_time_message_count']}",
        f"- 时间戳异常消息: {stats['invalid_timestamp_message_count']}",
        "- 月度趋势:",
    ]
    for item in stats["monthly_trend"]:
        lines.append(f"  - {item['month']}: 总{item['total']}条, 我{item['my_count']}条, 对方{item['other_count']}条")
    return "\n".join(lines)


def format_parse_diagnostics_report(diagnostics: dict) -> str:
    lines = [
        "## 解析诊断",
        f"- 原始消息数: {diagnostics['raw_message_count']}",
        f"- 有效解析消息数: {diagnostics['parsed_message_count']}",
        f"- 跳过消息数: {diagnostics['skipped_message_count']}",
        f"- 是否重新排序: {diagnostics['sort_changed']}",
        f"- 时间格式异常消息: {diagnostics['invalid_time_message_count']}",
        f"- 时间戳异常消息: {diagnostics['invalid_timestamp_message_count']}",
        "",
        "### 跳过原因",
    ]
    skip_reasons = diagnostics["skip_reasons"]
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- 无")

    lines.append("\n### 原始消息类型")
    message_types = diagnostics["message_types"]
    if message_types:
        for msg_type, count in sorted(message_types.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {msg_type}: {count}")
    else:
        lines.append("- 无")
    return "\n".join(lines)


def build_artifacts(
    chat_file: Path,
    output_dir: Path,
    write_supporting_files: bool = True,
    redaction_terms: list[str] | None = None,
    include_preview_prompt: bool = False,
) -> AnalysisArtifacts:
    data, redaction, redactor, output_name = redact_conversation(load_chat(str(chat_file)), redaction_terms)
    session = data["session"]
    raw_messages = data["messages"]
    parse_result = parse_chat_messages(raw_messages, session)
    messages = parse_result["messages"]
    parse_diagnostics = parse_result["diagnostics"]
    parse_diagnostics['redaction'] = redaction
    if not messages:
        raise ValueError(f"{chat_file.name} 没有可分析的有效消息，解析诊断: {parse_diagnostics}")
    stats = compute_statistics(messages, session)
    sample_count = min(max(len(messages) // 120, 150), 260)
    segment_count = min(max(len(messages) // 1800, 15), 28)
    samples = extract_text_samples(messages, n_samples=sample_count)
    segments = extract_conversation_segments(messages, n_segments=segment_count)
    report_text = format_statistics_report(stats, samples, segments)
    full_transcript_text = format_full_transcript(messages, session)
    analysis_prompt = generate_analysis_prompt(stats, samples, segments, session) if include_preview_prompt else None

    name = session.get("displayName", session.get("nickname", chat_file.stem))
    safe_name = sanitize_filename(output_name)
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
    if analysis_prompt is not None:
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(analysis_prompt, encoding="utf-8")

    return {
        "chat_file": chat_file,
        "redaction": redaction,
        "_redactor": redactor,
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
        **({"prompt_bytes": len(analysis_prompt.encode("utf-8")), "prompt": analysis_prompt}
           if analysis_prompt is not None else {}),
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


def select_chat_files(talks_dir: Path, targets: list[str], limit: int | None) -> list[Path]:
    from conversation_formats import conversation_files
    files = conversation_files(talks_dir)
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
