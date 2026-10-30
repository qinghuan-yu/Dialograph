"""内部数据契约；保留可序列化的字典形状，不替代输入边界校验。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from privacy_redaction import Redactor


class ChunkMetadata(TypedDict):
    chunk_index: int
    start_message: int
    end_message: int
    message_count: int
    start_time: str
    end_time: str
    transcript_bytes: int


class RunPaths(TypedDict):
    run_id: str
    run_dir: Path
    merge_dir: Path
    chunk_dir: Path
    evidence_dir: Path
    chunk_manifest_file: Path
    coverage_summary_file: Path
    evidence_ledger_file: Path
    evidence_summary_file: Path
    phase_summary_file: Path
    analysis_file: Path
    analysis_meta_file: Path
    run_metadata_file: Path


class _ArtifactFields(TypedDict):
    chat_file: Path
    redaction: dict[str, Any]
    _redactor: Redactor
    name: str
    safe_name: str
    session: dict[str, Any]
    messages: list[dict[str, Any]]
    stats: dict[str, Any]
    parse_diagnostics: dict[str, Any]
    stats_summary: str
    message_count: int
    sample_count: int
    segment_count: int
    support_dir: Path
    preproc_file: Path
    full_transcript_file: Path
    stats_file: Path
    diagnostics_file: Path
    diagnostics_report_file: Path
    prompt_file: Path
    analysis_file: Path
    persona_file: Path
    analysis_meta_file: Path
    persona_meta_file: Path
    chunk_dir: Path
    evidence_dir: Path
    runs_dir: Path
    latest_run_file: Path
    phase_summary_file: Path
    chunk_manifest_file: Path
    coverage_summary_file: Path
    evidence_ledger_file: Path
    evidence_summary_file: Path


class AnalysisArtifacts(_ArtifactFields, total=False):
    # 只有显式请求采样提示词时才生成这两个字段。
    prompt: str
    prompt_bytes: int
