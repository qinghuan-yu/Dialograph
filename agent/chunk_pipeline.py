"""分块证据工作流编排与受限缩块重跑。"""
from __future__ import annotations

import json
import time

from artifact_audit import audit_artifacts
from llm_client import generate_marked_report, should_retry_with_smaller_prompt
from report_generation import build_final_prompt, FINAL_REPORT_MAX_OUTPUT_TOKENS
from report_output import write_report
from run_promotion import promote_successful_run
from evidence_grounding import ground_document
from privacy_redaction import current_redactor, redact_artifact_call
from prompt_budget import text_bytes
from analysis_types import AnalysisArtifacts
from chunking import ChunkSettings
import chunking
import chunk_evidence
import summary_merge
import workflow_state


class RetryableChunkStageFailure(RuntimeError):
    """Raised when chunk analysis can be retried with smaller chunks."""


def _extract_chunk_evidence(artifacts: AnalysisArtifacts, run: workflow_state.PreparedRun,
                            config, timeout, chunk_sleep):
    chunk_sets, chunk_manifest, cached = run.chunks, run.manifest, run.cached
    chunk_dir, evidence_dir = run.paths["chunk_dir"], run.paths["evidence_dir"]
    evidence_docs = []
    chunk_max_tokens = chunk_evidence.CHUNK_OUTPUT_TOKENS
    for chunk_index, chunk in enumerate(chunk_sets, start=1):
        existing = cached.get(chunk_index)
        if existing:
            print(f"- 复用已完成分块 {chunk_index}/{len(chunk_sets)}")
            summary, evidence = existing
        else:
            print(f"- 分析分块 {chunk_index}/{len(chunk_sets)}")
            prompt = chunking.build_chunk_prompt(artifacts["stats"], chunk_index, len(chunk_sets), chunk)
            print(f"- 分块提示词大小: {len(prompt.encode('utf-8'))} bytes")
            try:
                summary, evidence, used_max_tokens = chunk_evidence.generate_chunk_analysis(
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
            chunk_evidence.write_chunk_evidence(evidence_dir, evidence)
            if chunk_sleep > 0 and chunk_index < len(chunk_sets):
                print(f"- 分块请求冷却: {chunk_sleep:g}s")
                time.sleep(chunk_sleep)
        evidence_docs.append(evidence)

    return evidence_docs


@redact_artifact_call()
def run_single_workflow(
    artifacts: AnalysisArtifacts,
    skill_prompt: str,
    config,
    timeout: int,
    settings: ChunkSettings = ChunkSettings(),
    resume_run_id: str | None = None,
) -> dict:
    run = workflow_state.prepare_run(artifacts, skill_prompt, config, settings, resume_run_id)
    run_paths, chunk_sets, coverage_summary = run.paths, run.chunks, run.coverage_summary
    evidence_docs = _extract_chunk_evidence(
        artifacts, run, config, min(timeout, settings.chunk_timeout), settings.chunk_sleep)
    chunk_summaries = [chunk_evidence.format_evidence_summary([doc]) for doc in evidence_docs]

    evidence_summary = chunk_evidence.write_evidence_ledger(
        evidence_docs,
        run_paths["evidence_ledger_file"],
        run_paths["evidence_summary_file"],
        artifacts["messages"],
        redaction=artifacts['redaction'],
    )

    if resume_run_id and run_paths["phase_summary_file"].exists():
        print(f"- 复用已完成阶段总结: {run_paths['phase_summary_file'].name}")
        phase_summaries = [run_paths["phase_summary_file"].read_text(encoding="utf-8").strip()]
    else:
        phase_summaries = summary_merge.merge_summaries_if_needed(
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
            "reuse_identity": workflow_state.report_identity(artifacts, skill_prompt, config),
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
    workflow_state.write_run_metadata(artifacts, run_paths, "success", {
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


def run_single_workflow_with_auto_shrink(
    artifacts: AnalysisArtifacts,
    skill_prompt: str,
    config,
    timeout: int,
    settings: ChunkSettings = ChunkSettings(),
    resume_run_id: str | None = None,
    auto_shrink: bool = True,
) -> dict:
    current = settings
    restart_count = 0

    while True:
        try:
            return run_single_workflow(
                artifacts=artifacts,
                skill_prompt=skill_prompt,
                config=config,
                timeout=timeout,
                settings=current,
                resume_run_id=resume_run_id if restart_count == 0 else None,
            )
        except RetryableChunkStageFailure:
            smaller = (current.smaller() if auto_shrink and resume_run_id is None
                       and restart_count < chunking.CHUNK_AUTO_SHRINK_MAX_RESTARTS else None)
            if smaller is None:
                raise

            restart_count += 1
            print(
                "- 检测到可恢复的 LLM 请求失败，自动缩小分块后重跑本对象: "
                f"chunk_size {current.chunk_size}->{smaller.chunk_size}, "
                f"chunk_max_bytes {current.chunk_max_bytes}->{smaller.chunk_max_bytes}, "
                f"chunk_max_prompt_bytes {current.chunk_max_prompt_bytes}->{smaller.chunk_max_prompt_bytes}"
            )
            current = smaller
