"""命令行和网页共用的报告复用、续跑选择与完整分析决策。"""
from __future__ import annotations

from artifact_audit import audit_artifacts, has_marker, PERSONA_MARKER
from workflow_state import run_identity, resume_metadata, LIMIT_KEYS, report_matches
import chunk_pipeline as workflow
from report_generation import generate_persona_report


def analyze_conversation(artifacts, skill_prompt, persona_skill_prompt, config, timeout,
                         *, force=False, force_persona=False, resume_run_id=None,
                         auto_resume=False, on_warning=None, **chunk_options):
    warnings = []

    def warn(message):
        warnings.append(message)
        if on_warning:
            on_warning(message)
        else:
            print(f'- 审计提示: {message}')

    relation_ready = (audit_artifacts(artifacts, require_persona=False).ok
                      and report_matches(artifacts, skill_prompt, config))
    if resume_run_id:
        # 用户点名的 run 必须严格校验，不偷偷改成付费新运行。
        resume_metadata(artifacts, resume_run_id, run_identity(artifacts, skill_prompt, config))
    elif auto_resume and not relation_ready:
        incompatible = False
        runs = sorted(artifacts['runs_dir'].glob('*'), reverse=True) if artifacts['runs_dir'].exists() else []
        for run in runs:
            if not run.is_dir():
                continue
            try:
                saved = resume_metadata(artifacts, run.name, run_identity(artifacts, skill_prompt, config))
            except ValueError:
                incompatible = True
                continue
            resume_run_id = run.name
            chunk_options.update({key: saved[key] for key in LIMIT_KEYS})
            break
        if incompatible:
            warn('旧缓存与当前输入或流程不兼容，已跳过；' + ('继续兼容运行。' if resume_run_id else '将新建运行，旧缓存保留。'))
    need_analysis = force or resume_run_id is not None or not relation_ready
    need_persona = (need_analysis or force_persona
                    or not has_marker(artifacts['persona_file'], PERSONA_MARKER)
                    or not report_matches(artifacts, persona_skill_prompt, config, persona=True))
    if need_analysis:
        result = workflow.run_single_workflow_with_auto_shrink(
            artifacts, skill_prompt, config, timeout=timeout,
            resume_run_id=resume_run_id, **chunk_options)
    else:
        result = {'name': artifacts['name'], 'status': 'success' if need_persona else 'skipped',
                  'analysis_file': str(artifacts['analysis_file']),
                  'evidence_ledger_file': str(artifacts['evidence_ledger_file']),
                  'reused_analysis': True,
                  'sample_count': artifacts['sample_count'], 'segment_count': artifacts['segment_count']}
    if need_persona:
        generate_persona_report(artifacts, persona_skill_prompt, config, timeout)
    result['persona_file'] = str(artifacts['persona_file'])
    audit = audit_artifacts(artifacts, require_persona=True)
    if not audit.ok:
        raise ValueError('结果完整性检查未通过：' + '；'.join(audit.errors))
    for message in audit.warnings:
        warn(message)
    result['warnings'] = warnings
    return result
