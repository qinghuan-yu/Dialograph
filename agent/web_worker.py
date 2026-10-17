"""网页任务的独立进程：复用现有工作流，仅补充结构化进度事件。"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import run_workflow as workflow
from artifact_audit import audit_artifacts, has_marker, PERSONA_MARKER
from config import load_llm_config

OUTPUTS = {
    'analysis': ('analysis_file', '关系分析', 'markdown'),
    'persona': ('persona_file', '人物侧写', 'markdown'),
    'evidence': ('evidence_ledger_file', '结构化证据', 'json'),
    'summary': ('phase_summary_file', '阶段总结', 'markdown'),
    'coverage': ('coverage_summary_file', '覆盖说明', 'markdown'),
    'manifest': ('chunk_manifest_file', '分块清单', 'json'),
    'stats': ('stats_file', '统计数据', 'json'),
    'diagnostics': ('diagnostics_file', '解析诊断', 'json'),
    'preprocess': ('preproc_file', '预处理报告', 'text'),
}


def progress_from_line(line: str) -> dict:
    match = re.search(r'(分析分块|复用已完成分块) (\d+)/(\d+)', line)
    if match:
        current, total = int(match[2]), int(match[3])
        return {'stage': 'chunks', 'current': current, 'total': total,
                'completed': current if match[1].startswith('复用') else current - 1}
    if '合并批次' in line or '压缩阶段总结' in line:
        return {'stage': 'merge'}
    if '综合全部分块' in line:
        return {'stage': 'analysis'}
    if '生成人物侧写' in line or '生成对方人物侧写' in line:
        return {'stage': 'persona'}
    if '自动缩小分块' in line:
        return {'stage': 'chunks', 'current': 0, 'total': 0, 'completed': 0}
    return {}


class ProgressStream(io.TextIOBase):
    def __init__(self, emit, secret=''):
        self.emit, self.secret, self.pending = emit, secret, ''

    def write(self, text):
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            if line.strip():
                if self.secret:
                    line = line.replace(self.secret, '[已隐藏]')
                self.emit({'type': 'progress', 'message': line[:1500], **progress_from_line(line)})
        return len(text)

    def flush(self):
        pass


def run_job(workspace: Path, job_dir: Path, options: dict, emit) -> dict:
    output_dir = job_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    config = load_llm_config(workspace)
    if options['mode'] == 'full' and not config.api_key:
        raise ValueError('尚未配置 LLM_API_KEY。请在项目根目录 .env 中配置后重试。')
    emit({'type': 'progress', 'stage': 'prepare', 'message': '正在解析消息、排序并计算会话统计。'})
    with contextlib.redirect_stdout(ProgressStream(emit, config.api_key)):
        artifacts = workflow.build_artifacts(job_dir / 'input.json', output_dir)
        emit({'type': 'stats', 'stats': artifacts['stats'], 'diagnostics': artifacts['parse_diagnostics']})
        if options['mode'] == 'full':
            relation_ready = audit_artifacts(artifacts, require_persona=False).ok
            if not relation_ready:
                runs = sorted(artifacts['runs_dir'].glob('*')) if artifacts['runs_dir'].exists() else []
                resume = runs[-1].name if options.get('resume') and runs else None
                limits = {'chunk_size': options.get('chunk_size', workflow.CHUNK_MESSAGE_COUNT)}
                if resume:
                    # 恢复自动缩块后的实际参数，保持已缓存分块与消息范围一致。
                    metadata = json.loads((runs[-1] / 'run_metadata.json').read_text(encoding='utf-8'))
                    for key in ('chunk_size', 'chunk_max_bytes', 'chunk_max_prompt_bytes'):
                        if key in metadata:
                            limits[key] = metadata[key]
                emit({'type': 'progress', 'stage': 'chunks', 'message': '开始逐块提取证据。' if not resume else '继续上次运行，复用已完成分块。'})
                workflow.run_single_workflow_with_auto_shrink(
                    artifacts, workflow.load_skill_prompt(Path(__file__).parent / 'SKILL.md'),
                    config, timeout=options.get('timeout', 600),
                    resume_run_id=resume,
                    **limits,
                )
            if not has_marker(artifacts['persona_file'], PERSONA_MARKER):
                emit({'type': 'progress', 'stage': 'persona', 'message': '正在根据关系报告与证据生成人物侧写。'})
                workflow.generate_persona_report(
                    artifacts, workflow.load_skill_prompt(Path(__file__).parent / 'PERSONA_SKILL.md'),
                    config, options.get('timeout', 600),
                )
            emit({'type': 'progress', 'stage': 'audit', 'message': '核验完成标记、分块覆盖与证据完整性。'})
            audit = audit_artifacts(artifacts, require_persona=True)
            if not audit.ok:
                raise ValueError('结果完整性检查未通过：' + '；'.join(audit.errors))
        files = {}
        for key, (field, label, kind) in OUTPUTS.items():
            path = artifacts[field]
            if path.is_file():
                files[key] = {'path': str(path.relative_to(job_dir)), 'label': label, 'kind': kind}
        result = {'stats': artifacts['stats'], 'diagnostics': artifacts['parse_diagnostics'], 'files': files}
        (job_dir / 'result.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
        emit({'type': 'progress', 'stage': 'complete', 'message': '预处理完成。' if options['mode'] == 'preprocess' else '分析完成，结果已通过完整性检查。'})
        return result


def main():
    workspace, job_dir = Path(sys.argv[1]), Path(sys.argv[2])
    original = sys.stdout
    def emit(event):
        original.write(json.dumps(event, ensure_ascii=False) + '\n')
        original.flush()
    try:
        options = json.loads((job_dir / 'request.json').read_text(encoding='utf-8'))
        result = run_job(workspace, job_dir, options, emit)
        emit({'type': 'result', **result})
        return 0
    except Exception as exc:
        secret = load_llm_config(workspace).api_key
        message = str(exc)
        if secret:
            message = message.replace(secret, '[已隐藏]')
        emit({'type': 'error', 'message': message[:2000]})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
