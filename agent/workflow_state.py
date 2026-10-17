"""续跑身份和参数校验；先验证，再允许修改原 run。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

LIMIT_KEYS = ('chunk_size', 'chunk_max_bytes', 'chunk_max_prompt_bytes')


def run_identity(artifacts, skill_prompt, config):
    source = artifacts['chat_file'].read_bytes()
    parsed = json.dumps({'messages': artifacts['messages'], 'session': artifacts.get('session', {})},
                        ensure_ascii=False, sort_keys=True).encode('utf-8')
    hasher = hashlib.sha256(b'dialograph-cache-v2\0')
    for name in ('run_workflow.py', 'analyze.py', 'conversation_input.py', 'evidence_validation.py', 'workflow_state.py', 'run_promotion.py'):
        hasher.update((Path(__file__).parent / name).read_bytes())
    hasher.update(skill_prompt.encode('utf-8'))
    hasher.update(str(getattr(config, 'model', '')).encode('utf-8'))
    hasher.update(str(getattr(config, 'api_base', '')).encode('utf-8'))
    return {'input_sha256': hashlib.sha256(source + b'\0' + parsed).hexdigest(),
            'pipeline_sha256': hasher.hexdigest()}


def report_identity(artifacts, skill_prompt, config, *, persona=False):
    identity = run_identity(artifacts, skill_prompt, config)
    if persona:
        for key in ('analysis_file', 'phase_summary_file', 'coverage_summary_file', 'evidence_summary_file'):
            identity[key] = hashlib.sha256(artifacts[key].read_bytes()).hexdigest()
    return identity


def report_matches(artifacts, skill_prompt, config, *, persona=False):
    key = 'persona_meta_file' if persona else 'analysis_meta_file'
    try:
        metadata = json.loads(artifacts[key].read_text(encoding='utf-8'))
        return isinstance(metadata, dict) and metadata.get('reuse_identity') == report_identity(
            artifacts, skill_prompt, config, persona=persona)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def resume_metadata(artifacts, run_id, identity):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('非法 run ID')
    path = artifacts['runs_dir'] / run_id / 'run_metadata.json'
    try:
        metadata = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('续跑元数据不可用，请重新运行。') from exc
    if not isinstance(metadata, dict) or any(key not in metadata for key in (*identity, *LIMIT_KEYS)):
        raise ValueError('旧缓存缺少输入/流程校验元数据，请不带 --resume-run 重新运行。')
    if metadata['input_sha256'] != identity['input_sha256']:
        raise ValueError('输入指纹已改变，拒绝复用旧 run；请重新运行。')
    if metadata['pipeline_sha256'] != identity['pipeline_sha256']:
        raise ValueError('流程或模型指纹已改变，拒绝复用旧 run；请重新运行。')
    if any(type(metadata[key]) is not int or metadata[key] <= 0 for key in LIMIT_KEYS):
        raise ValueError('续跑元数据包含非法分块参数。')
    return metadata
