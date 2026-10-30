"""运行路径、元数据与恢复预检；先验证，再允许修改原 run。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from datetime import datetime
import uuid
from dataclasses import dataclass, replace
from analysis_types import AnalysisArtifacts, RunPaths, ChunkMetadata
from chunking import ChunkSettings
import chunking
import chunk_evidence

LIMIT_KEYS = ('chunk_size', 'chunk_max_bytes', 'chunk_max_prompt_bytes')


def run_identity(artifacts, skill_prompt, config):
    source = artifacts['chat_file'].read_bytes()
    parsed = json.dumps({'messages': artifacts['messages'], 'session': artifacts.get('session', {}),
                         'redaction': artifacts.get('redaction', {})},
                        ensure_ascii=False, sort_keys=True).encode('utf-8')
    hasher = hashlib.sha256(b'dialograph-cache-v2\0')
    for name in ('run_workflow.py', 'chunk_pipeline.py', 'chunking.py', 'chunk_evidence.py', 'summary_merge.py', 'analysis_types.py', 'llm_client.py', 'prompt_budget.py', 'analysis_service.py', 'analysis_artifacts.py', 'report_generation.py', 'report_output.py', 'artifact_audit.py', 'analyze.py', 'conversation_input.py', 'conversation_formats.py', 'evidence_validation.py', 'evidence_grounding.py', 'privacy_redaction.py', 'workflow_state.py', 'run_promotion.py'):
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


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + uuid.uuid4().hex[:8]


def build_run_paths(artifacts: AnalysisArtifacts, run_id: str | None = None) -> RunPaths:
    run_id = run_id or make_run_id()
    run_dir = artifacts["runs_dir"] / run_id
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "merge_dir": run_dir / "merge_batches",
        "chunk_dir": run_dir / "分块分析",
        "evidence_dir": run_dir / "结构化证据",
        "chunk_manifest_file": run_dir / f"分块覆盖清单_{artifacts['safe_name']}.json",
        "coverage_summary_file": run_dir / f"全量覆盖说明_{artifacts['safe_name']}.md",
        "evidence_ledger_file": run_dir / f"结构化证据总表_{artifacts['safe_name']}.json",
        "evidence_summary_file": run_dir / f"结构化证据摘要_{artifacts['safe_name']}.md",
        "phase_summary_file": run_dir / f"阶段总结_{artifacts['safe_name']}.md",
        "analysis_file": run_dir / f"分析_{artifacts['safe_name']}.md",
        "analysis_meta_file": run_dir / f"报告来源_分析_{artifacts['safe_name']}.json",
        "run_metadata_file": run_dir / "run_metadata.json",
    }


def write_run_metadata(artifacts: AnalysisArtifacts, run_paths: RunPaths, status: str, extra: dict | None = None) -> None:
    run_paths["run_dir"].mkdir(parents=True, exist_ok=True)
    previous = {}
    if run_paths['run_metadata_file'].exists():
        previous = json.loads(run_paths['run_metadata_file'].read_text(encoding='utf-8'))
    metadata = {
        **previous,
        "run_id": run_paths["run_id"],
        "status": status,
        "name": artifacts["name"],
        "safe_name": artifacts["safe_name"],
        "chat_file": 'input' + artifacts["chat_file"].suffix.lower(),
        "message_count": artifacts["message_count"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        metadata.update(extra)
    run_paths["run_metadata_file"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class PreparedRun:
    paths: RunPaths
    chunks: list[list[dict]]
    manifest: list[ChunkMetadata]
    cached: dict[int, tuple[str, dict] | None]
    coverage_summary: str


def prepare_run(artifacts: AnalysisArtifacts, skill_prompt: str, config,
                settings: ChunkSettings, resume_run_id: str | None) -> PreparedRun:
    """先核对身份、分区和全部缓存，再允许写运行元数据与覆盖材料。"""
    identity = run_identity(artifacts, skill_prompt, config)
    if resume_run_id:
        saved = resume_metadata(artifacts, resume_run_id, identity)
        settings = replace(settings, **{key: saved[key] for key in LIMIT_KEYS})
    paths = build_run_paths(artifacts, resume_run_id)
    chunks = chunking.chunk_messages(artifacts['messages'], settings.chunk_size, settings.chunk_max_bytes)
    chunks = chunking.split_chunks_by_prompt_budget(artifacts['stats'], chunks, settings.chunk_max_prompt_bytes)
    chunking.validate_chunk_coverage(artifacts['messages'], chunks)
    manifest = chunking.build_chunk_manifest(chunks)
    cached = {}
    if resume_run_id:
        manifest_path = paths['chunk_manifest_file']
        if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('缓存覆盖清单与当前分块不一致，拒绝续跑。')
        cached = {index: chunk_evidence.load_existing_chunk_result(paths, index, meta, chunks[index - 1])
                  for index, meta in enumerate(manifest, start=1)}
        if paths['phase_summary_file'].exists() and not all(cached.values()):
            raise ValueError('缓存阶段总结存在但分块证据不完整，请重新运行。')
    write_run_metadata(artifacts, paths, 'running', {
        **identity, **{key: getattr(settings, key) for key in LIMIT_KEYS},
        'resumed': resume_run_id is not None,
    })
    paths['chunk_dir'].mkdir(parents=True, exist_ok=True)
    paths['evidence_dir'].mkdir(parents=True, exist_ok=True)
    coverage = chunking.format_coverage_summary(artifacts['stats'], manifest)
    paths['chunk_manifest_file'].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    paths['coverage_summary_file'].write_text(coverage + '\n', encoding='utf-8')
    print(f'- 分块数: {len(chunks)}')
    print(f"- 分块覆盖: {sum(item['message_count'] for item in manifest)}/{artifacts['message_count']} 条")
    return PreparedRun(paths, chunks, manifest, cached, coverage)
