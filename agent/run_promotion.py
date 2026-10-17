"""先暂存完整快照，发布失败时恢复旧产物；不保证断电事务性。"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

OUTPUT_KEYS = ('chunk_dir', 'evidence_dir', 'chunk_manifest_file',
               'coverage_summary_file', 'evidence_ledger_file', 'evidence_summary_file',
               'phase_summary_file', 'analysis_file', 'analysis_meta_file')


def promote_successful_run(artifacts: dict, run_paths: dict) -> None:
    # 所有来源必须存在，避免用残缺快照替换已有结果。
    for key in OUTPUT_KEYS:
        source = run_paths[key]
        if not (source.is_dir() if key.endswith('_dir') else source.is_file()):
            raise FileNotFoundError(f'发布来源缺失或类型错误: {source}')
    parent = artifacts['latest_run_file'].parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.promotion-', dir=parent))
    changes = []
    retain_backup = False
    try:
        for key in OUTPUT_KEYS:
            source, target = run_paths[key], staging / key
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        pointer = {
            'run_id': run_paths['run_id'], 'run_dir': str(run_paths['run_dir']),
            'analysis_file': str(artifacts['analysis_file']),
            'promoted_at': datetime.now().isoformat(timespec='seconds'),
        }
        (staging / 'latest_run_file').write_text(
            json.dumps(pointer, ensure_ascii=False, indent=2), encoding='utf-8')
        # 指针最后发布。旧目录先改名保留，所有替换成功后才清理备份。
        for key in (*OUTPUT_KEYS, 'latest_run_file'):
            destination = artifacts[key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = staging / f'backup-{key}'
            existed = destination.exists()
            if existed:
                os.replace(destination, backup)
            changes.append((destination, backup, existed))
            os.replace(staging / key, destination)
    except BaseException:
        try:
            for destination, backup, existed in reversed(changes):
                if destination.is_dir():
                    shutil.rmtree(destination)
                elif destination.exists():
                    destination.unlink()
                if existed:
                    os.replace(backup, destination)
        except BaseException as rollback_error:
            retain_backup = True
            raise RuntimeError(f'发布及回滚失败；保留恢复备份: {staging}') from rollback_error
        raise
    finally:
        if not retain_backup:
            shutil.rmtree(staging, ignore_errors=True)
