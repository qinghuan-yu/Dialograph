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
    pointer = {
        'run_id': run_paths['run_id'], 'run_dir': str(run_paths['run_dir']),
        'analysis_file': str(artifacts['analysis_file']),
        'promoted_at': datetime.now().isoformat(timespec='seconds'),
    }
    publish_files({artifacts[key]: run_paths[key] for key in OUTPUT_KEYS},
                  pointer=(artifacts['latest_run_file'], pointer))


def publish_files(files: dict[Path, Path], *, pointer: tuple[Path, dict] | None = None) -> None:
    """暂存后替换一组产物，异常时恢复整组；可选指针最后发布。"""
    for source in files.values():
        if not source.exists():
            raise FileNotFoundError(f'发布来源缺失: {source}')
    parent = pointer[0].parent if pointer else next(iter(files)).parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.promotion-', dir=parent))
    changes = []
    retain_backup = False
    try:
        destinations = list(files)
        for index, source in enumerate(files.values()):
            target = staging / str(index)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        if pointer:
            (staging / str(len(destinations))).write_text(
                json.dumps(pointer[1], ensure_ascii=False, indent=2), encoding='utf-8')
            destinations.append(pointer[0])
        # 指针最后发布。旧目录先改名保留，所有替换成功后才清理备份。
        for index, destination in enumerate(destinations):
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = staging / f'backup-{index}'
            existed = destination.exists()
            if existed:
                os.replace(destination, backup)
            changes.append((destination, backup, existed))
            os.replace(staging / str(index), destination)
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
