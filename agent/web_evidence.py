"""网页任务的派生检索索引：版本绑定、原输入核对和原子缓存发布。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading

import evidence_store


class TaskEvidenceIndex:
    def __init__(self):
        self.lock = threading.Lock()
        self.ready = set()

    @staticmethod
    def revision(ledger, source, terms):
        digest = hashlib.sha256()
        for part in (ledger.read_bytes(), source.read_bytes(), source.name.encode('utf-8'),
                     json.dumps(terms, ensure_ascii=False).encode('utf-8')):
            digest.update(len(part).to_bytes(8, 'big'))
            digest.update(part)
        return digest.hexdigest()

    def retrieve(self, ledger, cache_dir, source, terms, *, query=None, evidence_id=None,
                 expected_revision=None, limit=5):
        if evidence_id is None:
            if not isinstance(query, str) or not query.strip() or len(query) > 200:
                raise ValueError('请输入 1 至 200 字的查询关键词。')
            if type(limit) is not int or not 1 <= limit <= 10:
                raise ValueError('每类结果数量应为 1 至 10 的整数。')
        elif (not isinstance(evidence_id, str) or not re.fullmatch(r'E_[0-9a-f]{32}', evidence_id)
              or not isinstance(expected_revision, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_revision)):
            raise ValueError('证据编号或版本无效，请重新检索。')
        with self.lock:
            revision = self.revision(ledger, source, terms)
            if evidence_id is not None and revision != expected_revision:
                raise ValueError('当前任务的证据已更新，请重新检索后回查。')
            target = Path(cache_dir) / f'{revision}.sqlite3'
            try:
                if target not in self.ready or not target.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix='.index-', dir=target.parent) as temporary:
                        staged = Path(temporary) / 'index.sqlite3'
                        evidence_store.ingest_ledger(ledger, staged, input_file=source, redaction_terms=terms)
                        if self.revision(ledger, source, terms) != revision:
                            raise ValueError('证据文件正在更新，请稍后重新检索。')
                        os.replace(staged, target)
                    self.ready.add(target)
                if evidence_id is None:
                    result = evidence_store.search_evidence(target, query, limit=limit)
                else:
                    item = evidence_store.get_evidence(target, evidence_id)
                    if item is None:
                        raise ValueError('未找到当前任务中的这条证据。')
                    result = {'evidence': item}
                if self.revision(ledger, source, terms) != revision:
                    raise ValueError('证据文件正在更新，请稍后重新检索。')
                return {**result, 'ledger_revision': revision}
            except sqlite3.Error as exc:
                self.ready.discard(target)
                raise ValueError('本地证据索引暂不可用，请重新查询。') from exc
