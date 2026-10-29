"""已脱敏、原文绑定证据的单快照 SQLite/FTS5 检索库。"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import sys

from evidence_grounding import validate_grounded_ledger, LEDGER_VERSION, GROUNDING_VERSION
from evidence_validation import FIELDS
from privacy_redaction import POLICY_VERSION, load_redaction_terms

APPLICATION_ID = 0x44475231
STORE_VERSION = 1
ROLES = ('support', 'counter', 'mixed', 'uncertainty', 'quarantine')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _validated_ledger(path, input_file=None, redaction_terms=None):
    ledger = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if (not isinstance(ledger, dict) or ledger.get('schema_version') != LEDGER_VERSION
            or ledger.get('grounding_version') != GROUNDING_VERSION):
        raise ValueError('检索仅接受当前原文绑定 v2 账本。')
    redaction = ledger.get('redaction')
    if input_file is not None:
        # 旧账本不猜测脱敏状态；用显式输入重建，核对全部已编号消息。
        from analysis_artifacts import build_artifacts
        artifacts = build_artifacts(Path(input_file), Path(path).parent,
                                    write_supporting_files=False, redaction_terms=redaction_terms)
        validate_grounded_ledger(ledger, artifacts['messages'])
        expected = {**artifacts['redaction'], 'source_id': ledger['source_id']}
        if redaction is not None and (not isinstance(redaction, dict) or any(redaction.get(key) != expected[key]
                for key in ('enabled', 'policy_version', 'policy_sha256', 'source_id'))):
            raise ValueError('账本脱敏规则与当前输入重建结果不一致。')
        redaction = expected
    else:
        validate_grounded_ledger(ledger)
    if (not isinstance(redaction, dict) or redaction.get('enabled') is not True
            or redaction.get('policy_version') != POLICY_VERSION
            or not re.fullmatch(r'[0-9a-f]{64}', str(redaction.get('policy_sha256', '')))
            or redaction.get('source_id') != ledger.get('source_id')):
        raise ValueError('账本缺少绑定到来源的脱敏信息；旧 v2 账本请用 --input 在本机重新核对。')
    if not re.fullmatch(r'S_[0-9a-f]{64}', str(ledger.get('source_id', ''))):
        raise ValueError('账本输入快照编号无效。')
    if any(doc.get('parse_status') != 'ok' for doc in ledger['documents']):
        raise ValueError('账本包含未完整解析的分块。')
    scope = {'source_id': ledger['source_id'], 'ledger_version': LEDGER_VERSION,
             'grounding_version': GROUNDING_VERSION,
             'redaction': {key: redaction[key] for key in
                           ('enabled', 'policy_version', 'policy_sha256', 'source_id')}}
    return ledger, scope


def _role(category, item):
    if item['grounding']['status'] in ('unmatched', 'ambiguous'):
        return 'quarantine'
    if category == 'uncertainties':
        return 'uncertainty'
    if category == 'counter_evidence' or item.get('direction') == 'against':
        return 'counter'
    if item.get('direction') == 'mixed':
        return 'mixed'
    return 'support'


def _initialize(connection):
    application = connection.execute('PRAGMA application_id').fetchone()[0]
    tables = connection.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    if application == APPLICATION_ID:
        if connection.execute('PRAGMA user_version').fetchone()[0] != STORE_VERSION:
            raise ValueError('不支持此检索库版本；请另建数据库。')
        return
    if application or tables:
        raise ValueError('目标文件不是空数据库或受支持的证据检索库，拒绝覆盖。')
    connection.execute(f'PRAGMA application_id={APPLICATION_ID}')
    connection.execute(f'PRAGMA user_version={STORE_VERSION}')
    connection.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    connection.execute('CREATE TABLE messages (message_id TEXT PRIMARY KEY, data TEXT NOT NULL)')
    connection.execute('''CREATE TABLE evidence (
        evidence_id TEXT PRIMARY KEY, category TEXT NOT NULL, role TEXT NOT NULL,
        grounding_status TEXT NOT NULL, payload TEXT NOT NULL, search_text TEXT NOT NULL)''')
    connection.execute('''CREATE TABLE occurrences (
        evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id), chunk_index INTEGER NOT NULL,
        message_range TEXT NOT NULL, time_range TEXT NOT NULL, source_refs TEXT NOT NULL,
        PRIMARY KEY(evidence_id, chunk_index))''')
    try:
        connection.execute("CREATE VIRTUAL TABLE evidence_fts USING fts5(evidence_id UNINDEXED, search_text, tokenize='trigram')")
    except sqlite3.OperationalError as exc:
        raise ValueError('当前 SQLite 缺少 FTS5 trigram，请使用支持该功能的 Python/SQLite。') from exc


def _scope(connection):
    if (connection.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID
            or connection.execute('PRAGMA user_version').fetchone()[0] != STORE_VERSION):
        raise ValueError('不是受支持的证据检索库。')
    row = connection.execute("SELECT value FROM metadata WHERE key='scope'").fetchone()
    if row is None:
        raise ValueError('检索库缺少输入快照和脱敏范围信息。')
    return json.loads(row[0])


def ingest_ledger(ledger_file, db_file=None, *, input_file=None, redaction_terms=None):
    """先完整校验，再事务替换；不跨输入/脱敏范围合并。"""
    if redaction_terms and input_file is None:
        raise ValueError('补充脱敏词必须与显式输入一起重新核对。')
    ledger, scope = _validated_ledger(ledger_file, input_file, redaction_terms)
    path = Path(db_file) if db_file is not None else Path(ledger_file).parent / 'evidence_store.sqlite3'
    if path.resolve() == Path(ledger_file).resolve():
        raise ValueError('数据库路径不能覆盖输入账本。')
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('BEGIN IMMEDIATE')
        _initialize(connection)
        previous = connection.execute("SELECT value FROM metadata WHERE key='scope'").fetchone()
        if previous is not None and json.loads(previous[0]) != scope:
            raise ValueError('数据库绑定的输入快照或脱敏规则不同，请为新范围另建数据库。')
        connection.execute("INSERT OR IGNORE INTO metadata VALUES ('scope', ?)", (_json(scope),))
        for table in ('evidence_fts', 'occurrences', 'evidence', 'messages'):
            connection.execute(f'DELETE FROM {table}')
        connection.executemany('INSERT INTO messages VALUES (?, ?)',
            [(message['message_id'], _json(message)) for message in ledger['source_messages']])
        unique, occurrence_count = set(), 0
        quarantine = set()
        for document in ledger['documents']:
            for category in FIELDS:
                for item in document[category]:
                    evidence_id, role = item['evidence_id'], _role(category, item)
                    location_text = f" chunk_{document['chunk_index']:03d} {document['time_range']['start']} {document['time_range']['end']}"
                    location_text += ' ' + ' '.join(ref['message_id'] for ref in item['grounding']['source_refs'])
                    if evidence_id not in unique:
                        searchable = ' '.join(str(item[key]) for key in FIELDS[category])
                        searchable += f' {evidence_id} {category} {role}' + location_text
                        connection.execute('INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?)',
                            (evidence_id, category, role, item['grounding']['status'], _json(item), searchable))
                        unique.add(evidence_id)
                        if role == 'quarantine':
                            quarantine.add(evidence_id)
                    else:
                        connection.execute('UPDATE evidence SET search_text=search_text || ? WHERE evidence_id=?',
                                           (location_text, evidence_id))
                    connection.execute('INSERT INTO occurrences VALUES (?, ?, ?, ?, ?)',
                        (evidence_id, document['chunk_index'], _json(document['message_range']),
                         _json(document['time_range']), _json(item['grounding']['source_refs'])))
                    occurrence_count += 1
        connection.execute('INSERT INTO evidence_fts SELECT evidence_id, search_text FROM evidence')
    return {**scope, 'db_file': str(path), 'evidence_count': len(unique),
            'occurrence_count': occurrence_count, 'quarantine_count': len(quarantine)}


def _open_readonly(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('证据数据库不存在，请先导入账本。')
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('BEGIN')
    return connection


def _check_current(connection, ledger_file):
    scope = _scope(connection)
    if ledger_file is not None and _validated_ledger(ledger_file)[1] != scope:
        raise ValueError('当前账本快照或脱敏范围与数据库不一致，请使用对应范围的数据库。')
    return scope


def _render_record(connection, row):
    payload = json.loads(row['payload'])
    occurrences = []
    for saved in connection.execute('SELECT * FROM occurrences WHERE evidence_id=? ORDER BY chunk_index', (row['evidence_id'],)):
        refs = json.loads(saved['source_refs'])
        for ref in refs:
            source = connection.execute('SELECT data FROM messages WHERE message_id=?', (ref['message_id'],)).fetchone()
            if source is None:
                raise ValueError('检索库来源消息缺失。')
            ref['content'] = json.loads(source[0])['content']
        occurrences.append({'chunk_index': saved['chunk_index'], 'message_range': json.loads(saved['message_range']),
                            'time_range': json.loads(saved['time_range']), 'source_refs': refs})
    return {'evidence_id': row['evidence_id'], 'category': row['category'], 'role': row['role'],
            'grounding_status': row['grounding_status'], 'eligible_as_support': row['role'] == 'support',
            'quote': payload.get('evidence', ''), 'confidence': payload.get('confidence'),
            'payload': payload, 'occurrences': occurrences}


def search_evidence(db_file, query, *, limit=5, ledger_file=None, include_quarantine=False):
    """关键词相关性不代表语义支持；各角色分别限额，反证不被支持条目挤掉。"""
    if not isinstance(query, str) or not query.strip() or len(query) > 200:
        raise ValueError('查询应为 1 至 200 字的非空文本。')
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError('每类返回数量应为 1 至 50 的整数。')
    terms = query.split()
    short = any(len(term) < 3 for term in terms)
    with closing(_open_readonly(db_file)) as connection:
        scope = _check_current(connection, ledger_file)
        matches = {}
        for role in ROLES if include_quarantine else ROLES[:-1]:
            if short:
                predicate = ' AND '.join('instr(lower(search_text), lower(?)) > 0' for _ in terms)
                rows = connection.execute(f'''SELECT *, NULL AS rank FROM evidence WHERE role=? AND {predicate}
                    ORDER BY evidence_id LIMIT ?''', (role, *terms, limit)).fetchall()
            else:
                # 查询词全部作为字面短语，用户输入不能注入 MATCH 的 OR/NOT 等运算符。
                expression = ' AND '.join('"' + term.replace('"', '""') + '"' for term in terms)
                rows = connection.execute('''SELECT e.*, bm25(evidence_fts) AS rank
                    FROM evidence_fts JOIN evidence e ON e.evidence_id=evidence_fts.evidence_id
                    WHERE evidence_fts MATCH ? AND e.role=? ORDER BY rank, e.evidence_id LIMIT ?''',
                    (expression, role, limit)).fetchall()
            matches[role] = [{**_render_record(connection, row), 'rank': row['rank']} for row in rows]
    return {**scope, 'query': query, 'match_mode': 'literal_short_query' if short else 'fts5_trigram', 'matches': matches}


def get_evidence(db_file, evidence_id, *, ledger_file=None):
    with closing(_open_readonly(db_file)) as connection:
        _check_current(connection, ledger_file)
        row = connection.execute('SELECT * FROM evidence WHERE evidence_id=?', (evidence_id,)).fetchone()
        return _render_record(connection, row) if row else None


def main(argv=None):
    parser = argparse.ArgumentParser(description='本地脱敏证据入库、中文检索与来源回查；不调用模型')
    commands = parser.add_subparsers(dest='command', required=True)
    ingest = commands.add_parser('ingest', help='校验并导入当前 v2 账本')
    ingest.add_argument('ledger', type=Path)
    ingest.add_argument('--db', type=Path, help='默认在账本旁创建 evidence_store.sqlite3')
    ingest.add_argument('--input', type=Path, help='旧 v2 账本须提供原输入，在本机重新脱敏并核对')
    ingest.add_argument('--redact-terms-file', type=Path, help='配合 --input 使用原运行的补充脱敏词')
    search = commands.add_parser('search', help='按字面关键词查询，每类单独限额')
    search.add_argument('query')
    search.add_argument('--limit', type=int, default=5)
    search.add_argument('--include-quarantine', action='store_true', help='显式检查隔离条目，不升级为支持')
    show = commands.add_parser('show', help='按证据编号回查来源')
    show.add_argument('evidence_id')
    for command in (search, show):
        command.add_argument('--db', type=Path, required=True)
        command.add_argument('--ledger', type=Path, help='同时核对数据库是否对应当前账本')
    args = parser.parse_args(argv)
    try:
        if args.command == 'ingest':
            if args.redact_terms_file and not args.input:
                raise ValueError('--redact-terms-file 需要与 --input 一起使用。')
            result = ingest_ledger(args.ledger, args.db, input_file=args.input,
                                   redaction_terms=load_redaction_terms(args.redact_terms_file))
        elif args.command == 'search':
            result = search_evidence(args.db, args.query, limit=args.limit,
                                     ledger_file=args.ledger, include_quarantine=args.include_quarantine)
        else:
            result = get_evidence(args.db, args.evidence_id, ledger_file=args.ledger)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
