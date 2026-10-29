"""只用合成账本验证检索、来源、隔离与数据库事务。"""
import copy
from contextlib import closing
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'agent'))
import evidence_store as store
from analysis_artifacts import build_artifacts
from evidence_grounding import ground_document, grounding_issues, LEDGER_VERSION, GROUNDING_VERSION

KEYS = ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')


def make_ledger(root, *, terms=None, changed=False):
    raw = {'session': {'id': 'synthetic-store', 'displayName': '合成对象'}, 'messages': [
        {'type': '文本消息', 'isSend': i % 2, 'formattedTime': f'2026-10-29 10:0{i}:00', 'content': content}
        for i, content in enumerate(['修复意愿：我愿意把误会说清楚。',
            '压力撤退：今天压力很大，晚些再谈。', '我愿意修复，也会主动联系。', '重复短句', '重复短句'])]}
    if changed:
        raw['messages'][0]['content'] += '这是另一个输入版本。'
    source = root / 'input.json'
    source.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
    artifacts = build_artifacts(source, root / 'out', write_supporting_files=False, redaction_terms=terms)
    messages = artifacts['messages']
    document = {key: [] for key in KEYS}
    document.update(parse_status='ok', chunk_index=1, message_range={'start': 1, 'end': 3},
                    time_range={'start': messages[0]['time_str'], 'end': messages[2]['time_str']})
    event = {'summary': '修复意愿的直接表达', 'evidence': '我愿意把误会说清楚。', 'confidence': 'medium'}
    document['events'] = [event, dict(event, evidence='修复意愿：这是一段不存在的原文')]
    relation = {'model': '修复意愿', 'direction': 'support', 'signal': '愿意澄清误会',
                'evidence': event['evidence'], 'confidence': 'medium'}
    document['relation_signals'] = [relation,
        dict(relation, model='压力撤退', direction='against', signal='存在主动联系的反例', evidence='也会主动联系'),
        dict(relation, direction='mixed', signal='修复意愿仍有混合信号', evidence='我愿意修复')]
    document['persona_signals'] = [{'trait': '修复意愿', 'signal': '在当前片段愿意沟通',
        'evidence': event['evidence'], 'stability': 'local', 'confidence': 'low'}]
    document['counter_evidence'] = [{'against': '修复意愿具有持续性',
        'evidence': '今天压力很大，晚些再谈。', 'confidence': 'low'}]
    uncertainty = {'question': '修复意愿是否能持续', 'reason': '需要更多后续互动'}
    document['uncertainties'] = [uncertainty]
    second = {**{key: [] for key in KEYS}, 'parse_status': 'ok', 'chunk_index': 2,
              'message_range': {'start': 4, 'end': 5},
              'time_range': {'start': messages[3]['time_str'], 'end': messages[4]['time_str']},
              'events': [dict(event, evidence='重复短句')], 'uncertainties': [uncertainty]}
    documents = [ground_document(document, messages[:3]), ground_document(second, messages[3:])]
    ledger = {'schema_version': LEDGER_VERSION, 'grounding_version': GROUNDING_VERSION,
              'chunk_count': 2, 'documents': documents, 'source_id': messages[0]['source_id'],
              'source_messages': messages, 'grounding_issues': grounding_issues(documents),
              'redaction': {**artifacts['redaction'], 'source_id': messages[0]['source_id']}}
    path = root / 'ledger.json'
    path.write_text(json.dumps(ledger, ensure_ascii=False), encoding='utf-8')
    return path, source, ledger


class EvidenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger, self.source, self.data = make_ledger(self.root)
        self.db = self.root / 'analysis' / 'evidence.sqlite3'

    def ingest(self):
        return store.ingest_ledger(self.ledger, self.db)

    def test_reimport_deduplicates_ids_but_retains_chunk_occurrences(self):
        first = self.ingest()
        second = self.ingest()
        self.assertEqual(first['evidence_count'], 9)
        self.assertEqual(second['evidence_count'], 9)
        self.assertEqual(second['occurrence_count'], 10)
        self.assertEqual(second['quarantine_count'], 2)

    def test_chinese_query_returns_support_counter_mixed_and_uncertainty(self):
        self.ingest()
        found = store.search_evidence(self.db, '修复意愿')
        self.assertEqual(found['match_mode'], 'fts5_trigram')
        for group in ('support', 'counter', 'mixed', 'uncertainty'):
            self.assertTrue(found['matches'][group], group)
        self.assertEqual(found['source_id'], self.data['source_id'])
        self.assertNotIn('quarantine', found['matches'])
        self.assertTrue(all(row['eligible_as_support'] for row in found['matches']['support']))
        self.assertTrue(all(not row['eligible_as_support'] for row in found['matches']['uncertainty']))

    def test_lookup_returns_exact_redacted_message_and_offsets(self):
        self.ingest()
        item = self.data['documents'][0]['events'][0]
        found = store.get_evidence(self.db, item['evidence_id'])
        ref = found['occurrences'][0]['source_refs'][0]
        self.assertEqual(ref['content'][ref['start']:ref['end']], item['evidence'])
        self.assertEqual(ref['original_index'], 0)
        self.assertEqual(found['occurrences'][0]['chunk_index'], 1)

    def test_uncertainty_keeps_all_occurrences_without_inventing_source_quotes(self):
        self.ingest()
        item = self.data['documents'][0]['uncertainties'][0]
        found = store.get_evidence(self.db, item['evidence_id'])
        self.assertEqual([row['chunk_index'] for row in found['occurrences']], [1, 2])
        self.assertFalse(found['eligible_as_support'])
        self.assertEqual(found['quote'], '')
        self.assertEqual(found['occurrences'][0]['source_refs'], [])

    def test_quarantine_is_visible_only_by_explicit_request_and_never_support(self):
        self.ingest()
        ordinary = store.search_evidence(self.db, '修复意愿')
        explicit = store.search_evidence(self.db, '修复意愿', include_quarantine=True)
        self.assertNotIn('quarantine', ordinary['matches'])
        self.assertTrue(explicit['matches']['quarantine'])
        self.assertTrue(all(not row['eligible_as_support'] for row in explicit['matches']['quarantine']))
        ambiguous = store.get_evidence(self.db, self.data['documents'][1]['events'][0]['evidence_id'])
        self.assertEqual(ambiguous['grounding_status'], 'ambiguous')
        self.assertFalse(ambiguous['eligible_as_support'])

    def test_short_chinese_query_uses_literal_fallback_and_fts_syntax_is_not_executed(self):
        self.ingest()
        short = store.search_evidence(self.db, '修复')
        self.assertEqual(short['match_mode'], 'literal_short_query')
        self.assertTrue(short['matches']['support'])
        injected = store.search_evidence(self.db, '修复意愿 NOT 不存在的关键词')
        self.assertTrue(all(not rows for rows in injected['matches'].values()))
        with self.assertRaises(ValueError):
            store.search_evidence(self.db, '   ')

    def test_counter_results_have_their_own_limit(self):
        self.ingest()
        found = store.search_evidence(self.db, '修复意愿', limit=1)
        self.assertEqual(len(found['matches']['support']), 1)
        self.assertEqual(len(found['matches']['counter']), 1)

    def test_source_message_identifier_is_searchable(self):
        self.ingest()
        message_id = self.data['source_messages'][0]['message_id']
        found = store.search_evidence(self.db, message_id)
        self.assertTrue(found['matches']['support'])
        self.assertTrue(any(ref['message_id'] == message_id for row in found['matches']['support']
            for occurrence in row['occurrences'] for ref in occurrence['source_refs']))

    def test_other_source_or_redaction_policy_is_rejected_without_modifying_store(self):
        self.ingest()
        for options in ({'changed': True}, {'terms': ['不会出现在输入中的合成补充词']}):
            with self.subTest(options=options):
                other = self.root / ('other' + str(len(options)) + str('terms' in options))
                other.mkdir()
                ledger, _, _ = make_ledger(other, **options)
                with self.assertRaisesRegex(ValueError, '快照|脱敏|范围'):
                    store.ingest_ledger(ledger, self.db)
                self.assertEqual(store.search_evidence(self.db, '修复意愿')['source_id'], self.data['source_id'])

    def test_optional_current_ledger_check_rejects_stale_database(self):
        self.ingest()
        other = self.root / 'changed'
        other.mkdir()
        ledger, _, _ = make_ledger(other, changed=True)
        with self.assertRaisesRegex(ValueError, '快照|范围'):
            store.search_evidence(self.db, '修复意愿', ledger_file=ledger)

    def test_invalid_legacy_or_unbound_ledger_is_rejected_before_database_creation(self):
        invalid = [[], {}, {**self.data, 'schema_version': 'chat-evidence-v1'},
                   {k: v for k, v in self.data.items() if k != 'redaction'},
                   {**self.data, 'redaction': {**self.data['redaction'], 'enabled': False}}]
        tampered = copy.deepcopy(self.data)
        tampered['documents'][0]['events'][0]['grounding']['source_refs'][0]['start'] = 999
        invalid.append(tampered)
        for ledger in invalid:
            with self.subTest(kind=str(type(ledger))):
                self.ledger.write_text(json.dumps(ledger), encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.ingest()
                self.assertFalse(self.db.exists())

    def test_old_v2_ledger_can_be_revalidated_locally_against_explicit_input(self):
        del self.data['redaction']
        self.ledger.write_text(json.dumps(self.data), encoding='utf-8')
        with patch('llm_client.call_llm', side_effect=AssertionError('禁止模型请求')):
            imported = store.ingest_ledger(self.ledger, self.db, input_file=self.source)
        self.assertEqual(imported['evidence_count'], 9)
        self.assertFalse((self.root / 'out').exists())

    def test_same_snapshot_replaces_old_extraction_without_stale_fts_rows(self):
        self.ingest()
        self.data['documents'][0]['events'] = []
        self.data['documents'][0] = ground_document(self.data['documents'][0], self.data['source_messages'][:3])
        self.data['grounding_issues'] = grounding_issues(self.data['documents'])
        self.ledger.write_text(json.dumps(self.data), encoding='utf-8')
        updated = self.ingest()
        self.assertEqual(updated['evidence_count'], 7)
        old_id = ground_document({**{key: [] for key in KEYS}, 'events': [{'summary': '修复意愿的直接表达',
            'evidence': '我愿意把误会说清楚。', 'confidence': 'medium'}]}, self.data['source_messages'][:3])['events'][0]['evidence_id']
        self.assertIsNone(store.get_evidence(self.db, old_id))

    def test_transaction_failure_retains_previous_query_results(self):
        self.ingest()
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("CREATE TRIGGER fail_import BEFORE INSERT ON evidence BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END")
        with self.assertRaises(sqlite3.DatabaseError):
            self.ingest()
        self.assertTrue(store.search_evidence(self.db, '修复意愿')['matches']['support'])

    def test_read_queries_do_not_create_missing_databases(self):
        with self.assertRaises(ValueError):
            store.search_evidence(self.db, '修复意愿')
        self.assertFalse(self.db.exists())

    def test_full_workflow_writes_bound_redaction_provenance_and_empty_evidence_is_searchable(self):
        import chunk_pipeline
        from types import SimpleNamespace
        artifacts = build_artifacts(self.source, self.root / 'generated')
        config = SimpleNamespace(api_base='https://example.invalid', api_key='fake', model='synthetic')
        with patch('llm_client.call_llm', side_effect=[json.dumps({key: [] for key in KEYS}),
                '合成报告 <!-- END_FINAL_REPORT -->']):
            chunk_pipeline.run_single_workflow(artifacts, 'skill', config, 1, chunk_sleep=0)
        ledger = json.loads(artifacts['evidence_ledger_file'].read_text(encoding='utf-8'))
        self.assertEqual(ledger['redaction']['source_id'], ledger['source_id'])
        imported = store.ingest_ledger(artifacts['evidence_ledger_file'], self.db)
        self.assertEqual(imported['evidence_count'], 0)
        self.assertTrue(all(not rows for rows in store.search_evidence(self.db, '修复意愿')['matches'].values()))

    def test_cli_ingest_search_and_lookup_use_real_database(self):
        def command(*args):
            result = subprocess.run([sys.executable, str(ROOT / 'agent' / 'evidence_store.py'), *map(str, args)],
                cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        imported = command('ingest', self.ledger, '--db', self.db)
        self.assertEqual(imported['evidence_count'], 9)
        found = command('search', '压力撤退', '--db', self.db, '--ledger', self.ledger)
        self.assertTrue(found['matches']['counter'])
        item = command('show', found['matches']['counter'][0]['evidence_id'], '--db', self.db)
        self.assertTrue(item['occurrences'][0]['source_refs'])
        self.assertEqual(item['role'], 'counter')

    def test_foreign_database_is_preserved(self):
        self.db.parent.mkdir()
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute('CREATE TABLE unrelated(value TEXT)')
            connection.execute("INSERT INTO unrelated VALUES ('keep')")
        with self.assertRaisesRegex(ValueError, '拒绝|数据库'):
            self.ingest()
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute('SELECT * FROM unrelated').fetchall(), [('keep',)])

    def test_wrong_input_cannot_supply_provenance_for_old_ledger(self):
        del self.data['redaction']
        self.ledger.write_text(json.dumps(self.data), encoding='utf-8')
        other = self.root / 'other'
        other.mkdir()
        _, source, _ = make_ledger(other, changed=True)
        with self.assertRaisesRegex(ValueError, '输入|消息索引'):
            store.ingest_ledger(self.ledger, self.db, input_file=source)
        self.assertFalse(self.db.exists())

    def test_malformed_provenance_and_unused_term_options_are_rejected(self):
        self.data['redaction'] = []
        self.ledger.write_text(json.dumps(self.data), encoding='utf-8')
        with self.assertRaises(ValueError):
            store.ingest_ledger(self.ledger, self.db, input_file=self.source)
        self.assertFalse(self.db.exists())
        self.ledger, _, self.data = make_ledger(self.root)
        with self.assertRaises(ValueError):
            store.ingest_ledger(self.ledger, self.db, redaction_terms=['必须显式重新核对输入'])

    def test_cli_missing_database_returns_json_error_without_creating_it(self):
        result = subprocess.run([sys.executable, str(ROOT / 'agent' / 'evidence_store.py'),
            'search', '修复意愿', '--db', str(self.db)], cwd=ROOT, capture_output=True,
            text=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        self.assertEqual(result.returncode, 2)
        self.assertIn('error', json.loads(result.stderr))
        self.assertFalse(self.db.exists())
