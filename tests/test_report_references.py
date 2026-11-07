"""报告缩写引用解析和单次原文回查；只使用合成账本。"""
import http.client
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from test_evidence_store import make_ledger
import test_web_evidence as web_evidence_tests
import report_generation
import chunk_evidence


class ReportReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        _, _, self.ledger = make_ledger(Path(self.temp.name))

    def catalog(self):
        from report_references import ReferenceCatalog
        return ReferenceCatalog(self.ledger)

    def test_unique_evidence_prefix_resolves_and_returns_original_quote(self):
        item = self.ledger['documents'][0]['events'][0]
        token = item['evidence_id'][:10] + '…'
        catalog = self.catalog()
        self.assertEqual(catalog.describe(token)['canonical'], item['evidence_id'])
        result = catalog.read(token)
        ref = result['occurrences'][0]['source_refs'][0]
        self.assertEqual(ref['content'][ref['start']:ref['end']], item['evidence'])

    def test_message_and_chunk_references_show_sources_as_messages_and_passages(self):
        catalog = self.catalog()
        message = self.ledger['source_messages'][0]
        result = catalog.read(message['message_id'][:10] + '...')
        self.assertEqual(result['kind'], 'message')
        self.assertEqual(result['occurrences'][0]['source_refs'][0]['content'], message['content'])
        passage = catalog.read('chunk_001')
        self.assertEqual(passage['kind'], 'chunk')
        refs = passage['occurrences'][0]['source_refs']
        self.assertEqual([ref['position'] for ref in refs], [1, 2, 3])
        self.assertTrue(all(ref['start'] == ref['end'] == 0 for ref in refs))

    def test_compact_chunk_groups_resolve_each_named_passage(self):
        references = self.catalog().describe_report('参见 chunk_001/002')
        self.assertEqual(set(references), {'chunk_001', 'chunk_002'})
        self.assertTrue(all(row['status'] == 'resolved' for row in references.values()))

    def test_colliding_missing_and_quarantined_references_do_not_resolve_to_sources(self):
        first, second = self.ledger['documents'][0]['events']
        first['evidence_id'] = 'E_12345678' + '0' * 24
        second['evidence_id'] = 'E_12345678' + '1' * 24
        catalog = self.catalog()
        self.assertEqual(catalog.describe('E_12345678…')['status'], 'ambiguous')
        self.assertEqual(catalog.describe('E_' + 'f' * 32)['status'], 'missing')
        self.assertEqual(catalog.describe(second['evidence_id'])['status'], 'quarantined')
        for token in ('E_12345678…', second['evidence_id'], 'chunk_999'):
            with self.subTest(token=token), self.assertRaises(ValueError):
                catalog.read(token)

    def test_prompt_summary_uses_evidence_ids_and_times_without_message_or_chunk_ids(self):
        summary = chunk_evidence.format_evidence_summary(self.ledger['documents'])
        self.assertIn(self.ledger['documents'][0]['events'][0]['evidence_id'], summary)
        self.assertNotIn('M_', summary)
        self.assertNotIn('chunk_', summary)
        self.assertNotIn('原始序号', summary)

    def test_prompts_request_complete_evidence_ids_and_start_with_report_content(self):
        final = report_generation.build_final_prompt('对象', 'stats', 'coverage', 'evidence', ['phase'])
        persona = report_generation.build_persona_prompt('对象', 'stats', 'coverage', 'evidence', ['phase'], 'relation')
        for prompt in (final, persona):
            self.assertIn('完整的 E_ 加 32 位十六进制编号', prompt)
            self.assertIn('校验后生成报告', prompt)
            self.assertIn('“我”的自述不能写成对方说过的话', prompt)
        self.assertNotIn('先写一段 2 到 4 句声明', final)


class ReportReferenceHTTPTests(unittest.TestCase):
    setUp = web_evidence_tests.WebEvidenceTests.setUp
    tearDown = web_evidence_tests.WebEvidenceTests.tearDown
    request = web_evidence_tests.WebEvidenceTests.request

    def report(self, text):
        path = self.ledger_path.parent / 'report.md'
        path.write_text(text, encoding='utf-8')
        self.service.jobs['a' * 32]['files']['analysis'] = {
            'path': 'output/report.md', 'kind': 'markdown', 'label': '关系报告'}
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        connection.request('GET', '/api/outputs/' + 'a' * 32 + '/analysis',
                           headers={'X-Dialograph-Token': self.service.token})
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_report_supplies_resolved_reference_map_and_one_request_loads_source(self):
        item = self.ledger['documents'][0]['events'][0]
        token = item['evidence_id'][:10] + '…'
        status, report = self.report('结论 ' + token + ' / chunk_001')
        self.assertEqual(status, 200)
        info = report['citations']
        self.assertEqual(info['references'][token]['canonical'], item['evidence_id'])
        status, response = self.request('reference', {'id': 'a' * 32, 'reference': token,
                                                     'revision': info['ledger_revision']})
        self.assertEqual(status, 200, response)
        self.assertEqual(response['reference']['quote'], item['evidence'])

    def test_direct_reference_preserves_revision_task_and_auth_boundaries(self):
        _, report = self.report('片段 chunk_001')
        payload = {'id': 'a' * 32, 'reference': 'chunk_001', 'revision': report['citations']['ledger_revision']}
        self.assertEqual(self.request('reference', payload, headers={})[0], 403)
        self.assertEqual(self.request('reference', {**payload, 'revision': '0' * 64})[0], 400)
        self.assertEqual(self.request('reference', {**payload, 'path': '../input.json'})[0], 400)
        self.service.jobs['a' * 32]['status'] = 'running'
        self.assertEqual(self.request('reference', payload)[0], 400)

    def test_reading_and_direct_lookup_do_not_require_a_fulltext_index(self):
        with patch('evidence_store.ingest_ledger', side_effect=ValueError('阅读不需要全文检索库')):
            status, report = self.report('来源 chunk_001')
            self.assertEqual(status, 200, report)
            status, result = self.request('reference', {'id': 'a' * 32, 'reference': 'chunk_001',
                                                       'revision': report['citations']['ledger_revision']})
            self.assertEqual(status, 200, result)
