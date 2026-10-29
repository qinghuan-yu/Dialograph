"""真实本机 HTTP 验证任务隔离和网页检索；仅使用合成消息。"""
import copy
import http.client
import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from web_app import WebService, make_handler
from evidence_grounding import ground_document, grounding_issues
from test_evidence_store import make_ledger


def completed_job(service, job_id='a' * 32, *, changed=False):
    folder = service.jobs_dir / job_id
    folder.mkdir()
    source_ledger, _, ledger = make_ledger(folder, changed=changed)
    output = folder / 'output'
    output.mkdir()
    path = output / 'ledger.json'
    path.write_bytes(source_ledger.read_bytes())
    (folder / 'request.json').write_text(json.dumps({'input_name': 'input.json', 'redaction_terms': []}), encoding='utf-8')
    service.jobs[job_id] = {'id': job_id, 'name': '合成检索任务', 'mode': 'full', 'status': 'completed',
        'created': '2026-10-29T10:00:00', 'events': [],
        'files': {'evidence': {'path': 'output/ledger.json', 'label': '结构化证据', 'kind': 'json'}}}
    service.save(service.jobs[job_id])
    return path, ledger


class WebEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = WebService(self.root)
        self.ledger_path, self.ledger = completed_job(self.service)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.service, 0))
        self.port = self.server.server_address[1]
        self.server.RequestHandlerClass = make_handler(self.service, self.port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, operation, data, *, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        headers = headers if headers is not None else {'X-Dialograph-Token': self.service.token}
        connection.request('POST', '/api/evidence/' + operation,
                           body=json.dumps(data).encode('utf-8'), headers=headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def search(self, **options):
        return self.request('search', {'id': 'a' * 32, 'query': '修复意愿', **options})

    def test_search_and_lookup_return_task_bound_redacted_sources(self):
        with patch('llm_client.call_llm', side_effect=AssertionError('检索不得调用模型')):
            code, result = self.search()
            self.assertEqual(code, 200, result)
            self.assertEqual(result['job_id'], 'a' * 32)
            for group in ('support', 'counter', 'mixed', 'uncertainty'):
                self.assertTrue(result['matches'][group])
            self.assertNotIn('quarantine', result['matches'])
            row = result['matches']['support'][0]
            code, source = self.request('show', {'id': 'a' * 32, 'evidence_id': row['evidence_id'],
                                               'revision': result['ledger_revision']})
            self.assertEqual(code, 200, source)
            ref = source['evidence']['occurrences'][0]['source_refs'][0]
            self.assertEqual(ref['content'][ref['start']:ref['end']], source['evidence']['quote'])
            self.assertNotIn('db_file', result)
            self.assertNotIn('redaction_terms', json.dumps(result))

    def test_authentication_and_origin_are_required_for_both_endpoints(self):
        for operation in ('search', 'show'):
            for headers in ({}, {'X-Dialograph-Token': self.service.token, 'Origin': 'https://untrusted.example'}):
                self.assertEqual(self.request(operation, {'id': 'a' * 32, 'query': '修复'}, headers=headers)[0], 403)

    def test_other_task_cannot_lookup_evidence_from_first_task(self):
        _, other = completed_job(self.service, 'b' * 32, changed=True)
        code, result = self.search(id='b' * 32)
        self.assertEqual(code, 200, result)
        self.assertEqual(result['source_id'], other['source_id'])
        first_id = self.ledger['documents'][0]['events'][0]['evidence_id']
        self.assertEqual(self.request('show', {'id': 'b' * 32, 'evidence_id': first_id,
            'revision': result['ledger_revision']})[0], 400)

    def test_retried_or_preprocessing_task_cannot_query_previous_results(self):
        for change in ({'status': 'running'}, {'mode': 'preprocess'}):
            saved = copy.deepcopy(self.service.jobs['a' * 32])
            self.service.jobs['a' * 32].update(change)
            self.assertEqual(self.search()[0], 400)
            self.service.jobs['a' * 32] = saved

    def test_client_cannot_choose_files_or_quarantine_and_invalid_parameters_fail(self):
        for options in ({'db': '../.env'}, {'ledger': '../input.json'}, {'include_quarantine': True},
                        {'id': '../other'}, {'id': []}, {'query': []}, {'query': ' '}, {'limit': True}):
            with self.subTest(options=options):
                self.assertEqual(self.search(**options)[0], 400)

    def test_manifest_path_cannot_escape_task_output(self):
        self.service.jobs['a' * 32]['files']['evidence']['path'] = 'input.json'
        self.assertEqual(self.search()[0], 400)

    def test_copied_other_ledger_is_rejected_against_actual_task_snapshot(self):
        other = self.root / 'other'
        other.mkdir()
        ledger, _, _ = make_ledger(other, changed=True)
        self.ledger_path.write_bytes(ledger.read_bytes())
        code, result = self.search()
        self.assertEqual(code, 400)
        self.assertIn('error', result)

    def test_changed_ledger_rebuilds_index_and_old_source_request_is_rejected(self):
        code, old = self.search()
        self.assertEqual(code, 200, old)
        self.ledger['documents'][0]['events'][0]['summary'] = '修复意愿：新的抽取描述'
        self.ledger['documents'][0] = ground_document(self.ledger['documents'][0], self.ledger['source_messages'][:3])
        self.ledger['grounding_issues'] = grounding_issues(self.ledger['documents'])
        self.ledger_path.write_text(json.dumps(self.ledger), encoding='utf-8')
        code, new = self.search()
        self.assertEqual(code, 200, new)
        self.assertNotEqual(old['ledger_revision'], new['ledger_revision'])
        self.assertTrue(any('新的抽取描述' in row['payload'].get('summary', '') for row in new['matches']['support']))
        self.assertEqual(self.request('show', {'id': 'a' * 32,
            'evidence_id': old['matches']['support'][0]['evidence_id'], 'revision': old['ledger_revision']})[0], 400)

    def test_legacy_v2_uses_local_snapshot_revalidation_without_changing_ledger(self):
        del self.ledger['redaction']
        self.ledger_path.write_text(json.dumps(self.ledger), encoding='utf-8')
        before = self.ledger_path.read_bytes()
        self.assertEqual(self.search()[0], 200)
        self.assertEqual(self.ledger_path.read_bytes(), before)

    def test_concurrent_queries_share_one_index_build(self):
        import evidence_store
        # 外部数据库调用可替换；原始校验和 SQLite 入库仍真实执行。
        with patch('evidence_store.ingest_ledger', wraps=evidence_store.ingest_ledger) as ingest:
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(lambda _: self.search(), range(2)))
            self.assertTrue(all(code == 200 for code, _ in responses), responses)
            self.assertEqual(ingest.call_count, 1)

    def test_changed_snapshot_or_terms_cannot_reuse_cached_results(self):
        code, old = self.search()
        self.assertEqual(code, 200, old)
        folder = self.ledger_path.parent.parent
        for path, transform in (
            (folder / 'input.json', lambda data: {**data, 'messages': [
                {**message, 'content': message['content'] + '新版本'} for message in data['messages']]}),
            (folder / 'request.json', lambda data: {**data, 'redaction_terms': ['修复意愿']}),
        ):
            with self.subTest(file=path.name):
                before = path.read_bytes()
                path.write_text(json.dumps(transform(json.loads(before))), encoding='utf-8')
                self.assertEqual(self.search()[0], 400)
                self.assertEqual(self.request('show', {'id': 'a' * 32,
                    'evidence_id': old['matches']['support'][0]['evidence_id'],
                    'revision': old['ledger_revision']})[0], 400)
                path.write_bytes(before)

    def test_terms_changed_during_query_do_not_deliver_previous_results(self):
        retrieve = self.service.evidence_index.retrieve

        def change_after_query(*args, **kwargs):
            result = retrieve(*args, **kwargs)
            request = self.ledger_path.parent.parent / 'request.json'
            data = json.loads(request.read_text(encoding='utf-8'))
            data['redaction_terms'] = ['修复意愿']
            request.write_text(json.dumps(data), encoding='utf-8')
            return result

        with patch.object(self.service.evidence_index, 'retrieve', side_effect=change_after_query):
            self.assertEqual(self.search()[0], 400)
