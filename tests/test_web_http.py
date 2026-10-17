"""本机 HTTP 集成测试；仅用合成对话，绝不调用付费模型。"""
import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from web_app import WebService, make_handler


class WebHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = WebService(self.root)
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

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    def test_private_api_requires_token_and_rejects_foreign_origin(self):
        self.assertEqual(self.request('GET', '/api/jobs')[0], 403)
        headers = {'X-Dialograph-Token': self.service.token, 'Origin': 'https://untrusted.example'}
        self.assertEqual(self.request('POST', '/api/jobs', '{}', headers)[0], 403)
        self.assertEqual(self.request('GET', '/api/bootstrap', headers={'Host': 'untrusted.example'})[0], 403)
        self.assertEqual(self.request('GET', '/.env')[0], 404)
        self.assertEqual(self.request('GET', '/talks/private.json')[0], 404)

    def test_import_preprocess_poll_and_download(self):
        headers = {'X-Dialograph-Token': self.service.token, 'Content-Type': 'application/json'}
        fixture = Path(__file__).parent / 'fixtures' / 'web_smoke.json'
        code, raw = self.request('POST', '/api/import?name=smoke.json', fixture.read_bytes(), headers)
        self.assertEqual(code, 201)
        chat = json.loads(raw)
        code, raw = self.request('POST', '/api/jobs', json.dumps({'chat_id': chat['id'], 'mode': 'preprocess'}), headers)
        self.assertEqual(code, 202)
        job_id = json.loads(raw)['id']
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            code, raw = self.request('GET', f'/api/jobs/{job_id}', headers=headers)
            state = json.loads(raw)
            if state['status'] not in {'queued', 'running'}:
                break
            time.sleep(.05)
        self.assertEqual(state['status'], 'completed', state.get('message'))
        self.assertEqual(state['stats']['total_messages'], 12)
        code, raw = self.request('GET', f'/api/outputs/{job_id}/stats?download=1', headers=headers)
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)['total_messages'], 12)
        self.assertEqual(self.request('GET', f'/api/outputs/{job_id}/secret', headers=headers)[0], 400)
        restored = WebService(self.root)
        self.assertEqual(restored.snapshot(job_id)['status'], 'completed')

    def test_invalid_json_shapes_return_errors_without_starting_a_job(self):
        headers = {'X-Dialograph-Token': self.service.token, 'Content-Type': 'application/json'}
        for data in [[], {'mode': []}, {'retry_of': []}, {'mode': 'preprocess', 'chunk_size': -1}]:
            with self.subTest(data=data):
                self.assertEqual(self.request('POST', '/api/jobs', json.dumps(data), headers)[0], 400)
        self.assertEqual(self.service.snapshot(), [])


if __name__ == '__main__':
    unittest.main()
