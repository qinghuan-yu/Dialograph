"""仅监听本机的 Dialograph 网页服务，不将工作区作为静态目录。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

from analyze import parse_chat_messages
from config import load_llm_config
from web_worker import OUTPUTS
from conversation_input import validate_weflow

MAX_UPLOAD = 50 * 1024 * 1024
ACTIVE = {'queued', 'running'}


def validate_chat(data):
    validate_weflow(data)
    if not parse_chat_messages(data['messages'])['messages']:
        raise ValueError('没有可分析的有效消息。')
    return data


def timestamp():
    return datetime.now().isoformat(timespec='seconds')


class WebService:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.jobs_dir = self.root / 'analysis' / 'web_jobs'
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.jobs, self.processes = {}, {}
        self.lock = threading.RLock()
        for path in self.jobs_dir.glob('*/state.json'):
            try:
                state = json.loads(path.read_text(encoding='utf-8'))
                if state['status'] in ACTIVE:
                    state.update(status='interrupted', message='服务已重启，可继续上次任务。')
                self.jobs[state['id']] = state
            except (OSError, ValueError, KeyError):
                continue

    def conversations(self):
        rows = []
        for path in sorted((self.root / 'talks').glob('*.json')):
            if path.is_file() and path.resolve().parent == (self.root / 'talks').resolve():
                rows.append({'id': hashlib.sha256(path.name.encode('utf-8')).hexdigest(),
                             'name': path.name, 'bytes': path.stat().st_size})
        return rows

    def chat_path(self, chat_id):
        for row in self.conversations():
            if row['id'] == chat_id:
                return self.root / 'talks' / row['name']
        raise ValueError('未找到选中的对话，请刷新列表。')

    def import_chat(self, filename, raw):
        if len(raw) > MAX_UPLOAD:
            raise ValueError('文件超过 50 MB，请先拆分导出。')
        data = validate_chat(json.loads(raw.decode('utf-8-sig')))
        name = re.sub(r'[^\w\-\u4e00-\u9fff]', '_', Path(filename.replace('\\', '/')).stem)[:60] or 'conversation'
        filename = f'{name}_{uuid.uuid4().hex[:10]}.json'
        talks = self.root / 'talks'
        talks.mkdir(exist_ok=True)
        (talks / filename).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        return next(row for row in self.conversations() if row['name'] == filename)

    def save(self, state):
        folder = self.jobs_dir / state['id']
        temporary = folder / 'state.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
        temporary.replace(folder / 'state.json')

    def snapshot(self, job_id=None):
        with self.lock:
            data = self.jobs.get(job_id) if job_id else sorted(self.jobs.values(), key=lambda x: x['created'], reverse=True)
            if job_id and data is None:
                raise ValueError('任务不存在。')
            return json.loads(json.dumps(data))

    def start(self, request):
        with self.lock:
            if self.processes or any(s['status'] in ACTIVE for s in self.jobs.values()):
                raise ValueError('已有任务正在运行，请等待完成或先停止。')
            retry_id = request.get('retry_of')
            if retry_id is not None and not isinstance(retry_id, str):
                raise ValueError('无效的任务编号。')
            if retry_id:
                state = self.jobs.get(retry_id)
                if not state or state['status'] not in {'failed', 'cancelled', 'interrupted'}:
                    raise ValueError('这个任务不需要恢复。')
                job_dir = self.jobs_dir / retry_id
                options = json.loads((job_dir / 'request.json').read_text(encoding='utf-8'))
                options['resume'] = True
            else:
                mode = request.get('mode')
                if not isinstance(mode, str) or mode not in {'preprocess', 'full'}:
                    raise ValueError('请选择预处理或完整分析。')
                chunk_size = request.get('chunk_size', 1200)
                timeout = request.get('timeout', 600)
                if type(chunk_size) is not int or not 1 <= chunk_size <= 10000:
                    raise ValueError('分块消息数应为 1 至 10000 的整数。')
                if type(timeout) is not int or not 30 <= timeout <= 3600:
                    raise ValueError('请求超时应为 30 至 3600 秒的整数。')
                if mode == 'full' and not load_llm_config(self.root).api_key:
                    raise ValueError('未配置模型密钥。请填写项目根目录 .env，或先运行仅预处理。')
                source = self.chat_path(request.get('chat_id'))
                if source.stat().st_size > MAX_UPLOAD:
                    raise ValueError('文件超过 50 MB，请先拆分导出。')
                raw = source.read_bytes()
                validate_chat(json.loads(raw.decode('utf-8-sig')))
                job_id = uuid.uuid4().hex
                job_dir = self.jobs_dir / job_id
                job_dir.mkdir()
                (job_dir / 'input.json').write_text(raw.decode('utf-8-sig'), encoding='utf-8')
                options = {'mode': mode, 'chunk_size': chunk_size, 'timeout': timeout}
                state = {'id': job_id, 'name': source.name, 'mode': mode, 'created': timestamp(), 'events': []}
            if options['mode'] == 'full' and not load_llm_config(self.root).api_key:
                raise ValueError('未配置模型密钥。请填写项目根目录 .env，或先运行仅预处理。')
            (job_dir / 'request.json').write_text(json.dumps(options), encoding='utf-8')
            state.update(status='queued', stage='prepare', message='等待工作流启动。', files={}, current=0, total=0, completed=0)
            state.pop('finished', None)
            self.jobs[state['id']] = state
            self.save(state)
            threading.Thread(target=self.run, args=(state['id'],), daemon=True).start()
            return self.snapshot(state['id'])

    def run(self, job_id):
        with self.lock:
            state = self.jobs[job_id]
            if state['status'] == 'cancelled':
                return
            try:
                process = subprocess.Popen(
                    [sys.executable, '-u', str(Path(__file__).with_name('web_worker.py')), str(self.root), str(self.jobs_dir / job_id)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                )
            except OSError:
                state.update(status='failed', message='无法启动分析进程。请检查 Python 环境。')
                self.save(state)
                return
            self.processes[job_id] = process
            state['status'] = 'running'
            self.save(state)
        # 单独排空 stderr，避免底层异常输出阻塞子进程；不向前端泄露环境信息。
        def drain_errors():
            with process.stderr:
                while process.stderr.read(4096):
                    pass
        threading.Thread(target=drain_errors, daemon=True).start()
        received_result = False
        received_error = False
        for line in process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            with self.lock:
                if state['status'] == 'cancelled':
                    continue
                if event['type'] == 'result':
                    received_result = True
                    state.update(files=event['files'], stats=event['stats'], diagnostics=event['diagnostics'], warnings=event.get('warnings', []))
                elif event['type'] == 'stats':
                    state.update(stats=event['stats'], diagnostics=event['diagnostics'])
                else:
                    for field in ('stage', 'message', 'current', 'total', 'completed'):
                        if field in event:
                            state[field] = event[field]
                    state['events'] = (state['events'] + [{'time': timestamp(), 'message': event.get('message', '')}])[-150:]
                    if event['type'] == 'error':
                        received_error = True
                        state['status'] = 'failed'
                self.save(state)
        code = process.wait()
        process.stdout.close()
        with self.lock:
            self.processes.pop(job_id, None)
            if state['status'] != 'cancelled':
                if code == 0 and received_result:
                    state['status'] = 'completed'
                else:
                    state['status'] = 'failed'
                    if not received_error:
                        state['message'] = '分析进程未能完整结束，可继续任务。'
            state['finished'] = timestamp()
            self.save(state)

    def cancel(self, job_id):
        with self.lock:
            if not isinstance(job_id, str):
                raise ValueError('无效的任务编号。')
            if job_id not in self.jobs:
                raise ValueError('任务不存在。')
            state = self.jobs[job_id]
            if state['status'] in ACTIVE:
                state.update(status='cancelled', message='任务已停止，已完成的分块保留，可稍后继续。')
                process = self.processes.get(job_id)
                if process:
                    process.terminate()
                self.save(state)
            return self.snapshot(job_id)

    def output_path(self, job_id, key):
        if not re.fullmatch(r'[0-9a-f]{32}', job_id) or key not in OUTPUTS:
            raise ValueError('无效的结果请求。')
        state = self.snapshot(job_id)
        item = state.get('files', {}).get(key)
        if not item:
            raise ValueError('结果尚未生成。')
        folder = (self.jobs_dir / job_id).resolve()
        path = (folder / item['path']).resolve()
        if not path.is_relative_to(folder / 'output') or not path.is_file():
            raise ValueError('无效的结果路径。')
        return path, item


def make_handler(service, port):
    allowed_hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *_):
            pass

        def send(self, status, body, kind='application/json; charset=utf-8', filename=None):
            raw = json.dumps(body, ensure_ascii=False).encode('utf-8') if isinstance(body, (dict, list)) else body
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if filename:
                self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(filename))
            self.end_headers()
            self.wfile.write(raw)

        def authorized(self, api=True):
            if self.headers.get('Host') not in allowed_hosts:
                return False
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://{host}' for host in allowed_hosts}:
                return False
            return not api or secrets.compare_digest(self.headers.get('X-Dialograph-Token', ''), service.token)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if not self.authorized(api=path.startswith('/api/') and path != '/api/bootstrap'):
                return self.send(403, {'error': '请求来源无效，请从本机页面重新打开。'})
            try:
                if path == '/api/bootstrap':
                    config = load_llm_config(service.root)
                    return self.send(200, {'token': service.token, 'configured': bool(config.api_key), 'model': config.model,
                                           'provider': urlparse(config.api_base).hostname or '已配置的服务'})
                if path == '/api/conversations':
                    return self.send(200, service.conversations())
                if path == '/api/jobs':
                    return self.send(200, service.snapshot())
                parts = path.strip('/').split('/')
                if len(parts) == 3 and parts[:2] == ['api', 'jobs']:
                    return self.send(200, service.snapshot(parts[2]))
                if len(parts) == 4 and parts[:2] == ['api', 'outputs']:
                    file, item = service.output_path(parts[2], parts[3])
                    if 'download' in parse_qs(parsed.query):
                        return self.send(200, file.read_bytes(), 'application/octet-stream', file.name)
                    return self.send(200, {'content': file.read_text(encoding='utf-8'), 'kind': item['kind'], 'name': file.name})
                static = {'/': ('index.html', 'text/html'), '/index.html': ('index.html', 'text/html'),
                          '/app.js': ('app.js', 'text/javascript'), '/styles.css': ('styles.css', 'text/css')}
                if path in static:
                    name, kind = static[path]
                    return self.send(200, (service.root / 'frontend' / name).read_bytes(), kind + '; charset=utf-8')
                self.send(404, {'error': '页面不存在。'})
            except (ValueError, OSError) as exc:
                self.send(400, {'error': str(exc) if isinstance(exc, ValueError) else '读取文件失败。'})

        def do_POST(self):
            if not self.authorized():
                return self.send(403, {'error': '请求来源无效，请刷新页面。'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_UPLOAD:
                    return self.send(413, {'error': '请求为空或超过 50 MB。'})
                raw = self.rfile.read(length)
                path = urlparse(self.path).path
                if path == '/api/import':
                    filename = parse_qs(urlparse(self.path).query).get('name', ['conversation.json'])[0]
                    return self.send(201, service.import_chat(filename, raw))
                data = json.loads(raw.decode('utf-8'))
                if not isinstance(data, dict):
                    raise ValueError('请求必须是 JSON 对象。')
                if path == '/api/jobs':
                    return self.send(202, service.start(data))
                if path == '/api/cancel':
                    return self.send(200, service.cancel(data.get('id')))
                self.send(404, {'error': '接口不存在。'})
            except (ValueError, OSError) as exc:
                self.send(400, {'error': str(exc) if isinstance(exc, ValueError) else '无法保存本地文件。'})

    return Handler


def main():
    parser = argparse.ArgumentParser(description='Dialograph 本地网页')
    parser.add_argument('--port', type=int, default=5173)
    args = parser.parse_args()
    service = WebService(Path(__file__).resolve().parents[1])
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(service, args.port))
    print(f'Dialograph: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job_id in list(service.jobs):
            service.cancel(job_id)
        server.server_close()


if __name__ == '__main__':
    main()
