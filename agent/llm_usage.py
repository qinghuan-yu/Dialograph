"""完整分析的本地 API 用量账本；只保存计数，不保存提示词、响应或密钥。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
import uuid


_log = ContextVar('model_usage_log', default=None)
_operation = ContextVar('model_usage_operation', default=None)
TOKEN_FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens',
                'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens', 'reasoning_tokens')


class RequestBudgetExceeded(RuntimeError):
    """串行实测的实际请求额度已耗尽，尚未发起下一次传输。"""


class BoundedRequests:
    """串行实测使用的持久计数；先记尝试再传输，失败也占一次。"""
    def __init__(self, path, limit):
        self.path, self.limit = Path(path), limit
        if type(limit) is not int or limit < 1:
            raise ValueError('请求上限须为正整数。')

    def call(self, transport, *args, **kwargs):
        state = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {'limit': self.limit, 'attempts': 0}
        if state['limit'] != self.limit:
            raise ValueError('同一次实测不能修改已保存的请求上限。')
        if state['attempts'] >= self.limit:
            raise RequestBudgetExceeded('本次实测实际请求上限已达，未发送下一次请求。')
        state['attempts'] += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state), encoding='utf-8')
        temporary.replace(self.path)
        return transport(*args, **kwargs)


def normalize_usage(value):
    usage = value if isinstance(value, dict) else {}

    def count(value):
        return value if type(value) is int and value >= 0 else None

    result = {key: count(usage.get(key)) for key in TOKEN_FIELDS}
    details = usage.get('prompt_tokens_details')
    if result['prompt_cache_hit_tokens'] is None and isinstance(details, dict):
        result['prompt_cache_hit_tokens'] = count(details.get('cached_tokens'))
    prompt, hit, miss = (result[key] for key in ('prompt_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens'))
    if prompt is not None and hit is not None:
        if hit > prompt or (miss is not None and hit + miss != prompt):
            result['prompt_cache_hit_tokens'] = result['prompt_cache_miss_tokens'] = None
        elif miss is None:
            result['prompt_cache_miss_tokens'] = prompt - hit
    details = usage.get('completion_tokens_details')
    if isinstance(details, dict):
        result['reasoning_tokens'] = count(details.get('reasoning_tokens'))
    return result


def summarize(records):
    known = {key: sum(row['usage'][key] is not None for row in records) for key in TOKEN_FIELDS}
    totals = {key: sum(row['usage'][key] for row in records if row['usage'][key] is not None)
              if known[key] else None for key in TOKEN_FIELDS}
    cache = [row['usage'] for row in records if all(row['usage'][key] is not None
             for key in ('prompt_cache_hit_tokens', 'prompt_cache_miss_tokens'))]
    hits = sum(row['prompt_cache_hit_tokens'] for row in cache)
    measured = hits + sum(row['prompt_cache_miss_tokens'] for row in cache)
    return {**totals, 'requests': len(records), 'known_requests': known,
            'failed_requests': sum(row['error_type'] is not None for row in records),
            'usage_unknown_requests': sum(row['usage']['prompt_tokens'] is None
                                         or row['usage']['completion_tokens'] is None for row in records),
            'cache_measured_requests': len(cache), 'cache_unknown_requests': len(records) - len(cache),
            'cache_hit_rate': hits / measured if measured else None}


class UsageLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'model_usage.jsonl'
        self.session = uuid.uuid4().hex
        self.records = ([json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()]
                        if self.path.exists() else [])

    def append(self, record):
        with self.path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.records.append(record)
        self.save_summary()

    def save_summary(self):
        def groups(field):
            return {key: summarize([row for row in self.records if row[field] == key])
                    for key in sorted({row[field] for row in self.records})}
        summary = {'schema_version': 'model-usage-v1', 'totals': summarize(self.records),
                   'stages': groups('stage'), 'models': groups('model'),
                   'latest_session': {'id': self.session, 'totals': summarize(
                       [row for row in self.records if row['session_id'] == self.session])}}
        (self.directory / 'model_usage_summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


@contextmanager
def usage_session(directory):
    log = UsageLog(directory)
    token, operation_token = _log.set(log), _operation.set(None)
    try:
        yield log
    finally:
        _operation.reset(operation_token)
        _log.reset(token)
        log.save_summary()


@contextmanager
def usage_operation(stage, **details):
    operation = {'id': uuid.uuid4().hex, 'stage': stage, 'details': details, 'attempt': 0}
    token = _operation.set(operation)
    try:
        yield
    finally:
        _operation.reset(token)


def record_request(model, data, error_type, duration_ms, max_tokens):
    log = _log.get()
    if log is None:
        return
    operation = _operation.get()
    if operation is not None:
        operation['attempt'] += 1
    data = data if isinstance(data, dict) else {}
    choices = data.get('choices')
    reason = choices[0].get('finish_reason') if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    log.append({'schema_version': 'model-usage-v1', 'session_id': log.session,
                'request_id': uuid.uuid4().hex,
                'time': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'model': model, 'stage': operation['stage'] if operation else 'unscoped',
                'operation_id': operation['id'] if operation else None,
                'details': operation['details'] if operation else {},
                'attempt': operation['attempt'] if operation else 1,
                'duration_ms': duration_ms, 'max_tokens': max_tokens, 'error_type': error_type,
                'finish_reason': reason if reason in ('stop', 'length', 'content_filter', 'tool_calls') else None,
                'usage': normalize_usage(data.get('usage'))})


def track_analysis_usage(function):
    @wraps(function)
    def wrapped(artifacts, *args, **kwargs):
        with usage_session(artifacts['support_dir']):
            return function(artifacts, *args, **kwargs)
    return wrapped
