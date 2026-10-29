"""默认本地假名化：原始输入保留，分析材料不携带已识别身份字段。"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path

from conversation_input import IDENTITY_FIELDS
from conversation_formats import MESSAGE_TYPES, TIME_FORMAT

POLICY_VERSION = 'local-redaction-v1'
LABELS = {'phone': '手机号', 'email': '邮箱', 'identity': '证件', 'card': '银行卡',
          'url': '链接', 'account': '账号', 'custom': '自定义'}
RESERVED = {'我', '对方', 'me', 'other', 'unknown'}
PLACEHOLDER = re.compile(r'(\[(?:手机号|邮箱|证件|银行卡|链接|账号|自定义)\d+\]|\[参与者\])')
ENUM_FIELDS = {'confidence': {'high', 'medium', 'low'}, 'direction': {'support', 'against', 'mixed'},
               'stability': {'local', 'repeated', 'unclear'},
               'schema_version': {'chat-evidence-v1', 'chat-evidence-v2'}}
_ACTIVE = ContextVar('dialograph_redactor', default=None)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def validate_redaction_terms(terms):
    if not isinstance(terms, list) or len(terms) > 200:
        raise ValueError('补充脱敏词应为最多200项的文本列表。')
    if any(not isinstance(term, str) or not term.strip() or len(term) > 500 for term in terms):
        raise ValueError('每个补充脱敏词必须为1至500字的非空文本。')
    return sorted({term.strip() for term in terms})


def load_redaction_terms(path):
    if path is None:
        return []
    text = Path(path).read_text(encoding='utf-8-sig')
    terms = json.loads(text) if Path(path).suffix.lower() == '.json' else text.splitlines()
    return validate_redaction_terms(terms)


class Redactor:
    def __init__(self, aliases=None, custom_terms=None):
        self.aliases = aliases or {}
        self.custom_terms = validate_redaction_terms([] if custom_terms is None else custom_terms)
        self.counts, self.tokens, self.serials = Counter(), {}, Counter()
        patterns = [
            ('technical', r'(?:[ME]_[0-9a-f]{32}|S_[0-9a-f]{64})(?![0-9a-f])|high\|medium\|low|support\|against\|mixed|local\|repeated\|unclear|chat-evidence-v[12]'),
            ('url', r'(?:https?://|www\.)[^\s<>"\[\]，。；！？、]+'),
            ('email', r'(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z])'),
            ('identity', r'(?<![0-9A-Za-z])(?:[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]|[1-9]\d{5}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3})(?![0-9A-Za-z])'),
            ('card', r'(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)'),
            ('account', r'(?i:wxid_[A-Za-z0-9_-]+|(?:QQ|微信号?|WeChat)\s*[:：=]\s*[A-Za-z0-9_-]{5,64})'),
            ('phone', r'(?<!\d)(?:(?:\+?86[ -]?)?1[3-9]\d[ -]?\d{4}[ -]?\d{4}|0\d{2,3}[ -]?\d{7,8}|\+[1-9](?:[ -]?\d){6,14})(?!\d)'),
        ]
        if self.aliases:
            patterns.append(('known', '|'.join(re.escape(value) for value in sorted(self.aliases, key=lambda s: (-len(s), s)))))
        if self.custom_terms:
            patterns.append(('custom', '|'.join(re.escape(value) for value in sorted(self.custom_terms, key=lambda s: (-len(s), s)))))
        self.pattern = re.compile('|'.join(f'(?P<{kind}>{pattern})' for kind, pattern in patterns))

    def redact(self, text):
        def substitute(match):
            kind, value = match.lastgroup, match.group()
            if kind == 'technical':
                return value
            self.counts[kind] += 1
            if kind == 'known':
                return self.aliases[value]
            key = (kind, re.sub(r'[ -]', '', value) if kind in ('phone', 'card') else value)
            if kind == 'phone' and key[1].lstrip('+').startswith('86') and len(re.sub(r'\D', '', key[1])) == 13:
                key = (kind, re.sub(r'\D', '', key[1])[-11:])
            if key not in self.tokens:
                self.serials[kind] += 1
                self.tokens[key] = f'[{LABELS[kind]}{self.serials[kind]}]'
            return self.tokens[key]
        return ''.join(part if PLACEHOLDER.fullmatch(part) else self.pattern.sub(substitute, part)
                       for part in PLACEHOLDER.split(text))

    def tree(self, value):
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, list):
            return [self.tree(item) for item in value]
        if isinstance(value, dict):
            return {key: item if isinstance(item, str) and item in ENUM_FIELDS.get(key, set()) else self.tree(item)
                    for key, item in value.items()}
        return value

    def response(self, text):
        # 先解码结构化内容，再替换自然语言字段；保留机器枚举，避免昵称重名破坏契约。
        decoder = json.JSONDecoder()
        for match in re.finditer(r'[\[{]', text):
            try:
                value, end = decoder.raw_decode(text[match.start():])
            except ValueError:
                continue
            if isinstance(value, (dict, list)):
                return (self.redact(text[:match.start()])
                        + json.dumps(self.tree(value), ensure_ascii=False)
                        + self.redact(text[match.start() + end:]))
        return self.redact(text)


def redact_conversation(data, custom_terms=None):
    session = data.get('session', {})
    aliases = {}
    def alias(value, role):
        if isinstance(value, (str, int)) and str(value).strip() and str(value) not in RESERVED:
            value = str(value)
            aliases[value] = role if value not in aliases or aliases[value] == role else '[参与者]'
    for field in ('displayName', 'nickname', 'id', 'username', 'userName'):
        alias(session.get(field), '对方')
    alias(session.get('myDisplayName'), '我')
    for message in data['messages']:
        for field in (*IDENTITY_FIELDS, 'senderName', 'senderDisplayName', 'senderNickname'):
            alias(message.get(field), '我' if message['isSend'] else '对方')
    redactor = Redactor(aliases, custom_terms)
    identity = {key: session[key] for key in ('id', 'username', 'userName', 'displayName', 'nickname', 'myDisplayName')
                if key in session}
    token = fingerprint(identity or data)[:12]
    safe = {'session': {'id': f'conversation:{token}', 'type': 'private', 'input_fingerprint': fingerprint(data),
                        'displayName': '对方', 'myDisplayName': '我'}, 'messages': []}
    for index, message in enumerate(data['messages']):
        safe_message = {'type': message['type'] if message['type'] in MESSAGE_TYPES.values() else '其他消息',
                        'content': redactor.tree(message.get('content')), 'isSend': message['isSend']}
        try:
            datetime.strptime(message['formattedTime'], TIME_FORMAT)
            safe_message['formattedTime'] = message['formattedTime']
        except ValueError:
            safe_message['formattedTime'] = redactor.redact(message['formattedTime'])
        if 'createTime' in message:
            timestamp = message['createTime']
            if type(timestamp) in (int, float, str):
                masked = redactor.redact(str(timestamp))
                safe_message['createTime'] = timestamp if masked == str(timestamp) else masked
            else:
                safe_message['createTime'] = None
        if 'source_message_id' in message:
            safe_message['source_message_id'] = f'source:{index + 1}'
        safe['messages'].append(safe_message)
    metadata = {'enabled': True, 'policy_version': POLICY_VERSION,
                'policy_sha256': fingerprint([POLICY_VERSION, redactor.custom_terms]),
                'custom_term_count': len(redactor.custom_terms), 'replacement_counts': dict(redactor.counts)}
    return safe, metadata, redactor, f'对话_{token}'


def current_redactor():
    redactor = _ACTIVE.get()
    return redactor if redactor is not None else Redactor()


def redact_artifact_call(position=0):
    """将一次报告调用的规则限定在当前上下文，避免不同任务串用词表。"""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            artifacts = kwargs.get('artifacts') if 'artifacts' in kwargs else args[position]
            token = _ACTIVE.set(artifacts.get('_redactor'))
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                redactor = current_redactor()
                exc.args = tuple(redactor.tree(value) for value in exc.args)
                for field in ('reason', 'msg'):
                    value = getattr(exc, field, None)
                    if isinstance(value, str):
                        try:
                            setattr(exc, field, redactor.redact(value))
                        except AttributeError:
                            # HTTPError.reason 是 msg 的只读别名，随后处理 msg。
                            continue
                raise
            finally:
                _ACTIVE.reset(token)
        return wrapped
    return decorate
