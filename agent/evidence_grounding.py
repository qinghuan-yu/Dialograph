"""确定性消息身份与逐字引文绑定；匹配出处不等于验证推断。"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter

from evidence_validation import FIELDS, validate_evidence

GROUNDING_VERSION = 'literal-grounding-v1'
LEDGER_VERSION = 'chat-evidence-v2'
TEXT_TYPES = {'文本消息', '引用消息'}
MAX_CANDIDATES = 20


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def message_identity(message):
    return 'M_' + digest([message['source_id'], message['original_index'],
                          message.get('source_message_id'), message['sender'],
                          message['time_str'], message['type'], message['content']])[:32]


def identify_messages(raw_messages, messages, session=None):
    """编号绑定规范化输入快照；相同快照不受路径、运行或分块参数影响。"""
    source_id = 'S_' + digest({'session': session or {}, 'messages': raw_messages})
    for position, message in enumerate(messages, start=1):
        message.update(source_id=source_id, position=position)
        message['message_id'] = message_identity(message)
    return source_id


def source_reference(message, start, end):
    result = {key: message[key] for key in ('message_id', 'original_index', 'position', 'sender')}
    result.update(time=message['time_str'], start=start, end=end)
    if 'source_message_id' in message:
        result['source_message_id'] = message['source_message_id']
    return result


def match_quote(item, messages):
    quote = item['evidence']
    hints = item.get('source_message_ids')
    candidates = messages
    if hints is not None:
        if (not isinstance(hints, list) or len(hints) != 1
                or not isinstance(hints[0], str)):
            return {'status': 'unmatched', 'reason': 'invalid_source_ids', 'source_refs': []}
        candidates = [m for m in messages if m['message_id'] == hints[0]]
        if not candidates:
            return {'status': 'unmatched', 'reason': 'source_outside_chunk', 'source_refs': []}
    refs, count = [], 0
    for message in candidates:
        if message['type'] not in TEXT_TYPES:
            continue
        text, offset = message['content'], 0
        while True:
            start = text.find(quote, offset)
            if start < 0:
                break
            count += 1
            if len(refs) < MAX_CANDIDATES:
                refs.append(source_reference(message, start, start + len(quote)))
            offset = start + 1
    if count == 1:
        return {'status': 'matched', 'source_refs': refs}
    if count:
        return {'status': 'ambiguous', 'reason': 'multiple_literal_matches', 'source_refs': [],
                'candidate_count': count, 'candidate_refs': refs}
    return {'status': 'unmatched', 'reason': 'no_literal_match', 'source_refs': []}


def ground_document(document, messages):
    """仅在本块内查找原文，覆盖模型自行填写的编号和绑定结果。"""
    validate_evidence(document)
    if not messages or any('message_id' not in m for m in messages):
        raise ValueError('引文绑定缺少已编号的消息。')
    result = copy.deepcopy(document)
    result.update(schema_version=LEDGER_VERSION, grounding_version=GROUNDING_VERSION,
                  source_id=messages[0]['source_id'])
    counts = Counter()
    for category, required in FIELDS.items():
        items, seen = [], set()
        for original in document[category]:
            # 只保留契约字段，避免把模型伪造的来源和额外正文带入已核验材料。
            item = {key: copy.deepcopy(original[key]) for key in required}
            for key in ('time', 'source_message_ids'):
                if key in original:
                    item[key] = copy.deepcopy(original[key])
            grounding = ({'status': 'not_applicable', 'reason': 'uncertainty_is_not_evidence', 'source_refs': []}
                         if category == 'uncertainties' else match_quote(item, messages))
            payload = {key: item[key] for key in required}
            if 'time' in item:
                payload['time'] = item['time']
            item['evidence_id'] = 'E_' + digest([result['source_id'], category, payload, grounding])[:32]
            item['category'] = category
            item['grounding'] = grounding
            if item['evidence_id'] not in seen:
                seen.add(item['evidence_id'])
                items.append(item)
                counts[grounding['status']] += 1
        result[category] = items
    result['grounding_stats'] = {status: counts[status] for status in
                                 ('matched', 'unmatched', 'ambiguous', 'not_applicable')}
    return result


def grounding_issues(documents):
    return [{**item, 'chunk_index': doc['chunk_index']} for doc in documents for key in FIELDS
            for item in doc[key] if item['grounding']['status'] in ('unmatched', 'ambiguous')]


def validate_grounded_document(document, messages):
    if document.get('grounding_version') != GROUNDING_VERSION:
        raise ValueError('证据缺少当前原文绑定版本，请重新运行。')
    rebuilt = ground_document(document, messages)
    for key in (*FIELDS, 'schema_version', 'grounding_version', 'source_id', 'grounding_stats'):
        if digest(document.get(key)) != digest(rebuilt[key]):
            raise ValueError(f'证据编号或原文绑定被修改：{key}。')


def validate_grounded_ledger(ledger, expected_messages=None):
    """独立审计重算匹配与编号；有当前输入时还核对来源索引。"""
    documents = ledger.get('documents')
    if (not isinstance(documents, list) or any(not isinstance(doc, dict) for doc in documents)
            or type(ledger.get('chunk_count')) is not int or ledger['chunk_count'] != len(documents)):
        raise ValueError('原文绑定分块数组或计数无效。')
    messages = ledger.get('source_messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('原文绑定总表缺少 source_messages。')
    if expected_messages is not None and messages != expected_messages:
        raise ValueError('总表消息索引与当前输入不一致。')
    source_id, ids, raw_indices = ledger.get('source_id'), set(), set()
    if not isinstance(source_id, str) or not source_id.startswith('S_'):
        raise ValueError('无效的 source_id。')
    for position, message in enumerate(messages, start=1):
        if (not isinstance(message, dict) or message.get('source_id') != source_id
                or message.get('position') != position
                or type(message.get('original_index')) is not int
                or message['original_index'] < 0
                or not all(isinstance(message.get(key), str) for key in ('sender', 'time_str', 'type', 'content'))):
            raise ValueError('来源消息字段或位置无效。')
        if message.get('message_id') != message_identity(message) or message['message_id'] in ids:
            raise ValueError('来源消息编号不一致或重复。')
        if message['original_index'] in raw_indices:
            raise ValueError('来源消息原始序号重复。')
        ids.add(message['message_id'])
        raw_indices.add(message['original_index'])
    next_position = 1
    for chunk_index, document in enumerate(ledger['documents'], start=1):
        span = document.get('message_range', {})
        if not isinstance(span, dict):
            raise ValueError('证据分块消息范围必须是对象。')
        start, end = span.get('start'), span.get('end')
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(messages):
            raise ValueError('证据分块消息范围无效。')
        if start != next_position or document.get('chunk_index') != chunk_index:
            raise ValueError('证据分块没有连续覆盖来源消息。')
        expected_time = {'start': messages[start - 1]['time_str'], 'end': messages[end - 1]['time_str']}
        if document.get('time_range') != expected_time:
            raise ValueError('证据分块时间范围与原文不一致。')
        validate_grounded_document(document, messages[start - 1:end])
        next_position = end + 1
    if next_position != len(messages) + 1:
        raise ValueError('证据分块未覆盖全部来源消息。')
    if ledger.get('grounding_issues') != grounding_issues(ledger['documents']):
        raise ValueError('未匹配/歧义证据清单与分块条目不一致。')
    return dict(Counter(item['grounding']['status'] for doc in ledger['documents']
                        for key in FIELDS for item in doc[key]))
