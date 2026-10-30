"""证据结构校验与跨分块摘要选取，不把格式完整当成推断正确。"""
from __future__ import annotations

FIELDS = {
    'events': ('summary', 'evidence', 'confidence'),
    'relation_signals': ('model', 'direction', 'signal', 'evidence', 'confidence'),
    'persona_signals': ('trait', 'signal', 'evidence', 'stability', 'confidence'),
    'counter_evidence': ('against', 'evidence', 'confidence'),
    'uncertainties': ('question', 'reason'),
}
SUMMARY_LIMIT = 60
EXTRACTION_SCHEMA_VERSION = 'chat-evidence-v1'


def validate_evidence(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError('证据必须是对象')
    for field, required in FIELDS.items():
        if not isinstance(data.get(field), list):
            raise ValueError(f'证据缺少列表字段 {field}')
        for item in data[field]:
            if not isinstance(item, dict):
                raise ValueError(f'{field} 条目必须是对象')
            if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required):
                raise ValueError(f'{field} 条目缺少非空文本字段 {required}')
            for key, choices in (('confidence', {'high', 'medium', 'low'}),
                                 ('direction', {'support', 'against', 'mixed'}),
                                 ('stability', {'local', 'repeated', 'unclear'})):
                if key in required and item[key] not in choices:
                    raise ValueError(f'{field}.{key} 非法枚举')


def balanced_items(documents, key, limit):
    """按各块条目深度轮询，避免一块占满全部摘要名额。"""
    groups = []
    for doc in documents:
        items = doc.get(key, [])
        if not isinstance(items, list):
            continue
        groups.append([{**item, 'chunk_index': doc.get('chunk_index'),
                        'time_range': doc.get('time_range', {}), 'message_range': doc.get('message_range', {})}
                       for item in items if isinstance(item, dict)])
    # 块数超过预算时均匀覆盖时间轴，而非仅保留最早的块。
    groups = [group for group in groups if group]
    if limit <= 0 or not groups:
        return []
    if len(groups) > limit:
        if limit == 1:
            groups = [groups[-1]]
        else:
            groups = [groups[round(i * (len(groups) - 1) / (limit - 1))] for i in range(limit)]
    selected = []
    for depth in range(max(map(len, groups))):
        for group in groups:
            if depth < len(group):
                selected.append(group[depth])
                if len(selected) == limit:
                    return selected
    return selected
