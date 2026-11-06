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
    if not groups:
        return []
    if limit is None:
        return [item for group in groups for item in group]
    if limit <= 0:
        return []
    if len(groups) > limit:
        if limit == 1:
            groups = [groups[-1]]
        else:
            groups = [groups[round(i * (len(groups) - 1) / (limit - 1))] for i in range(limit)]
    selected_pairs = []
    for depth in range(max(map(len, groups))):
        candidates = [idx for idx, group in enumerate(groups) if depth < len(group)]
        remaining = limit - len(selected_pairs)
        if len(candidates) > remaining:
            if remaining == 1:
                candidates = [candidates[(len(candidates) - 1) // 2]]
            else:
                candidates = [candidates[round(i * (len(candidates) - 1) / (remaining - 1))] for i in range(remaining)]
        selected_pairs.extend((idx, depth) for idx in candidates)
        if len(selected_pairs) >= limit:
            break
    selected_pairs.sort()
    return [groups[idx][depth] for idx, depth in selected_pairs]
