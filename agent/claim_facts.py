"""从已核验引文生成事实与时间，单独约束模型的动作判读。"""
import re

from analyze import parse_message_time

FACTS_MARKER = '## 原文事实与模型解释（v3）'
QUOTE_MARKER = '## 逐字引语校验（v1）'
ACTION_STATES = {'planned': '计划或意向', 'reported_done': '自述完成',
                 'confirmed_result': '记录中的结果确认', 'unknown': '状态待确认'}

# 明确日期和量化间隔统一交给时间字段；定性的先后关系仍需语义复核。
TIME_PROSE = re.compile(
    r'\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?|\d{1,2}:\d{2}'
    r'|[0-9零〇一二两三四五六七八九十百]+年(?:[0-9一二三四五六七八九十]+月)?'
    r'|[0-9一二三四五六七八九十]+月[0-9一二三四五六七八九十]+[日号]'
    r'|[0-9零〇一二两三四五六七八九十百半几数]+(?:个半|个多|个|多)?(?:年|日(?!期)|天|周|星期|月|小时|分钟|秒)'
    r'|次日|翌日|隔天|当晚|当天|同日(?!期)|明天|昨天|后天|前天')


def normalize_fact_fields(claim, references):
    """可选数组的省略规范化为空数组，短引用沿用现有快照映射。"""
    for field in ('actions', 'time_links'):
        claim.setdefault(field, [])
        if not isinstance(claim[field], list):
            continue
        for row in claim[field]:
            if not isinstance(row, dict):
                continue
            for key in ('evidence_id', 'from_id', 'to_id'):
                if isinstance(row.get(key), str):
                    row[key] = references.get(row[key], row[key])
            confirmation = row.get('confirmation')
            if isinstance(confirmation, dict) and isinstance(confirmation.get('evidence_id'), str):
                confirmation['evidence_id'] = references.get(confirmation['evidence_id'], confirmation['evidence_id'])


def check_time_prose(claim):
    if not isinstance(claim, dict):
        return
    texts = [(field, claim.get(field)) for field in ('text', 'context_review', 'uncertainty')]
    for field in ('counter_evidence', 'limiting_evidence'):
        entries = claim.get(field)
        if isinstance(entries, list):
            texts.extend((f'{field}[{n}].reason', row.get('reason'))
                         for n, row in enumerate(entries) if isinstance(row, dict))
    errors = []
    for field, text in texts:
        if isinstance(text, str):
            for match in TIME_PROSE.finditer(text):
                errors.append(f'{field}中的“{match.group()}”涉及具体时间，须用time_links交给程序计算，叙述保留定性解释。')
    if errors:
        raise ValueError('\n'.join(dict.fromkeys(errors)))


def source_facts(claims, evidence_index):
    ids = set()
    for claim in claims:
        ids.update(claim['supporting_evidence'])
        ids.update(row['evidence_id'] for field in ('counter_evidence', 'limiting_evidence') for row in claim[field])
    rows = []
    for eid in ids:
        item = evidence_index[eid]
        ref = item['grounding']['source_refs'][0]
        rows.append({'evidence_id': eid, 'message_id': ref['message_id'], 'position': ref['position'],
                     'sender': ref['sender'], 'time': ref['time'], 'quote': item['evidence']})
    return sorted(rows, key=lambda row: (row['position'], row['evidence_id']))


def check_quoted_prose(claim, evidence_index):
    """引号专用于逐字原话，概念与改写用普通叙述表达。"""
    quotes = [row['quote'] for row in source_facts([claim], evidence_index)]
    texts = [(field, claim[field]) for field in ('text', 'context_review', 'uncertainty')]
    texts.extend((f'{field}[{n}].reason', row['reason'])
                 for field in ('counter_evidence', 'limiting_evidence') for n, row in enumerate(claim[field]))
    errors = []
    for field, text in texts:
        for match in re.finditer(r'“([^”]+)”|「([^」]+)」|"([^"]+)"', text):
            quoted = next(value for value in match.groups() if value is not None)
            if not any(quoted in original for original in quotes):
                errors.append(f'{field}引语“{quoted}”不在本条引用中。补充该原话自身引用；概念或改写请用普通叙述，仍须核验含义。')
    if errors:
        raise ValueError('\n'.join(dict.fromkeys(errors)))


def _validate_action(action, support, facts):
    if (not isinstance(action, dict)
            or set(action) != {'evidence_id', 'quote', 'actor', 'target', 'state', 'confirmation'}
            or not isinstance(action['evidence_id'], str)
            or not isinstance(action['state'], str) or action['state'] not in ACTION_STATES):
        raise ValueError('动作判读须绑定本条支持引文及有效状态。')
    if action['evidence_id'] not in support:
        raise ValueError(f'动作编号{action["evidence_id"]}未登记为本条支持引文。须补充其支持引用或重新核对动作依据。')
    original = facts[action['evidence_id']]
    quote = action['quote']
    if not isinstance(quote, str) or not quote.strip() or quote not in original['quote']:
        raise ValueError('动作原话须逐字属于所引用的消息，不能借用相邻消息。')
    errors = []
    for field, label in (('actor', '主体'), ('target', '对象')):
        value = action[field]
        if value is not None and (not isinstance(value, str) or not value.strip() or value not in quote):
            errors.append(f'动作{label}须是动作原话中的逐字指代，归属不明时填null。')
    if errors:
        raise ValueError('\n'.join(errors))
    confirmation = action['confirmation']
    if action['state'] != 'confirmed_result':
        if confirmation is not None:
            raise ValueError('结果确认依据仅用于confirmed_result，其他状态填null。')
        return
    if (not isinstance(confirmation, dict) or set(confirmation) != {'evidence_id', 'quote'}
            or not isinstance(confirmation['evidence_id'], str)
            or confirmation['evidence_id'] not in support):
        raise ValueError('结果确认须提供本条另一发送方的支持引文。')
    corroboration = facts[confirmation['evidence_id']]
    if (corroboration['sender'] == original['sender']
            or corroboration['position'] <= original['position']
            or not isinstance(confirmation['quote'], str) or not confirmation['quote'].strip()
            or confirmation['quote'] not in corroboration['quote']):
        raise ValueError('结果确认须来自后续另一发送方的逐字原话。')


def _validate_time_link(link, facts, seen):
    if (not isinstance(link, dict) or set(link) != {'from_id', 'to_id'}
            or any(not isinstance(link[key], str) or link[key] not in facts for key in ('from_id', 'to_id'))):
        raise ValueError('时间关系只填写本条引用的from_id与to_id，间隔由程序计算。')
    first, last = facts[link['from_id']], facts[link['to_id']]
    start, end = parse_message_time(first['time']), parse_message_time(last['time'])
    pair = (first['message_id'], last['message_id'])
    if first['position'] >= last['position'] or pair in seen or (start and end and end < start):
        raise ValueError('时间关系须按来源顺序指向不同消息，不能重复或倒置。')
    seen.add(pair)
    known = start is not None and end is not None
    return {**link, 'calendar_days': (end.date() - start.date()).days if known else None,
            'elapsed_seconds': int((end - start).total_seconds()) if known else None}


def validate_fact_fields(claim, evidence_index):
    """逐项收集可独立核验的动作与时间错误，状态含义继续交给语义复核。"""
    facts = {row['evidence_id']: row for row in source_facts([claim], evidence_index)}
    actions, links = claim['actions'], claim['time_links']
    errors, intervals, seen = [], [], set()
    if not isinstance(actions, list):
        errors.append('actions须为动作判读数组。')
    else:
        for number, action in enumerate(actions):
            try:
                _validate_action(action, claim['supporting_evidence'], facts)
            except ValueError as exc:
                errors.append(f'actions[{number}]：{exc}')
    if not isinstance(links, list):
        errors.append('time_links须为时间关系数组。')
    else:
        for number, link in enumerate(links):
            try:
                intervals.append(_validate_time_link(link, facts, seen))
            except ValueError as exc:
                errors.append(f'time_links[{number}]：{exc}')
    if errors:
        raise ValueError('\n'.join(errors))
    return intervals
