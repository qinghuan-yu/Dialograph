"""结论输入协议、来源校验与确定性编译；不发起模型请求。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re

from evidence_grounding import match_quote, digest, message_identity, TEXT_TYPES
from privacy_redaction import current_redactor
from claim_facts import (FACTS_MARKER, QUOTE_MARKER, normalize_fact_fields, check_time_prose,
                         check_quoted_prose, source_facts, validate_fact_fields)

VERSION = 'chat-claims-v3'
LEGACY_VERSION = 'chat-claims-v2'
SECTIONS = {
    'analysis': ('互动概况', '对方的表达风格', '对方的互动模式', '可能的关系定位',
                 '可能的动机逻辑', '呈现出的性格倾向', '误判风险', '总结与验证建议'),
    'persona': ('基本画像', '性格与表达风格', '互动模式与边界感', '动机与需求',
                '对方面前呈现的倾向', '不能判断的部分', '最重要的认知'),
}
MARKERS = {'analysis': '<!-- END_FINAL_REPORT -->', 'persona': '<!-- END_PERSONA_REPORT -->'}
KINDS = {'fact': '事实', 'inference': '推断', 'hypothesis': '假设', 'uncertain': '存疑'}
SCOPES = {'local': '局部', 'repeated': '重复出现', 'long_term': '长期'}
CONFIDENCES = {'high': '高', 'medium': '中', 'low': '低'}


@dataclass
class ReportInput:
    prompt: str
    evidence_ids: set[str]
    context_ids: set[str]
    source_positions: frozenset[int] = frozenset()
    retrieval: dict | None = None


def claim_contract(report_kind, *, raw_sources=False):
    headings = '；'.join(f'{n}. {title}' for n, title in enumerate(SECTIONS[report_kind], 1))
    reference = 'P_原文序号' if raw_sources else '完整E编号'
    source_rule = ('支持、反证和情境限制使用本次完整会话原文行的P_序号，逐字对应其自身消息，不能重复或重叠。'
                   '每个被陈述的具体动作须有对应引用，邻文只用于解释指代。') if raw_sources else (
                   '支持、反证和情境限制只引用本次入选的已绑定完整 E 编号，或完整窗口目录中明确列出的 M 原文消息编号，'
                   '不能重复或重叠。M 编号仅说明逐字出处，须按具体命题判断其角色。')
    example = {'claims': [dict(section=1, text='仅描述相关场景中的判断', kind='inference',
        subject='interaction', scope='local', confidence='low', supporting_evidence=[reference],
        counter_evidence=[], limiting_evidence=[], counter_review='not_found_in_selected',
        context_review='说明连续原文如何支持解释、反证是否相关。', uncertainty='还缺少哪些信息。',
        actions=[], time_links=[])]}
    return f'''\n## 结论输出协议
仅输出JSON对象，校验后生成报告。条数由独立证据决定，通常6—10条，素材少时可更少。合并重复判断，省略无据维度。
每条 text 不超过80个汉字，context_review 不超过60个汉字，uncertainty 不超过40个汉字，以免输出截断。
{json.dumps(example, ensure_ascii=False)}
section 为章节序号：{headings}。允许省略无证据的章节。
kind 仅为 fact/inference/hypothesis/uncertain；subject 仅为 peer/interaction/user。
scope为local/repeated/long_term。仅统计supporting_evidence。repeated须至少两个日期、两个独立会话；long_term须至少三个日期、三个会话且跨度至少30天。同日多编号不算多日，范围门槛不证明稳定人格。
confidence为high/medium/low。uncertain 的 scope 固定为 local，confidence 固定为 low，写明uncertainty。
除 uncertain 外均须有 supporting_evidence。{source_rule}
supporting_evidence 为编号数组。counter_evidence 为对象数组，每项为{{"evidence_id":"{reference}","target":"当前text中被削弱的原样片段","reason":"这条原文如何直接削弱该片段"}}。
limiting_evidence 为对象数组，每项为{{"evidence_id":"{reference}","reason":"仅限制什么场景或适用范围"}}。限制与反证分别记录，无关材料不引用；三类引用不得重复或重叠。
证据角色以最终text为准。若text同时描述接受和拒绝，接受与拒绝的原话都支持该判断。已经写入判断的例外须按支持处理，只有与判断中仍然保留的具体断言相反的事实才进入counter_evidence。
inference/hypothesis 及 repeated/long_term 的全部引用必须在“核验引文”对应的连续原文内；未回填的解释写成 uncertain，不能把未展示材料当作不存在。
counter_review 为 found（有反证编号）、not_found_in_selected（仅本次入选材料未找到相关反证）或 not_assessed（尚未判断）；非存疑结论不能为 not_assessed。
context_review 必须说明上下文和支持/反证的相关性；不要照搬抽取标签。uncertainty 写局限、替代解释或待验证信息。
所有叙述断言均须由本条引用支持。
图片、文件、通话等占位记录只证明该类消息出现；附件内容未解析时，注册成功、截图内容和通话结果均保留未知。
用户的期待、投射或特殊关系解读须有用户自己的明确表达。仅有对方的关心、邀约或亲昵称呼时，聚焦这些实际互动及适用范围。
讨论第三人的关系时保留内层话题对象。kind=uncertain 的文字同样须忠于来源，标签不能替代对归属疑点的说明。
反证须直接削弱target。取消活动不自动反驳关心，文字简短不自动反驳情绪支持。情境限制放limiting_evidence，无反证时counter_evidence为空、counter_review为not_found_in_selected。
fact 只记录直接可观察的言语或行为；关系定位、动机、能力和稳定倾向均不能标为 fact 绕过语境或范围要求。local 的文字也必须限定到实际事件，不能用局部标签承载总体判断。
对方的直接人物结论 subject=peer，支持中必须包含对方发送的已核验引文。转发/引用内容不自动代表发送者立场。
所有判断写在claims内。text、context_review、uncertainty、reason和target用纯文本，禁止P/E/M/C编号、HTML或换行。编号只放证据数组和evidence_id字段，text不含Markdown。JSON无需完成标记。

{FACTS_MARKER}
程序复制引文、发言者和日期。结论与理由均属模型解释，fact显示为“原话概括”。
日期、钟点及量化间隔（含次日/数日后）用time_links=[{{"from_id":"{reference}","to_id":"后续引用"}}]，端点须属本条引用。程序计算消息日历日差及秒差，事件关联另核验。
text、context_review、uncertainty及证据reason中的原话复述也须去除时间词。原话时间只保留在actions的quote或程序展示的引文中。
未来意向的执行时间不能用任意后续消息充当端点。
计划、动作及结果填写actions=[{{"evidence_id":"{reference}","quote":"逐字动作原话","actor":null,"target":null,"state":"unknown","confirmation":null}}]。引用须属本条支持，actor/target为quote内逐字指代，歧义时null，转述按内层人物核对。
state为planned意向、reported_done自述完成、confirmed_result另有结果确认、unknown待确认。confirmed_result须附confirmation={{"evidence_id":"另一发送方后续支持编号","quote":"逐字确认原话"}}，其余填null。主体、对象、状态及确认是否属于同一动作均须语义复核。
纯风格/关系解释可用空actions，无需比较时间时time_links为空。

{QUOTE_MARKER}
引号内文字须逐字属于本条引用。概念及改写用普通叙述，含义仍须有依据。
'''


def context_catalog(context_ids, ledger):
    index = _evidence_index(ledger) if ledger is not None else {}
    lines = ['\n## 本轮可用于解释性结论的完整窗口编号']
    for eid in sorted(context_ids):
        if eid.startswith('M_'):
            continue
        ref = index[eid]['grounding']['source_refs'][0]
        speaker = '我' if ref['sender'] == 'me' else '对方'
        lines.append(f'{eid} / {ref["time"]} / {speaker} / 第{ref["position"]}条')
    lines.append('推断、假设和跨时间模式的全部引用必须从以上编号中选择；不同编号绑定同一消息不增加覆盖。'
                 '完整会话原文中展示的P_序号也可引用，其支持范围按该行说话人与时间核算。其他已匹配入选编号只可用于局部事实或明确存疑。')
    return '\n'.join(lines) + '\n'


def _evidence_index(ledger):
    index = {item['evidence_id']: item for document in ledger['documents']
            for category in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')
            for item in document[category]}
    for message in ledger['source_messages']:
        if message['message_id'] != message_identity(message):
            raise ValueError('原文消息编号与快照内容不一致。')
        if message['type'] in TEXT_TYPES:
            item = {'evidence': message['content'], 'source_message_ids': [message['message_id']]}
            index[message['message_id']] = {**item, 'grounding': match_quote(item, [message])}
    return index


def _coverage(refs, messages):
    # 会话采用与原文窗口相同的四小时间隔，日期和跨度只用合法时间。
    sessions, session, previous = {}, 0, None
    times = {}
    for message in messages:
        try:
            time = datetime.strptime(message['time_str'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            time = None
        if previous is None or time is None or abs((time - previous).total_seconds()) > 14400:
            session += 1
        sessions[message['message_id']] = session
        times[message['message_id']] = time
        previous = time
    dates = sorted({times[ref['message_id']].date().isoformat() for ref in refs if times[ref['message_id']]})
    span = (datetime.fromisoformat(dates[-1]) - datetime.fromisoformat(dates[0])).days if dates else 0
    return {'positions': sorted({ref['position'] for ref in refs}), 'dates': dates,
            'sessions': len({sessions[ref['message_id']] for ref in refs}), 'span_days': span}


def _check_claim_text(value, key, *, legacy=False):
    if not isinstance(value, str) or not value.strip() or len(value) > 1200:
        raise ValueError(f'结论文字缺失或过长：{key}')
    pattern = r'[<>\n\r]|\b[EMC]_[0-9a-f]{8,}' if legacy else r'[<>\n\r]|[EMC]_[0-9a-f]{8,}|P_[0-9]+'
    if re.search(pattern, value):
        raise ValueError('结论正文须为纯文本，引用放入证据数组。')


def _reviewed_references(entries, text, *, counter, legacy=False):
    if not isinstance(entries, list):
        raise ValueError('反证与情境限制必须为对象数组。')
    required = {'evidence_id', 'reason', 'target'} if counter else {'evidence_id', 'reason'}
    refs = []
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != required
                or not isinstance(entry['evidence_id'], str)):
            raise ValueError('反证或情境限制字段不完整。')
        _check_claim_text(entry['reason'], '证据相关性说明', legacy=legacy)
        if counter:
            _check_claim_text(entry['target'], '被削弱的判断片段', legacy=legacy)
            if entry['target'] not in text:
                raise ValueError('反证target必须原样指向当前结论text中的片段。')
        refs.append(entry['evidence_id'])
    return refs


def _validate_claim_shape(claim, report_kind, *, facts_enabled):
    required = {'section', 'text', 'kind', 'subject', 'scope', 'confidence', 'supporting_evidence',
                'counter_evidence', 'limiting_evidence', 'counter_review', 'context_review', 'uncertainty'}
    if facts_enabled:
        required.update(('actions', 'time_links'))
    if not isinstance(claim, dict):
        raise ValueError('结论字段不完整或包含未知字段。')
    if set(claim) != required:
        missing, extra = sorted(required - set(claim)), sorted(set(claim) - required)
        raise ValueError(f'结论字段不完整或包含未知字段。缺少 {missing}，多出 {extra}。')
    if type(claim['section']) is not int or not 1 <= claim['section'] <= len(SECTIONS[report_kind]):
        raise ValueError('结论章节无效。')
    for key, choices in (('kind', KINDS), ('scope', SCOPES), ('confidence', CONFIDENCES),
                         ('subject', ('peer', 'interaction', 'user')),
                         ('counter_review', ('found', 'not_found_in_selected', 'not_assessed'))):
        if not isinstance(claim[key], str) or claim[key] not in choices:
            raise ValueError(f'结论枚举无效：{key}')
    for key in ('text', 'context_review', 'uncertainty'):
        _check_claim_text(claim[key], key, legacy=not facts_enabled)


def _validate_claim_references(claim, ledger, prepared, index, messages, *, facts_enabled):
    support = claim['supporting_evidence']
    if not isinstance(support, list) or any(not isinstance(eid, str) for eid in support):
        raise ValueError('支持证据必须为编号数组。')
    counter = _reviewed_references(claim['counter_evidence'], claim['text'], counter=True, legacy=not facts_enabled)
    limits = _reviewed_references(claim['limiting_evidence'], claim['text'], counter=False, legacy=not facts_enabled)
    all_ids = support + counter + limits
    aliases = {mid: alias for alias, mid in (prepared.retrieval or {}).get('source_aliases', {}).items()}
    locations = {}
    for field, entries in (('supporting_evidence', support), ('counter_evidence', counter),
                           ('limiting_evidence', limits)):
        for number, eid in enumerate(entries):
            locations.setdefault(eid, []).append(f'{field}[{number}]')
    reference_errors = []
    for eid, fields in locations.items():
        label = aliases.get(eid, eid)
        if len(fields) > 1:
            reference_errors.append(f'支持、反证与情境限制编号重复或重叠：{label} 位于 '
                                    + '、'.join(fields) + '。按当前判断选择一个证据角色，仅保留一次。')
        if eid not in prepared.evidence_ids or eid not in index:
            reference_errors.append(f'结论引用未知或未入选证据：{label} 位于 ' + '、'.join(fields)
                                    + '。只能使用本次提供的可引用编号，不能为占位记录自行编造编号。')
    if reference_errors and any(eid not in prepared.evidence_ids or eid not in index for eid in all_ids):
        raise ValueError('\n'.join(reference_errors))
    if any(eid.startswith('M_') and eid not in prepared.context_ids for eid in all_ids):
        raise ValueError('原文消息引用必须具有完整连续上下文。')
    refs = []
    for eid in all_ids:
        item = index[eid]
        grounding = item['grounding']
        if grounding['status'] != 'matched':
            raise ValueError('结论证据未核验原文，存疑或隔离条目不能成为支持或反证。')
        bound = grounding['source_refs']
        if len(bound) != 1 or bound[0]['message_id'] not in messages:
            raise ValueError('结论原文锚点无效。')
        message = messages[bound[0]['message_id']]
        if message['source_id'] != ledger['source_id'] or match_quote(item, [message]) != grounding:
            raise ValueError('结论原文绑定不一致。')
        if eid in support:
            refs.extend(bound)
    if reference_errors:
        # 引文已绑定，角色重叠不妨碍同时检查独立的动作和时间错误。
        if facts_enabled:
            try:
                validate_fact_fields(claim, index)
            except ValueError as exc:
                reference_errors.append(str(exc))
        raise ValueError('\n'.join(reference_errors))
    return support, counter, all_ids, refs


def _validate_claim(claim, ledger, prepared, report_kind, index, messages, *, facts_enabled):
    _validate_claim_shape(claim, report_kind, facts_enabled=facts_enabled)
    support, counter, all_ids, refs = _validate_claim_references(
        claim, ledger, prepared, index, messages, facts_enabled=facts_enabled)
    uncertain = claim['kind'] == 'uncertain'
    if not uncertain and not support:
        raise ValueError('非存疑结论必须有支持证据。')
    if uncertain and (claim['confidence'] != 'low' or claim['scope'] != 'local'):
        raise ValueError('存疑结论须为低置信度、局部范围。')
    if (bool(counter) != (claim['counter_review'] == 'found')
            or (not uncertain and claim['counter_review'] == 'not_assessed')):
        raise ValueError('反证编号与复核状态不一致。')
    if (not uncertain and (claim['kind'] != 'fact' or claim['scope'] != 'local')
            and not set(all_ids) <= prepared.context_ids):
        missing = ' '.join(sorted(set(all_ids) - prepared.context_ids))
        raise ValueError('解释性结论缺少连续上下文：' + missing + '；这些编号只可用于局部事实或明确存疑，不能继续作为推断依据。')
    if claim['subject'] == 'peer' and not uncertain and not any(ref['sender'] == 'other' for ref in refs):
        raise ValueError('对方的直接结论必须有对方言语或回应证据。')
    coverage = _coverage(refs, ledger['source_messages'])
    if claim['scope'] != 'local':
        minimum = 2 if claim['scope'] == 'repeated' else 3
        if (len(coverage['dates']) < minimum or coverage['sessions'] < minimum
                or (claim['scope'] == 'long_term' and coverage['span_days'] < 30)):
            raise ValueError(f'结论范围超出支持证据：仅{len(coverage["dates"])}个日期、'
                             f'{coverage["sessions"]}个会话、跨度{coverage["span_days"]}天；'
                             f'范围标签为{claim["scope"]}。须补充合格支持或缩小判断本身。')
    validated = {**claim, 'coverage': coverage}
    if facts_enabled:
        fact_errors = []
        try:
            validated['time_intervals'] = validate_fact_fields(claim, index)
        except ValueError as exc:
            fact_errors.extend(str(exc).splitlines())
        try:
            check_time_prose(claim)
        except ValueError as exc:
            fact_errors.extend(str(exc).splitlines())
        if QUOTE_MARKER in prepared.prompt:
            try:
                check_quoted_prose(claim, index)
            except ValueError as exc:
                fact_errors.extend(str(exc).splitlines())
        if fact_errors:
            raise ValueError('\n'.join(dict.fromkeys(fact_errors)))
    return validated

def _parse_response(response, report_kind):
    body = response.strip()
    marker = MARKERS[report_kind]
    if body.endswith(marker):
        body = body[:-len(marker)].strip()
    if body.startswith('```json\n') and body.endswith('```'):
        body = body[8:-3].strip()
    try:
        return current_redactor().tree(json.loads(body))
    except json.JSONDecodeError as exc:
        raise ValueError('结论响应必须为单个 JSON 对象。') from exc


def _reference_map(prepared, index, messages):
    """编译和纠错共用短编号及已重核的旧引用映射。"""
    aliases = prepared.retrieval.get('source_aliases', {}) if prepared.retrieval is not None else {}
    for alias, mid in aliases.items():
        if mid not in messages or alias != f'P_{messages[mid]["position"]}' or mid not in prepared.context_ids:
            raise ValueError('原文短引用与当前快照或完整会话不一致。')
    references = dict(aliases)
    visible_messages = set(aliases.values())
    for eid, item in index.items():
        refs = item['grounding'].get('source_refs', [])
        if (eid not in prepared.evidence_ids and len(refs) == 1
                and refs[0]['message_id'] in visible_messages
                and match_quote(item, [messages[refs[0]['message_id']]]) == item['grounding']):
            references[eid] = refs[0]['message_id']
    return references


def compile_claims(response, ledger, prepared, report_kind, *, schema_version=VERSION):
    if schema_version not in (VERSION, LEGACY_VERSION):
        raise ValueError('结论账本版本无效。')
    facts_enabled = schema_version == VERSION
    if not facts_enabled and FACTS_MARKER in prepared.prompt:
        raise ValueError('实际输入使用新版事实协议，不能降级账本版本。')
    payload = _parse_response(response, report_kind)
    if (not isinstance(payload, dict) or set(payload) != {'claims'}
            or not isinstance(payload['claims'], list) or not payload['claims']):
        raise ValueError('结论账本须包含非空的结论数组。')
    index = _evidence_index(ledger)
    messages = {message['message_id']: message for message in ledger['source_messages']}
    references = _reference_map(prepared, index, messages)
    claims, seen, errors = [], set(), []
    for number, claim in enumerate(payload['claims'], 1):
        try:
            if isinstance(claim, dict):
                for field in ('supporting_evidence', 'counter_evidence', 'limiting_evidence'):
                    entries = claim.get(field)
                    if not isinstance(entries, list):
                        continue
                    for n, entry in enumerate(entries):
                        if isinstance(entry, str):
                            entries[n] = references.get(entry, entry)
                        elif isinstance(entry, dict) and isinstance(entry.get('evidence_id'), str):
                            entry['evidence_id'] = references.get(entry['evidence_id'], entry['evidence_id'])
                if facts_enabled:
                    normalize_fact_fields(claim, references)
            validated = _validate_claim(claim, ledger, prepared, report_kind, index, messages, facts_enabled=facts_enabled)
        except ValueError as exc:
            details = str(exc)
            if facts_enabled:
                try:
                    check_time_prose(claim)
                except ValueError as time_error:
                    details = '\n'.join(dict.fromkeys(details.splitlines() + str(time_error).splitlines()))
            errors.append(f'第 {number} 条结论：{details}')
            continue
        identity = json.dumps([ledger['source_id'], report_kind, validated], ensure_ascii=False, sort_keys=True)
        cid = 'C_' + hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]
        if cid in seen:
            errors.append(f'第 {number} 条结论：结论重复。')
            continue
        seen.add(cid)
        claims.append({**validated, 'claim_id': cid})
    if errors:
        raise ValueError('\n'.join(errors))
    result = {'schema_version': schema_version, 'source_id': ledger['source_id'], 'report_kind': report_kind,
            'input_sha256': hashlib.sha256(prepared.prompt.encode('utf-8')).hexdigest(),
            'selected_evidence_ids': sorted(prepared.evidence_ids),
            'context_evidence_ids': sorted(prepared.context_ids), 'claims': claims}
    if facts_enabled:
        result['source_facts'] = source_facts(claims, index)
    if (prepared.retrieval is None and re.search(
            r'\n## 本轮命题检索覆盖\n检索材料指纹：S_[0-9a-f]{64}\b', prepared.prompt)):
        raise ValueError('报告输入包含命题检索，但缺少对应检索记录。')
    if prepared.retrieval is not None:
        metadata = prepared.retrieval
        if (metadata['schema_version'] != 'claim-retrieval-v1'
                or metadata['source_id'] != ledger['source_id'] or metadata['ledger_sha256'] != digest(ledger)
                or 'S_' + digest(metadata) not in prepared.prompt):
            raise ValueError('命题检索记录与来源或实际报告输入不一致。')
        result['retrieval'] = metadata
    return result
