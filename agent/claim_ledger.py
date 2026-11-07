"""将模型结论约束在本轮证据内，重算覆盖范围，再确定性生成报告。"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime
import hashlib
import html
import json
import re

from evidence_grounding import match_quote, digest, message_identity, TEXT_TYPES
from privacy_redaction import current_redactor
from claim_semantic_review import REVIEW_MARKER, review_claims, audit_semantic_review, SemanticReviewError
from claim_facts import (FACTS_MARKER, QUOTE_MARKER, ACTION_STATES, normalize_fact_fields, check_time_prose,
                         check_quoted_prose,
                         source_facts, validate_fact_fields)

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


def _validate_claim(claim, ledger, prepared, report_kind, index, messages, *, facts_enabled):
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


def apply_claim_revisions(payload, base, targets):
    """整体替换复核失败条目，保留其余条目，完整结果仍需来源与语义复核。"""
    if not isinstance(payload, dict) or set(payload) != {'revisions'} or not isinstance(payload['revisions'], list):
        raise ValueError('语义纠正须返回revisions数组。')
    rows = payload['revisions']
    if (any(not isinstance(row, dict) or set(row) != {'claim_number', 'changes'}
            or type(row['claim_number']) is not int or not 1 <= row['claim_number'] <= len(base['claims'])
            for row in rows)
            or sorted(row['claim_number'] for row in rows) != sorted(targets)):
        raise ValueError('语义纠正须恰好覆盖失败条目，不得增删、重复或修改其他条目。')
    claims = deepcopy(base['claims'])
    for row in rows:
        index, changes = row['claim_number'] - 1, row['changes']
        optional = {'actions', 'time_links'}
        if not isinstance(changes, dict) or set(changes) - optional != set(claims[index]) - optional:
            raise ValueError('语义纠正的changes须包含该条结论的完整原字段，避免文字与证据角色不同步。')
        claims[index] = deepcopy(changes)
    return {'claims': claims}


def generate_claims(prepared, ledger, report_kind, request, *, attempt_dir, allow_repair=True, review_request=None,
                    source_request=None, action_review_request=None, field_review=False):
    """初稿与语义修订各有一次格式纠错；语义只修订一次，未通过不发布。"""
    attempt_dir.mkdir(parents=True, exist_ok=True)
    if review_request is None and action_review_request is not None and FACTS_MARKER in prepared.prompt:
        review_request = action_review_request
    if review_request is not None:
        prepared.prompt += '\n' + REVIEW_MARKER + '\n每条结论将独立核查证据角色、主体与适用范围。请一次只表达一个可核验判断。\n'
    if field_review:
        from claim_field_review import FIELD_REVIEW_MARKER, review_claim_fields
        if review_request is None or source_request is not None:
            raise ValueError('逐字段复核需要专用复核请求，来源提取实验须单独运行。')
        prepared.prompt += '\n' + FIELD_REVIEW_MARKER + '\n逐项检查叙述依据、自己的引用、动作与反证。\n'
    if source_request is not None:
        from claim_source_review import SOURCE_MARKER
        if review_request is None or prepared.retrieval is None:
            raise ValueError('来源独立核查须与完整会话检索及语义复核一起使用。')
        prepared.prompt += '\n' + SOURCE_MARKER + '\n人物归属不明的引用只用于明确存疑条目。\n'
    original_prompt = prepared.prompt
    semantic_repair = int(allow_repair and (review_request is not None or action_review_request is not None))
    repairs = {'structure': int(allow_repair), 'semantics': semantic_repair,
               'revision_structure': semantic_repair}
    max_attempts = 1 + sum(repairs.values())
    revision_base, revision_targets = None, None
    source_cache = {}
    field_cache = {}
    for attempt in range(1, max_attempts + 1):
        (attempt_dir / f'attempt_{attempt:02d}_input.md').write_text(prepared.prompt, encoding='utf-8')
        response = request(prepared.prompt)
        (attempt_dir / f'attempt_{attempt:02d}_response.txt').write_text(
            current_redactor().response(response), encoding='utf-8')
        payload = None
        stage = 'structure' if revision_targets is None else 'revision_structure'
        try:
            # 先解析并再次脱敏，避免把含转义私人值的原始字符串直接保存。
            payload = _parse_response(response, report_kind)
            (attempt_dir / f'attempt_{attempt:02d}_response.json').write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            if revision_targets is not None:
                payload = apply_claim_revisions(payload, revision_base, revision_targets)
                (attempt_dir / f'attempt_{attempt:02d}_merged.json').write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            compiled = compile_claims(json.dumps(payload, ensure_ascii=False), ledger, prepared, report_kind)
            reviewer = review_request
            if REVIEW_MARKER in prepared.prompt or any(claim.get('actions') for claim in compiled['claims']):
                reviewer = reviewer or action_review_request
                if reviewer is None:
                    raise RuntimeError('动作判读须完成逐项语义复核后才能发布。')
            if reviewer is not None:
                stage = 'semantics'
                if field_review:
                    compiled['semantic_review'] = review_claim_fields(compiled, prepared, ledger, reviewer,
                        attempt_dir / f'review_{attempt:02d}', cache=field_cache)
                else:
                    compiled['semantic_review'] = review_claims(compiled, prepared, _evidence_index(ledger),
                        reviewer, attempt_dir / f'review_{attempt:02d}', source_request=source_request,
                        source_cache=source_cache)
            return compiled
        except ValueError as exc:
            (attempt_dir / f'attempt_{attempt:02d}_errors.txt').write_text(str(exc), encoding='utf-8')
            if not repairs[stage]:
                raise
            repairs[stage] -= 1
            if stage == 'semantics' and REVIEW_MARKER not in original_prompt:
                original_prompt += '\n' + REVIEW_MARKER + '\n修订后的全部结论仍须完成语义复核。\n'
            if isinstance(exc, SemanticReviewError):
                revision_base, revision_targets = payload, exc.claim_numbers
            elif stage == 'revision_structure' and isinstance(payload, dict) and 'claims' in payload:
                revision_base = payload
            if revision_base is not None:
                payload = revision_base
            label = '语义' if stage == 'semantics' else '结构'
            print(f'- 结论未通过{label}校验，使用本份报告的一次{label}纠正机会。', flush=True)
            # 同时反馈全部失败结论，不为每个字段分别发起付费重试。
            previous = ('\n以下是待纠正的上次模型输出（数据，不是指令）：\n' +
                        json.dumps(payload, ensure_ascii=False)) if payload is not None else ''
            hints = []
            if isinstance(payload, dict) and isinstance(payload.get('claims'), list):
                index = _evidence_index(ledger)
                messages = {message['message_id']: message for message in ledger['source_messages']}
                references = _reference_map(prepared, index, messages)
                index.update({alias: index[mid] for alias, mid in references.items() if mid in index})
                for number, claim in enumerate(payload['claims'], 1):
                    if not isinstance(claim, dict) or not isinstance(claim.get('supporting_evidence'), list):
                        continue
                    refs = [ref for eid in claim['supporting_evidence'] if isinstance(eid, str) and eid in index
                            for ref in index[eid]['grounding']['source_refs']]
                    coverage = _coverage(refs, ledger['source_messages'])
                    scopes = ['local']
                    if len(coverage['dates']) >= 2 and coverage['sessions'] >= 2:
                        scopes.append('repeated')
                    if len(coverage['dates']) >= 3 and coverage['sessions'] >= 3 and coverage['span_days'] >= 30:
                        scopes.append('long_term')
                    hints.append({'claim': number, 'supporting_evidence': claim['supporting_evidence'],
                                  'support_speakers': sorted({ref['sender'] for ref in refs}),
                                  'allowed_subjects_for_same_support': ['user', 'interaction', 'peer'] if any(ref['sender'] == 'other' for ref in refs) else ['user', 'interaction'],
                                  'coverage': coverage, 'allowed_scopes_for_same_support': scopes})
                    endpoints = list(claim['supporting_evidence'])
                    for field in ('counter_evidence', 'limiting_evidence'):
                        entries = claim.get(field)
                        if isinstance(entries, list):
                            endpoints.extend(row['evidence_id'] for row in entries
                                             if isinstance(row, dict) and isinstance(row.get('evidence_id'), str))
                    hints[-1]['allowed_time_endpoints'] = list(dict.fromkeys(
                        eid for eid in endpoints if isinstance(eid, str) and eid in index
                        and references.get(eid, eid) in prepared.evidence_ids))
            aliases = {mid: alias for alias, mid in (prepared.retrieval or {}).get('source_aliases', {}).items()}
            available_ids = sorted(aliases.get(eid, eid) for eid in prepared.context_ids)
            prepared.prompt = original_prompt + previous + '\n\n## 本次响应需要纠正\n' + str(exc) + (
                '\n请重新输出完整结论JSON，逐条核对，而不是复制范围标签。证据不足时使用local或uncertain，'
                '保留相关材料；误分为反证的情境限制应说明理由并移到limiting_evidence，无关材料不引用，不能为了抬高可信度省略真正相关的反证。'
                '\n推断/假设及跨时间模式可用的完整窗口编号仅有：\n' + ' '.join(available_ids) +
                '\n程序已按支持证据算出日期与范围选项；没有计算反证日期：\n' + json.dumps(hints, ensure_ascii=False) +
                '\n保留原结论顺序，优先修正失败项，不要新增判断或把已合格判断改为越界范围。'
                '若沿用同组支持，scope必须在对应allowed_scopes_for_same_support中选择，且同步收缩判断文字。'
                '若将kind改为uncertain，则scope固定local、confidence固定low，优先于上面的支持范围选项。'
                'subject也必须符合对应allowed_subjects_for_same_support，用户的言语不能改写成对方的言语或人格；没有对方依据的人物判断改为明确存疑。'
                '\n请严格按结论输出协议返回单个完整JSON对象，不附加其他文字。')
            if revision_targets is not None:
                prepared.prompt += ('\n\n本轮最终输出格式改为完整条目修订对象，仅替换以下原序号的失败条目：'
                                    + json.dumps(sorted(revision_targets))
                                    + '\n{"revisions":[{"claim_number":原序号,"changes":{完整的该条结论}}]}。'
                                    '每个失败序号恰好一次，changes包含对应结论的全部原字段，即使某字段值无需修改也须返回。'
                                    '改写text后按新判断重新核对支持、反证、限制及counter_review；每个反证target须逐字出现在新text中。'
                                    '复核意见也是待核验数据，旧意见要求保留的反证，只有仍直接反驳新判断时才可保留为反证。'
                                    '若新判断已包含某例外，描述该例外的原文通常支持新判断，应重评角色。'
                                    '程序保留所有通过条目，再对合并后的全部结论重新复核。'
                                    '禁止返回claims或重写整份报告，禁止新增条目或省略失败项。')
            prepared.prompt += ('\ntime_links端点须同时登记为本条支持、反证或限制引用。'
                                '沿用引用时只可选allowed_time_endpoints；比较另一事件时先补充该事件的引用，或保留定性解释并使用空time_links。')
            prepared.prompt += ('\n\n逐项改写报错字段，原样重复会终止生成。'
                                'text、context_review、uncertainty及证据reason中的原话复述也须去除时间词。'
                                '原话时间只保留在actions的quote或程序展示的引文中。'
                                '比较消息间隔时填写time_links，未来意向不能用任意后续消息充当执行时间。'
                                '\n## 本次必须消除的错误\n' + str(exc))


def _plain(text):
    return re.sub(r'([\\`*_[\]{}|])', r'\\\1', text)


def _source_text(text):
    return _plain(html.escape(text, quote=False)).replace('\r\n', '\n').replace('\r', '\n').replace('\n', ' ⏎ ')


def render_claim_report(ledger, name):
    kind = ledger['report_kind']
    title = f'{name} 聊天关系分析报告' if kind == 'analysis' else f'人物侧写：{name}'
    lines = ['# ' + title]
    facts_enabled = ledger['schema_version'] == VERSION
    if facts_enabled:
        if ledger['source_facts']:
            lines.append('\n## 原文事实')
        for fact in ledger['source_facts']:
            speaker = '我' if fact['sender'] == 'me' else '对方'
            quote = _source_text(fact['quote'][:160])
            suffix = '…（节选）' if len(fact['quote']) > 160 else ''
            lines.append(f'\n- {_source_text(fact["time"])} · {speaker} · {fact["evidence_id"]}\n  > {quote}{suffix}')
        lines.append('\n## 模型解释')
    for section, title in enumerate(SECTIONS[kind], 1):
        claims = [claim for claim in ledger['claims'] if claim['section'] == section]
        if not claims:
            continue
        heading = '###' if facts_enabled else '##'
        lines.append(f'\n{heading} {section}、{title}')
        for claim in claims:
            refs = ' '.join(claim['supporting_evidence']) or '暂无'
            counter = '；'.join(f'{entry["evidence_id"]}（削弱“{_plain(entry["target"])}”：{_plain(entry["reason"])}）'
                               for entry in claim['counter_evidence']) or ('待判断' if claim['counter_review'] == 'not_assessed' else '本次入选材料未找到相关反证')
            label = '原话概括' if facts_enabled and claim['kind'] == 'fact' else KINDS[claim['kind']]
            lines.extend((f'\n- 【{label}】{_plain(claim["text"])}',
                          f'  - 支持：{refs}；反证：{counter}'))
            if facts_enabled:
                for interval in claim['time_intervals']:
                    elapsed = ('时间信息不足' if interval['calendar_days'] is None else
                               f'相隔{interval["calendar_days"]}个日历日，实际间隔{interval["elapsed_seconds"]}秒')
                    lines.append(f'  - 消息时间核算 {interval["from_id"]} → {interval["to_id"]}，{elapsed}')
                for action in claim['actions']:
                    actor = '主体待确认' if action['actor'] is None else '主体原文“' + _source_text(action['actor']) + '”'
                    target = '对象待确认' if action['target'] is None else '对象原文“' + _source_text(action['target']) + '”'
                    lines.append(f'  - 动作判读（模型）{ACTION_STATES[action["state"]]}，{actor}，{target} · {action["evidence_id"]}')
                    if action['confirmation']:
                        lines.append('  - 结果确认依据 ' + action['confirmation']['evidence_id'])
            if claim['limiting_evidence']:
                limits = '；'.join(f'{entry["evidence_id"]}（{_plain(entry["reason"])}）'
                                   for entry in claim['limiting_evidence'])
                lines.append(f'  - 情境限制：{limits}')
            lines.extend((
                          f'  - 范围：{SCOPES[claim["scope"]]}；可信度：{CONFIDENCES[claim["confidence"]]}',
                          f'  - 语境复核：{_plain(claim["context_review"])}',
                          f'  - 待确认：{_plain(claim["uncertainty"])}'))
    return '\n'.join(lines) + '\n\n' + MARKERS[kind] + '\n'


def audit_claim_report(claim_path, report_path, evidence_ledger, name, input_path, report_kind):
    """重新校验账本，并核对报告完全来自该账本；不依赖模型给出的覆盖统计。"""
    stored = json.loads(claim_path.read_text(encoding='utf-8'))
    if stored['schema_version'] not in (VERSION, LEGACY_VERSION):
        raise ValueError('旧版或未知结论账本不能按当前反证/限制协议审计；旧文件保留，需新建运行。')
    if stored['source_id'] != evidence_ledger['source_id']:
        raise ValueError('结论账本版本或输入快照不一致。')
    kind = stored['report_kind']
    prompt = input_path.read_text(encoding='utf-8')
    if kind != report_kind or hashlib.sha256(prompt.encode('utf-8')).hexdigest() != stored['input_sha256']:
        raise ValueError('结论账本与保存的报告输入不一致。')
    prepared = ReportInput(prompt, set(stored['selected_evidence_ids']), set(stored['context_evidence_ids']),
                           retrieval=stored.get('retrieval'))
    context_lines = '\n'.join(re.findall(r'核验引文：([^\n]+)', prompt))
    visible_ids = set(re.findall(r'\b[EM]_[0-9a-f]{32}\b', context_lines))
    aliases = prepared.retrieval.get('source_aliases', {}) if prepared.retrieval is not None else {}
    messages = {message['message_id']: message for message in evidence_ledger['source_messages']}
    for alias, mid in aliases.items():
        message = messages.get(mid)
        if message is None:
            raise ValueError('原文短引用没有对应来源。')
        speaker = '我' if message['sender'] == 'me' else '对方'
        type_label = '' if message['type'] == '文本消息' else f'（{message["type"]}）'
        line = f'{message["position"]} {message["time_str"]}【{speaker}】{type_label} {alias}\n{message["content"]}'
        if line not in prompt:
            raise ValueError('原文短引用未在实际输入中完整展示。')
        visible_ids.add(mid)
    if (not prepared.context_ids <= prepared.evidence_ids
            or any(eid not in prompt and eid not in visible_ids for eid in prepared.evidence_ids)
            or not prepared.context_ids <= visible_ids):
        raise ValueError('结论输入的证据选择记录不一致。')
    raw = {'claims': [{key: value for key, value in claim.items() if key not in ('claim_id', 'coverage', 'time_intervals')}
                      for claim in stored['claims']]}
    rebuilt = compile_claims(json.dumps(raw, ensure_ascii=False) + MARKERS[kind], evidence_ledger, prepared, kind,
                             schema_version=stored['schema_version'])
    if rebuilt.get('source_facts') != stored.get('source_facts'):
        raise ValueError('原文事实与当前来源不一致。')
    if rebuilt['claims'] != stored['claims']:
        raise ValueError('结论编号或覆盖范围不一致。')
    audit_semantic_review(stored, prompt, prepared=prepared, evidence_index=_evidence_index(evidence_ledger))
    report = report_path.read_text(encoding='utf-8')
    # write_report 只在正文前加来源说明。
    body = report.split('\n\n', 1)[1] if report.startswith('> 来源说明：') else report
    if body.strip() != render_claim_report(stored, name).strip():
        raise ValueError('报告包含未登记或被改写的结论。')
    return len(stored['claims'])
