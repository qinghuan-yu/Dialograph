"""结论生成、有限重试、报告渲染与审计；兼容原有导入入口。"""
from __future__ import annotations

import hashlib
import html
import json
import re

from privacy_redaction import current_redactor
from claim_semantic_review import REVIEW_MARKER, review_claims, audit_semantic_review, SemanticReviewError
from claim_facts import FACTS_MARKER, ACTION_STATES
from claim_compiler import (VERSION, LEGACY_VERSION, SECTIONS, MARKERS, KINDS, SCOPES, CONFIDENCES,
    ReportInput, claim_contract, context_catalog, compile_claims, _evidence_index, _coverage,
    _reference_map, _parse_response, _check_claim_text, _reviewed_references, _validate_claim)
from claim_repair import apply_claim_revisions, build_repair_prompt


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
            prepared.prompt = build_repair_prompt(original_prompt, payload, exc,
                prepared=prepared, ledger=ledger, revision_targets=revision_targets)


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
