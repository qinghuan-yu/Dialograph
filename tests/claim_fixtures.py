"""完整流程模型替身使用的最小合法结论响应。"""
import json


def field_review_response(prompt):
    data, _ = json.JSONDecoder().raw_decode(prompt.split('\n本条草稿及登记引用\n', 1)[1])
    return json.dumps({
        'field_checks': [dict(field=field, content_supported=True, citations_complete=True,
                              certainty_preserved=True, outcome_context_complete=True,
                              reason='合成流程替身确认该字段的有限判断。', basis=[]) for field in data['statement_fields']],
        'action_checks': [dict(action_index=number, actor_supported=True, target_supported=True,
                               state_supported=True, reason='合成动作逐项一致。')
                          for number in range(1, len(data['actions']) + 1)],
        'counter_checks': [dict(evidence_id=row['evidence_id'], directly_contradicts=True,
                                reason='合成反证直接相反。') for row in data['counter_evidence']],
    })


def with_reviews(responses):
    """流程测试补齐独立复核响应，原有生成响应序列保持各自的测试含义。"""
    iterator = None if callable(responses) else iter(responses)

    def request(*args, **kwargs):
        from claim_semantic_review import REVIEW_INSTRUCTIONS
        from claim_field_review import FIELD_REVIEW_INSTRUCTIONS
        system = kwargs.get('system_prompt', args[1] if len(args) > 1 else '')
        if system == FIELD_REVIEW_INSTRUCTIONS:
            prompt = kwargs.get('user_prompt', args[2] if len(args) > 2 else '')
            return field_review_response(prompt)
        if system == REVIEW_INSTRUCTIONS:
            prompt = kwargs.get('user_prompt', args[2] if len(args) > 2 else '')
            drafts, _ = json.JSONDecoder().raw_decode(prompt.split('待复核结论及逐字引用\n', 1)[1])
            assert all(not row.get('actions') and not row['counter_evidence'] for row in drafts)
            return json.dumps({'reviews': [{'claim_id': row['review_id'], 'verdict': 'pass',
                                          'reason': '合成流程替身确认此条有限判断。'} for row in drafts]})
        value = responses(*args, **kwargs) if iterator is None else next(iterator)
        if isinstance(value, Exception):
            raise value
        return value

    return request


def claim_response(text='当前材料不足以判断。', report_kind='analysis', **changes):
    claim = dict(section=1, text=text, kind='uncertain', subject='interaction',
                 scope='local', confidence='low', supporting_evidence=[], counter_evidence=[], limiting_evidence=[],
                 uncertainty='需要更多相关场景。', context_review='当前材料不足。',
                 counter_review='not_assessed')
    claim.update(changes)
    marker = 'END_FINAL_REPORT' if report_kind == 'analysis' else 'END_PERSONA_REPORT'
    return json.dumps({'claims': [claim]}, ensure_ascii=False) + f'\n<!-- {marker} -->'


def claim_revision(text='当前材料不足以判断。', claim_number=1, **changes):
    """语义修订整体替换单个失败条目，覆盖全部相关字段。"""
    payload = json.loads(claim_response(text, **changes).split('\n<!--')[0])
    return json.dumps({'revisions': [{'claim_number': claim_number, 'changes': payload['claims'][0]}]}, ensure_ascii=False)
