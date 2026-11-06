"""原文事实与模型解释分离，使用独立合成消息验证。"""
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from analyze import parse_chat_messages
from claim_fixtures import claim_response, claim_revision
from claim_ledger import ReportInput, compile_claims, render_claim_report, audit_claim_report
from evidence_grounding import ground_document
from claim_semantic_review import SemanticReviewError, review_claims, build_review_prompt, validate_reviews
from claim_ledger import _evidence_index, generate_claims, claim_contract, apply_claim_revisions


def fact_source(rows=None):
    rows = rows or [
        ('2026-04-08 09:00:00', False, '我准备把文件发给小林。'),
        ('2026-04-08 09:01:00', False, '我已经发给小林了。'),
        ('2026-04-08 09:02:00', True, '小林告诉我，文件已经收到。'),
        ('2026-06-28 09:00:00', False, '最近状态恢复了吗？'),
    ]
    messages = parse_chat_messages([
        {'type': '文本消息', 'formattedTime': time, 'isSend': sender, 'content': content}
        for time, sender, content in rows])['messages']
    document = {'chunk_index': 1, 'message_range': {'start': 1, 'end': len(messages)},
                'time_range': {'start': rows[0][0], 'end': rows[-1][0]},
                **{key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')}}
    document['events'] = [{'summary': '合成节点', 'evidence': m['content'], 'source_message_ids': [m['message_id']],
                           'confidence': 'low'} for m in messages]
    return {'source_id': messages[0]['source_id'], 'source_messages': messages,
            'documents': [ground_document(document, messages)]}


class ClaimFactTests(unittest.TestCase):
    def setUp(self):
        self.data = fact_source()
        self.ids = [m['message_id'] for m in self.data['source_messages']]
        self.prepared = ReportInput('核验引文：' + ' '.join(self.ids), set(self.ids), set(self.ids))

    def compile(self, **changes):
        options = dict(text='对方表达过处理意向与关心。', kind='inference', supporting_evidence=[self.ids[0], self.ids[-1]],
                       counter_review='not_found_in_selected', context_review='依据原话理解这两次表达。')
        options.update(changes)
        return compile_claims(claim_response(**options),
                              self.data, self.prepared, 'analysis')

    def test_facts_copy_source_quote_speaker_and_time_without_model_paraphrase(self):
        compiled = self.compile()
        self.assertEqual(compiled['schema_version'], 'chat-claims-v3')
        facts = compiled['source_facts']
        self.assertEqual([row['evidence_id'] for row in facts], [self.ids[0], self.ids[3]])
        self.assertEqual(facts[0]['quote'], '我准备把文件发给小林。')
        self.assertEqual(facts[0]['sender'], 'other')
        self.assertEqual(facts[0]['time'], '2026-04-08 09:00:00')
        report = render_claim_report(compiled, '合成')
        self.assertIn('原文事实', report)
        self.assertIn('模型解释', report)
        self.assertIn('2026-04-08 09:00:00', report)

    def test_quoted_prose_requires_this_claims_own_citation(self):
        self.prepared.prompt += '\n## 逐字引语校验（v1）'
        for field in ('text', 'context_review', 'uncertainty'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field + '.*引语'):
                self.compile(**{field: '原话是“我已经发给小林了”。'})
        self.compile(context_review='原话是“我准备把文件发给小林”。')
        self.compile(context_review='原话是「我准备把文件发给小林」。')
        self.compile(context_review='原话是“我已经发给小林了”。', supporting_evidence=[self.ids[1]])

    def test_quoted_prose_checks_counter_reasons_and_preserves_historical_inputs(self):
        self.compile(context_review='原话是“我已经发给小林了”。')
        self.prepared.prompt += '\n## 逐字引语校验（v1）'
        with self.assertRaisesRegex(ValueError, 'limiting_evidence.*引语'):
            self.compile(limiting_evidence=[{'evidence_id': self.ids[2],
                                            'reason': '“已删除文件”尚无确认。'}])

    def test_new_report_without_actions_still_uses_configured_reviewer(self):
        self.prepared.prompt += claim_contract('analysis')
        reviewer = Mock(return_value=json.dumps({'reviews':[
            {'claim_id':'R_1', 'verdict':'pass', 'reason':'仅说明材料无法判断。'}]}))
        with tempfile.TemporaryDirectory() as directory:
            compiled = generate_claims(self.prepared, self.data, 'analysis', Mock(return_value=claim_response()),
                                       attempt_dir=Path(directory), action_review_request=reviewer)
        reviewer.assert_called_once()
        self.assertIn('semantic_review', compiled)
        self.assertIn('## 独立语义复核门槛', self.prepared.prompt)

    def test_semantic_revision_has_one_format_repair_after_initial_format_repair(self):
        request = Mock(side_effect=['invalid json', claim_response(),
            claim_revision(text='数日后表达支持。'), claim_revision(text='后续表达支持仍待核验。')])
        reviewer = Mock(side_effect=[json.dumps({'reviews':[
            {'claim_id':'R_1','verdict':'revise','reason':'需重新描述可判断范围。'}]}),
            json.dumps({'reviews':[{'claim_id':'R_1','verdict':'pass','reason':'已明确范围。'}]})])
        with tempfile.TemporaryDirectory() as directory:
            compiled = generate_claims(self.prepared,self.data,'analysis',request,
                                       attempt_dir=Path(directory),review_request=reviewer)
        self.assertEqual(request.call_count,4)
        self.assertEqual(reviewer.call_count,2)
        self.assertEqual(compiled['claims'][0]['text'],'后续表达支持仍待核验。')
        self.assertIn('数日后表达支持。',request.call_args.args[0])
        self.assertIn('allowed_time_endpoints',request.call_args.args[0])

    def test_repeated_invalid_semantic_replacement_stops_at_its_format_limit(self):
        request = Mock(side_effect=[claim_response(), claim_revision(text='数日后回应。'),
                                    claim_revision(text='次日回应。')])
        reviewer = Mock(return_value=json.dumps({'reviews':[
            {'claim_id':'R_1','verdict':'revise','reason':'需重新描述。'}]}))
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError,'次日'):
            generate_claims(self.prepared,self.data,'analysis',request,
                            attempt_dir=Path(directory),review_request=reviewer)
        self.assertEqual(request.call_count,3)
        self.assertEqual(reviewer.call_count,1)

    def test_revision_format_hints_handle_malformed_reference_arrays(self):
        request = Mock(side_effect=[claim_response(), claim_revision(limiting_evidence=None), claim_revision()])
        reviewer = Mock(side_effect=[json.dumps({'reviews':[
            {'claim_id':'R_1','verdict':'revise','reason':'需明确局限。'}]}),
            json.dumps({'reviews':[{'claim_id':'R_1','verdict':'pass','reason':'已明确局限。'}]})])
        with tempfile.TemporaryDirectory() as directory:
            result = generate_claims(self.prepared,self.data,'analysis',request,
                                     attempt_dir=Path(directory),review_request=reviewer)
        self.assertEqual(result['claims'][0]['limiting_evidence'],[])
        self.assertEqual(request.call_count,3)

    def test_interval_is_computed_from_named_messages(self):
        compiled = self.compile(time_links=[{'from_id': self.ids[0], 'to_id': self.ids[3]}])
        interval = compiled['claims'][0]['time_intervals'][0]
        self.assertEqual(interval['calendar_days'], 81)
        self.assertEqual(interval['elapsed_seconds'], 81 * 86400)
        self.assertIn('相隔81个日历日', render_claim_report(compiled, '合成'))

    def test_model_cannot_supply_an_interval_or_reverse_endpoints(self):
        for link in ({'from_id': self.ids[0], 'to_id': self.ids[3], 'days': 3},
                     {'from_id': self.ids[3], 'to_id': self.ids[0]}):
            with self.subTest(link=link), self.assertRaisesRegex(ValueError, '时间'):
                self.compile(time_links=[link])

    def test_date_count_is_allowed_and_time_errors_identify_the_field(self):
        self.compile(context_review='引文覆盖两个日期，仍仅概括相关表达。')
        self.compile(context_review='多个不同日期的表达，含义需要结合场景。')
        with self.assertRaisesRegex(ValueError, 'uncertainty.*次日'):
            self.compile(uncertainty='次日情况未知。')

    def test_missing_required_field_error_names_the_field_for_repair(self):
        payload = json.loads(claim_response().split('\n<!--')[0])
        del payload['claims'][0]['limiting_evidence']
        with self.assertRaisesRegex(ValueError, 'limiting_evidence'):
            compile_claims(json.dumps(payload), self.data, self.prepared, 'analysis')

    def test_structure_repair_keeps_exact_errors_at_the_end_of_the_request(self):
        request = Mock(side_effect=[claim_response(uncertainty='次日情况未知。'), claim_response()])
        with tempfile.TemporaryDirectory() as directory:
            generate_claims(self.prepared, self.data, 'analysis', request, attempt_dir=Path(directory))
        prompt = request.call_args.args[0]
        self.assertIn('uncertainty中的“次日”', prompt.split('## 本次必须消除的错误\n')[-1])
        self.assertTrue(prompt.rstrip().endswith('叙述保留定性解释。'))

    def test_unknown_citation_does_not_hide_time_errors_in_other_fields(self):
        with self.assertRaises(ValueError) as caught:
            self.compile(supporting_evidence=['P_999'], text='2026-04-08发生交流。',
                         context_review='当晚提到次日返回。', uncertainty='两天后情况未知。')
        error = str(caught.exception)
        self.assertIn('未知或未入选证据', error)
        for field, term in (('text', '2026-04-08'), ('context_review', '当晚'),
                            ('context_review', '次日'), ('uncertainty', '两天')):
            self.assertIn(f'{field}中的“{term}”', error)

    def test_time_errors_are_collected_once_without_hiding_invalid_structure(self):
        with self.assertRaises(ValueError) as caught:
            self.compile(text='当晚回应，次日返回。', uncertainty='两天后情况未知。')
        self.assertEqual(str(caught.exception).count('text中的“当晚”'), 1)
        self.assertIn('uncertainty中的“两天”', str(caught.exception))
        with self.assertRaisesRegex(ValueError, '反证与情境限制'):
            self.compile(limiting_evidence=None)

    def test_calendar_prose_requires_structured_time_link_even_when_uncertain(self):
        for field in ('text', 'context_review', 'uncertainty'):
            for text in ('对方数日后回问状态。', '对方在2026年4月8日回问。', '对方次日再次回应。',
                         '两条表达相隔约两个半月。', '这类互动持续两个月。', '半年之后再联系。'):
                with self.subTest(field=field, text=text), self.assertRaisesRegex(ValueError, '时间'):
                    self.compile(kind='uncertain', confidence='low', **{field: text})

    def test_unknown_action_participants_remain_explicit(self):
        action = {'evidence_id': self.ids[0], 'quote': '我准备把文件发给小林。',
                  'actor': None, 'target': None, 'state': 'planned', 'confirmation': None}
        compiled = self.compile(actions=[action])
        self.assertEqual(compiled['claims'][0]['actions'][0]['actor'], None)
        report = render_claim_report(compiled, '合成')
        self.assertIn('主体待确认', report)
        self.assertIn('计划或意向', report)

    def test_action_cannot_borrow_neighbors_quote_or_unbound_actor(self):
        action = {'evidence_id': self.ids[0], 'quote': '我已经发给小林了。',
                  'actor': '我', 'target': '小林', 'state': 'reported_done', 'confirmation': None}
        with self.assertRaisesRegex(ValueError, '动作原话'):
            self.compile(actions=[action])
        action.update(quote='我准备把文件发给小林。', actor='室友', state='planned')
        with self.assertRaisesRegex(ValueError, '主体'):
            self.compile(actions=[action])

    def test_result_confirmation_requires_another_speakers_cited_quote(self):
        action = {'evidence_id': self.ids[1], 'quote': '我已经发给小林了。',
                  'actor': '我', 'target': '小林', 'state': 'confirmed_result', 'confirmation': None}
        with self.assertRaisesRegex(ValueError, '确认'):
            self.compile(supporting_evidence=self.ids[:3], actions=[action])
        action['confirmation'] = {'evidence_id': self.ids[0], 'quote': '我准备把文件发给小林。'}
        with self.assertRaisesRegex(ValueError, '确认'):
            self.compile(supporting_evidence=self.ids[:3], actions=[action])
        action['confirmation'] = {'evidence_id': self.ids[2], 'quote': '文件已经收到'}
        compiled = self.compile(supporting_evidence=self.ids[:3], actions=[action])
        self.assertEqual(compiled['claims'][0]['actions'][0]['state'], 'confirmed_result')

    def test_fact_tampering_is_rejected_by_independent_audit(self):
        compiled = self.compile()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input.md').write_text(self.prepared.prompt, encoding='utf-8')
            (root / 'report.md').write_text(render_claim_report(compiled, '合成'), encoding='utf-8')
            changed = copy.deepcopy(compiled)
            changed['source_facts'][0]['sender'] = 'me'
            (root / 'claims.json').write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '原文事实'):
                audit_claim_report(root / 'claims.json', root / 'report.md', self.data, '合成', root / 'input.md', 'analysis')

    def test_reviewer_sees_actions_and_program_time_intervals(self):
        action = {'evidence_id': self.ids[0], 'quote': '我准备把文件发给小林。',
                  'actor': '我', 'target': '小林', 'state': 'planned', 'confirmation': None}
        compiled = self.compile(actions=[action], time_links=[{'from_id': self.ids[0], 'to_id': self.ids[3]}])
        prompt = build_review_prompt(compiled, self.prepared, _evidence_index(self.data))
        self.assertIn('"state": "planned"', prompt)
        self.assertIn('"calendar_days": 81', prompt)
        self.assertIn('"source_facts"', prompt)

    def test_false_action_state_check_overrides_general_pass(self):
        action = {'evidence_id': self.ids[0], 'quote': '我准备把文件发给小林。',
                  'actor': '我', 'target': '小林', 'state': 'reported_done', 'confirmation': None}
        compiled = self.compile(actions=[action])
        response = {'reviews': [{'claim_id': 'R_1', 'verdict': 'pass', 'reason': '需要逐项核验。',
                     'action_checks': [{'action_index': 1, 'actor_supported': True, 'target_supported': True,
                                        'state_supported': False, 'reason': '原话为计划，完成状态无依据。'}]}]}
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SemanticReviewError) as failure:
            review_claims(compiled, self.prepared, _evidence_index(self.data),
                          Mock(return_value=json.dumps(response)), Path(directory))
        self.assertEqual(failure.exception.claim_numbers, {1})
        self.assertIn('完成状态无依据', str(failure.exception))

    def test_new_fact_label_is_rendered_as_model_summary(self):
        report = render_claim_report(self.compile(kind='fact'), '合成')
        self.assertIn('【原话概括】', report)
        self.assertNotIn('【事实】', report)

    def test_generation_cannot_publish_action_interpretations_without_review(self):
        action = {'evidence_id': self.ids[0], 'quote': '我准备把文件发给小林。',
                  'actor': None, 'target': None, 'state': 'planned', 'confirmation': None}
        response = claim_response('存在处理计划。', kind='inference', supporting_evidence=[self.ids[0]],
                                  counter_review='not_found_in_selected', actions=[action])
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, '动作判读.*复核'):
            generate_claims(self.prepared, self.data, 'analysis', Mock(return_value=response),
                            attempt_dir=Path(directory))

    def test_conditional_action_review_rechecks_repair_that_removes_action(self):
        action = {'evidence_id': self.ids[0], 'quote': '我准备把文件发给小林。',
                  'actor': '我', 'target': '小林', 'state': 'reported_done', 'confirmation': None}
        response = claim_response('对方已完成发送。', kind='inference', supporting_evidence=[self.ids[0]],
                                  counter_review='not_found_in_selected', actions=[action])
        correction = claim_revision('结果待确认。', actions=[])
        review = Mock(side_effect=[json.dumps({'reviews': [{'claim_id': 'R_1', 'verdict': 'revise',
                                                            'reason': '原话为意向，完成结果待确认。'}]}),
                                   json.dumps({'reviews': [{'claim_id': 'R_1', 'verdict': 'pass',
                                                            'reason': '修订为结果待确认。'}]})])
        with tempfile.TemporaryDirectory() as directory:
            result = generate_claims(self.prepared, self.data, 'analysis', Mock(side_effect=[response, correction]),
                                     attempt_dir=Path(directory), action_review_request=review)
            self.assertEqual(review.call_count, 2)
            self.assertIn('semantic_review', result)
            root = Path(directory)
            (root / 'input.md').write_text(self.prepared.prompt, encoding='utf-8')
            (root / 'report.md').write_text(render_claim_report(result, '合成'), encoding='utf-8')
            (root / 'claims.json').write_text(json.dumps(result), encoding='utf-8')
            self.assertEqual(audit_claim_report(root / 'claims.json', root / 'report.md', self.data,
                                               '合成', root / 'input.md', 'analysis'), 1)
            del result['semantic_review']
            (root / 'claims.json').write_text(json.dumps(result), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '语义复核'):
                audit_claim_report(root / 'claims.json', root / 'report.md', self.data,
                                   '合成', root / 'input.md', 'analysis')

    def test_revision_can_add_optional_fact_fields_to_a_previous_omission(self):
        base = json.loads(claim_response().split('\n<!--')[0])
        changes = json.loads(claim_revision(actions=[], time_links=[]))
        result = apply_claim_revisions(changes, base, {1})
        self.assertEqual(result['claims'][0]['actions'], [])
        self.assertNotIn('actions', base['claims'][0])

    def test_modern_input_cannot_be_audited_as_legacy(self):
        self.prepared.prompt += claim_contract('analysis')
        with self.assertRaisesRegex(ValueError, '版本'):
            compile_claims(claim_response(), self.data, self.prepared, 'analysis', schema_version='chat-claims-v2')

    def test_review_requires_each_action_check_exactly_once(self):
        claim = self.compile()['claims'][0]
        claim['actions'] = [{}]
        check = {'action_index': 1, 'actor_supported': True, 'target_supported': True,
                 'state_supported': True, 'reason': '原话为计划。'}
        review = {'claim_id': claim['claim_id'], 'verdict': 'pass', 'reason': '逐项核对。'}
        validate_reviews([claim], [{**review, 'action_checks': [check]}])
        for checks in ([], [check, check], [{**check, 'action_index': 2}], [{**check, 'state_supported': 'true'}]):
            with self.subTest(checks=checks), self.assertRaisesRegex(RuntimeError, '动作'):
                validate_reviews([claim], [{**review, 'action_checks': checks}])

    def test_calendar_days_differ_from_elapsed_days_and_missing_time_stays_unknown(self):
        for last_time, days, seconds in [('2026-04-09 00:01:00', 1, 120), ('未知', None, None)]:
            with self.subTest(last_time=last_time):
                self.data = fact_source([('2026-04-08 23:59:00', False, '我准备发送。'),
                                         (last_time, True, '我已经收到。')])
                self.ids = [m['message_id'] for m in self.data['source_messages']]
                self.prepared = ReportInput('核验引文：' + ' '.join(self.ids), set(self.ids), set(self.ids))
                result = self.compile(supporting_evidence=self.ids,
                                      time_links=[{'from_id': self.ids[0], 'to_id': self.ids[1]}])
                interval = result['claims'][0]['time_intervals'][0]
                self.assertEqual((interval['calendar_days'], interval['elapsed_seconds']), (days, seconds))

    def test_duplicate_physical_message_cannot_form_an_interval(self):
        event_id = self.data['documents'][0]['events'][0]['evidence_id']
        self.prepared.evidence_ids.add(event_id)
        self.prepared.context_ids.add(event_id)
        with self.assertRaisesRegex(ValueError, '时间'):
            self.compile(supporting_evidence=[event_id, self.ids[0]],
                         time_links=[{'from_id': event_id, 'to_id': self.ids[0]}])

    def test_short_aliases_normalize_in_actions_confirmation_and_time_links(self):
        from claim_facts import normalize_fact_fields
        claim = {'actions': [{'evidence_id': 'P_2', 'confirmation': {'evidence_id': 'P_3'}}],
                 'time_links': [{'from_id': 'P_1', 'to_id': 'P_4'}]}
        aliases = {f'P_{n}': mid for n, mid in enumerate(self.ids, 1)}
        normalize_fact_fields(claim, aliases)
        self.assertEqual(claim['actions'][0]['evidence_id'], self.ids[1])
        self.assertEqual(claim['actions'][0]['confirmation']['evidence_id'], self.ids[2])
        self.assertEqual(claim['time_links'][0], {'from_id': self.ids[0], 'to_id': self.ids[3]})

    def test_normal_workflow_reviews_actions_in_both_reports_without_retrieval(self):
        import analysis_artifacts
        import analysis_service
        import llm_client
        from artifact_audit import audit_artifacts
        from claim_semantic_review import REVIEW_INSTRUCTIONS
        stages = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / 'input.json'
            messages = [{'formattedTime': row['time_str'], 'content': row['content'],
                         'isSend': int(row['sender'] == 'me'), 'type': row['type']}
                        for row in self.data['source_messages']]
            input_path.write_text(json.dumps({'messages': messages}), encoding='utf-8')
            artifacts = analysis_artifacts.build_artifacts(input_path, root / 'out')
            def response(**kwargs):
                prompt = kwargs['user_prompt']
                if not stages:
                    stages.append('chunk')
                    return json.dumps({'events': [{'summary': '表达处理计划', 'evidence': messages[0]['content'],
                                                   'source_message_ids': ['m1'], 'confidence': 'low'}],
                                       **{key: [] for key in ('relation_signals', 'persona_signals',
                                                             'counter_evidence', 'uncertainties')}})
                if kwargs['system_prompt'] == REVIEW_INSTRUCTIONS:
                    stages.append('review')
                    return json.dumps({'reviews': [{'claim_id': 'R_1', 'verdict': 'pass', 'reason': '原话为计划。',
                        'action_checks': [{'action_index': 1, 'actor_supported': True, 'target_supported': True,
                                           'state_supported': True, 'reason': '只表示计划，结果未知。'}]}]})
                kind = 'persona' if '## 本次具体任务：对方人物侧写' in prompt else 'analysis'
                stages.append(kind)
                eid = re.search(r'核验引文：([EM]_[0-9a-f]{32})', prompt).group(1)
                action = {'evidence_id': eid, 'quote': messages[0]['content'], 'actor': '我', 'target': '小林',
                          'state': 'planned', 'confirmation': None}
                return claim_response('对方表达过发送文件的计划。', report_kind=kind, kind='fact',
                                      supporting_evidence=[eid], counter_review='not_found_in_selected', actions=[action])
            config = SimpleNamespace(api_base='https://test.invalid', api_key='test', model='test')
            with patch.object(llm_client, 'call_llm', side_effect=response):
                analysis_service.analyze_conversation(artifacts, 'skill', 'skill', config, 1,
                                                      chunk_sleep=0)
            self.assertEqual(stages, ['chunk', 'analysis', 'review', 'persona', 'review'])
            self.assertTrue(audit_artifacts(artifacts, require_persona=True, require_claims=True).ok)
            for kind in ('analysis', 'persona'):
                claims = json.loads(artifacts[kind + '_claims_file'].read_text(encoding='utf-8'))
                self.assertEqual(claims['schema_version'], 'chat-claims-v3')
                self.assertEqual(len(claims['source_facts']), 1)
                self.assertIn('semantic_review', claims)

    def test_source_and_action_text_cannot_inject_markdown_lines(self):
        self.data = fact_source([('<script>bad</script>', False, '我\n# 主体\n准备发给小林。')])
        self.ids = [self.data['source_messages'][0]['message_id']]
        self.prepared = ReportInput('核验引文：' + self.ids[0], set(self.ids), set(self.ids))
        result = self.compile(supporting_evidence=self.ids, actions=[{
            'evidence_id': self.ids[0], 'quote': '我\n# 主体\n准备发给小林。',
            'actor': '我\n# 主体', 'target': '小林', 'state': 'planned', 'confirmation': None}])
        report = render_claim_report(result, '合成')
        self.assertNotIn('<script>', report)
        self.assertNotIn('\n# 主体', report)
        self.assertIn('⏎', report)

    def test_historical_v2_can_still_be_reaudited_with_its_original_fields(self):
        response = claim_response('原文P_1表达计划。', kind='inference', supporting_evidence=[self.ids[0]],
                                  counter_review='not_found_in_selected')
        compiled = compile_claims(response, self.data, self.prepared, 'analysis', schema_version='chat-claims-v2')
        self.assertNotIn('source_facts', compiled)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input.md').write_text(self.prepared.prompt, encoding='utf-8')
            (root / 'report.md').write_text(render_claim_report(compiled, '合成'), encoding='utf-8')
            (root / 'claims.json').write_text(json.dumps(compiled), encoding='utf-8')
            self.assertEqual(audit_claim_report(root / 'claims.json', root / 'report.md', self.data,
                                               '合成', root / 'input.md', 'analysis'), 1)


if __name__ == '__main__':
    unittest.main()
