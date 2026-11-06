"""结论的证据范围、原文语境和报告发布约束；仅使用合成输入。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from test_report_context_budget import ledger
import report_generation
from claim_fixtures import claim_response, with_reviews


class ClaimLedgerTests(unittest.TestCase):
    def setUp(self):
        self.data = ledger()
        self.eid = self.data['documents'][0]['events'][0]['evidence_id']
        self.other = self.data['documents'][0]['counter_evidence'][0]['evidence_id']
        summary = '## 结构化证据摘要\n### 事件\n- ' + self.eid
        self.prepared = report_generation.prepare_final_input('合成', '统计', '覆盖', summary, [], ledger=self.data)

    def build(self, **changes):
        from claim_ledger import compile_claims
        options = dict(kind='inference', supporting_evidence=[self.eid], counter_review='not_found_in_selected',
                       context_review='结合连续回应，仅描述本次协作。')
        options.update(changes)
        return compile_claims(claim_response(**options), self.data, self.prepared, 'analysis')

    def test_valid_claim_has_computed_source_range_and_report_is_derived(self):
        from claim_ledger import render_claim_report
        result = self.build(text='对方在本次协作中给予回应。')
        item = result['claims'][0]
        self.assertTrue(item['claim_id'].startswith('C_'))
        self.assertEqual(item['coverage']['positions'], [23])
        self.assertEqual(result['source_id'], self.data['source_id'])
        report = render_claim_report(result, '合成')
        self.assertIn(item['text'], report)
        self.assertIn(self.eid, report)
        self.assertIn('END_FINAL_REPORT', report)
        self.assertEqual(self.build(text=item['text']), result)

    def test_complete_json_needs_no_extra_marker_or_paid_retry(self):
        from claim_ledger import compile_claims
        raw = claim_response().split('\n<!--')[0]
        result = compile_claims(raw, self.data, self.prepared, 'analysis')
        self.assertEqual(len(result['claims']), 1)

    def test_raw_position_references_cannot_leak_into_narrative_fields(self):
        for field in ('text', 'context_review', 'uncertainty'):
            for value in ('依据P_23可知。', 'P_23至P_24的原话。', '依据' + self.eid + '可知。'):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, '引用放入证据数组'):
                        self.build(**{field: value})
        with self.assertRaisesRegex(ValueError, '引用放入证据数组'):
            self.build(limiting_evidence=[{'evidence_id': self.other, 'reason': 'P_21限制了范围。'}])

    def test_report_contract_requires_evidence_before_user_motive_hypotheses(self):
        from claim_ledger import claim_contract
        contract = claim_contract('analysis', raw_sources=True)
        self.assertIn('条数由独立证据决定', contract)
        self.assertIn('用户自己的明确表达', contract)
        self.assertIn('图片、文件、通话等占位记录', contract)

    def test_full_pipeline_accepts_complete_json_with_generation_and_review(self):
        import analysis_artifacts
        import analysis_service
        import chunking
        import llm_client
        chunk = json.dumps({key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')})
        responses = [chunk, claim_response().split('\n<!--')[0],
                     claim_response(report_kind='persona').split('\n<!--')[0]]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.json'
            source.write_bytes((Path(__file__).parent / 'fixtures/web_smoke.json').read_bytes())
            artifacts = analysis_artifacts.build_artifacts(source, Path(tmp) / 'out')
            config = SimpleNamespace(api_base='https://test.invalid', api_key='test', model='test')
            with patch.object(llm_client, 'call_llm', side_effect=with_reviews(responses)) as api:
                analysis_service.analyze_conversation(artifacts, 'skill', 'skill', config, 1)
            self.assertEqual(api.call_count, 5)
            self.assertTrue(artifacts['analysis_claims_file'].exists())
            self.assertTrue(artifacts['persona_claims_file'].exists())

    def test_bad_json_is_saved_and_corrected_once_without_token_escalation(self):
        from claim_ledger import generate_claims
        requests = []
        responses = iter(('{"claims": [', claim_response().split('\n<!--')[0]))
        def request(prompt):
            requests.append(prompt)
            return next(responses)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = generate_claims(self.prepared, self.data, 'analysis', request, attempt_dir=folder)
            self.assertEqual(len(result['claims']), 1)
            self.assertEqual(len(requests), 2)
            self.assertIn('单个 JSON', requests[1])
            self.assertEqual((folder / 'attempt_01_response.txt').read_text(encoding='utf-8'), '{"claims": [')

    def test_invalid_model_scope_is_repaired_once_with_all_errors_and_actual_input_saved(self):
        from claim_ledger import generate_claims
        invalid = json.loads(claim_response().split('\n<!--')[0])
        invalid['claims'] *= 2
        invalid['claims'][0] = {**invalid['claims'][0], 'kind': 'inference', 'scope': 'long_term',
            'supporting_evidence': [self.eid], 'counter_review': 'not_found_in_selected'}
        invalid['claims'][1] = {**invalid['claims'][1], 'kind': 'inference',
            'supporting_evidence': [self.other], 'counter_review': 'not_found_in_selected'}
        requests = []
        responses = [json.dumps(invalid) + '\n<!-- END_FINAL_REPORT -->', claim_response()]
        def request(prompt):
            requests.append(prompt)
            return responses.pop(0)
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_claims(self.prepared, self.data, 'analysis', request, attempt_dir=Path(tmp))
            self.assertEqual(len(requests), 2)
            self.assertIn('第 1 条结论', requests[1])
            self.assertIn('第 2 条结论', requests[1])
            self.assertEqual(self.prepared.prompt, requests[-1])
            self.assertEqual((Path(tmp) / 'attempt_02_input.md').read_text(encoding='utf-8'), requests[-1])
            from hashlib import sha256
            self.assertEqual(result['input_sha256'], sha256(requests[-1].encode('utf-8')).hexdigest())

    def test_invalid_model_claims_stop_after_two_responses(self):
        from claim_ledger import generate_claims
        with tempfile.TemporaryDirectory() as tmp, patch.object(sys.modules[__name__], 'claim_response', wraps=claim_response) as fixture:
            def request(prompt):
                return fixture(kind='inference', supporting_evidence=[])
            with self.assertRaisesRegex(ValueError, '支持'):
                generate_claims(self.prepared, self.data, 'analysis', request, attempt_dir=Path(tmp))
            self.assertEqual(fixture.call_count, 2)

    def test_valid_extra_claims_are_not_rejected_by_arbitrary_count_limit(self):
        from claim_ledger import compile_claims
        base = json.loads(claim_response().split('\n<!--')[0])['claims'][0]
        payload = {'claims': [{**base, 'text': f'待判断的不同问题{n}。'} for n in range(24)]}
        result = compile_claims(json.dumps(payload) + '\n<!-- END_FINAL_REPORT -->', self.data, self.prepared, 'analysis')
        self.assertEqual(len(result['claims']), 24)

    def test_unknown_unseen_and_uncertain_evidence_cannot_become_support(self):
        for eid in ('E_' + 'f' * 32, self.other):
            with self.subTest(eid=eid), self.assertRaisesRegex(ValueError, '入选'):
                self.build(supporting_evidence=[eid])
        self.data['documents'][0]['events'][0]['grounding']['status'] = 'not_applicable'
        with self.assertRaisesRegex(ValueError, '原文'):
            self.build()

    def test_interpretation_without_context_and_single_event_long_term_are_rejected(self):
        without_context = copy.deepcopy(self.prepared)
        without_context.context_ids.clear()
        from claim_ledger import compile_claims
        with self.assertRaisesRegex(ValueError, '上下文'):
            compile_claims(claim_response(kind='inference', supporting_evidence=[self.eid], counter_review='not_found_in_selected'), self.data, without_context, 'analysis')
        for scope in ('repeated', 'long_term'):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, '范围'):
                self.build(scope=scope)

    def test_support_and_counter_cannot_overlap_or_silently_omit_counter_review(self):
        with self.assertRaisesRegex(ValueError, '重复'):
            self.build(counter_evidence=[{'evidence_id': self.eid, 'target': '当前材料', 'reason': '反向说明'}], counter_review='found')
        with self.assertRaisesRegex(ValueError, '反证'):
            self.build(counter_review='found')
        with self.assertRaisesRegex(ValueError, '支持'):
            self.build(supporting_evidence=[])

    def test_selected_counter_evidence_is_preserved_without_trusting_extraction_role(self):
        from claim_ledger import compile_claims, render_claim_report
        prepared = report_generation.prepare_final_input('合成', '', '',
            '### 事件\n- ' + self.eid + '\n### 反证\n- ' + self.other, [], ledger=self.data)
        result = compile_claims(claim_response('本次有回应，解释仍需结合前后文。', kind='inference',
            supporting_evidence=[self.eid], counter_evidence=[{'evidence_id': self.other,
                'target': '本次有回应', 'reason': '另一处原文对回应情况提出反向线索。'}], counter_review='found'),
            self.data, prepared, 'analysis')
        self.assertEqual(result['claims'][0]['counter_evidence'][0]['evidence_id'], self.other)
        self.assertIn('反证：' + self.other, render_claim_report(result, '合成'))

    def test_user_quote_cannot_directly_support_peer_character_and_uncertainty_is_low(self):
        # 合成账本第 23 条 sender=other；第 21 条也为 other，改用真实 me 锚点重绑定。
        from evidence_grounding import ground_document
        doc = self.data['documents'][0]
        doc['events'][0]['evidence'] = self.data['source_messages'][23]['content']
        doc['events'][0]['source_message_ids'] = [self.data['source_messages'][23]['message_id']]
        self.data['documents'] = [ground_document(doc, self.data['source_messages'])]
        eid = self.data['documents'][0]['events'][0]['evidence_id']
        prepared = report_generation.prepare_final_input('合成', '', '', '### 事件\n- ' + eid, [], ledger=self.data)
        from claim_ledger import compile_claims
        with self.assertRaisesRegex(ValueError, '对方'):
            compile_claims(claim_response(kind='inference', subject='peer', supporting_evidence=[eid], counter_review='not_found_in_selected'), self.data, prepared, 'analysis')
        with self.assertRaisesRegex(ValueError, '低'):
            self.build(kind='uncertain', supporting_evidence=[], confidence='high')

    def test_malformed_response_is_rejected_without_accepting_free_form_report(self):
        from claim_ledger import compile_claims
        for response in ('自由成文 <!-- END_FINAL_REPORT -->', '{"claims": []}', '[]',
                         claim_response() + '\n未登记的另一个结论'):
            with self.subTest(response=response), self.assertRaises(ValueError):
                compile_claims(response, self.data, self.prepared, 'analysis')

    def test_audit_checks_saved_input_and_cannot_accept_changed_coverage_or_report(self):
        from claim_ledger import audit_claim_report, render_claim_report
        result = self.build(text='本次协作有回应。')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp, rp, ip = [root / file for file in ('claims.json', 'report.md', 'input.md')]
            cp.write_text(json.dumps(result), encoding='utf-8')
            rp.write_text(render_claim_report(result, '合成'), encoding='utf-8')
            ip.write_text(self.prepared.prompt, encoding='utf-8')
            self.assertEqual(audit_claim_report(cp, rp, self.data, '合成', ip, 'analysis'), 1)
            ip.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '输入'):
                audit_claim_report(cp, rp, self.data, '合成', ip, 'analysis')
            ip.write_text(self.prepared.prompt, encoding='utf-8')
            result['claims'][0]['coverage']['span_days'] = 999
            cp.write_text(json.dumps(result), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '覆盖'):
                audit_claim_report(cp, rp, self.data, '合成', ip, 'analysis')
            result['claims'][0]['coverage']['span_days'] = 0
            cp.write_text(json.dumps(result), encoding='utf-8')
            rp.write_text(render_claim_report(result, '合成') + '未登记的结论', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '未登记'):
                audit_claim_report(cp, rp, self.data, '合成', ip, 'analysis')

    def test_context_at_window_edge_does_not_count_as_complete_review(self):
        from report_context import select_evidence_context
        for budget in range(700, 1600, 50):
            result = select_evidence_context(self.eid + ' ' + self.other, self.data, radius=2, max_bytes=budget)
            for eid, center in ((self.eid, 22), (self.other, 20)):
                if eid in result.context_ids:
                    for position in range(center - 2, center + 3):
                        self.assertIn(f'连续消息{position}\n', result.text)

    def test_long_term_scope_uses_support_dates_not_counter_dates(self):
        from analyze import parse_chat_messages
        from evidence_grounding import ground_document
        from claim_ledger import compile_claims
        messages = parse_chat_messages([
            {'type': '文本消息', 'isSend': 0, 'formattedTime': time, 'content': f'协作回应{n}'}
            for n, time in enumerate(('2026-01-01 10:00:00', '2026-01-20 10:00:00', '2026-02-10 10:00:00'))])['messages']
        doc = copy.deepcopy(self.data['documents'][0])
        doc['events'] = [{'summary': '回应', 'evidence': message['content'], 'confidence': 'medium',
                          'source_message_ids': [message['message_id']]} for message in messages]
        doc['counter_evidence'] = []
        bound = ground_document(doc, messages)
        data = {'source_id': messages[0]['source_id'], 'source_messages': messages, 'documents': [bound]}
        ids = [item['evidence_id'] for item in bound['events']]
        prepared = report_generation.prepare_final_input('合成', '', '', '### 事件\n' + '\n'.join('- ' + eid for eid in ids), [], ledger=data)
        response = claim_response('多次协作有回应。', kind='inference', scope='long_term', supporting_evidence=ids, counter_review='not_found_in_selected')
        result = compile_claims(response, data, prepared, 'analysis')
        self.assertEqual(result['claims'][0]['coverage']['span_days'], 40)
        response = claim_response(kind='inference', scope='long_term', supporting_evidence=ids[:1], counter_evidence=[
            {'evidence_id': eid, 'target': '当前材料', 'reason': '反向线索'} for eid in ids[1:]], counter_review='found')
        with self.assertRaisesRegex(ValueError, '范围'):
            compile_claims(response, data, prepared, 'analysis')

    def test_pipeline_publishes_both_ledgers_and_rejects_invalid_replacement(self):
        import analysis_artifacts
        import chunk_pipeline
        import chunking
        import llm_client
        from artifact_audit import audit_artifacts
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.json'
            source.write_bytes((Path(__file__).parent / 'fixtures/web_smoke.json').read_bytes())
            artifacts = analysis_artifacts.build_artifacts(source, root / 'out')
            config = SimpleNamespace(api_base='https://test.invalid', api_key='test', model='test')
            common_system = report_generation.build_report_system_prompt('skill', 'skill')
            chunk = json.dumps({key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')})
            with patch.object(llm_client, 'call_llm', side_effect=with_reviews([chunk, claim_response(), claim_response(report_kind='persona')])):
                result = chunk_pipeline.run_single_workflow(artifacts, 'skill', config, 1,
                    settings=chunking.ChunkSettings(chunk_sleep=0), report_system_prompt=common_system)
                report_generation.generate_persona_report(artifacts, 'skill', config, 1, report_system_prompt=common_system)
            audit = audit_artifacts(artifacts, require_persona=True)
            self.assertTrue(audit.ok, audit.errors)
            self.assertEqual(audit.stats['analysis_claims'], 1)
            self.assertEqual(audit.stats['persona_claims'], 1)
            files = [artifacts[key] for key in ('analysis_file', 'analysis_claims_file', 'analysis_input_file', 'persona_file', 'persona_claims_file', 'persona_input_file', 'latest_run_file')]
            before = [file.read_bytes() for file in files]
            with patch.object(chunk_pipeline, 'call_llm_with_retry', return_value='unregistered <!-- END_FINAL_REPORT -->'):
                with self.assertRaisesRegex(ValueError, 'JSON'):
                    chunk_pipeline.run_single_workflow(artifacts, 'skill', config, 1, settings=chunking.ChunkSettings(chunk_sleep=0),
                        resume_run_id=result['run_id'], report_system_prompt=common_system)
            self.assertEqual([file.read_bytes() for file in files], before)
            artifacts['persona_claims_file'].unlink()
            self.assertFalse(audit_artifacts(artifacts, require_persona=True).ok)
            # 关系产物完整时只补侧写账本，不重新抽取和生成关系报告。
            from analysis_service import analyze_conversation
            with patch.object(llm_client, 'call_llm', side_effect=with_reviews([claim_response(report_kind='persona')])) as api:
                analyze_conversation(artifacts, 'skill', 'skill', config, 1)
            self.assertEqual(api.call_count, 2)
            self.assertTrue(audit_artifacts(artifacts, require_persona=True).ok)


if __name__ == '__main__':
    unittest.main()
