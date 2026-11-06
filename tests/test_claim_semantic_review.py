"""逐条语义复核的协议与发布门槛，只使用合成材料和模型替身。"""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from types import SimpleNamespace

from test_claim_retrieval import source, QUESTIONS
from claim_fixtures import claim_response, claim_revision
from claim_ledger import generate_claims, audit_claim_report, render_claim_report
from report_generation import prepare_final_input


class ClaimSemanticReviewTests(unittest.TestCase):
    def test_review_distinguishes_literal_participant_fields_and_proposed_advice(self):
        from claim_semantic_review import REVIEW_INSTRUCTIONS
        self.assertIn('null可表示原话没有显式指代或归属不明',REVIEW_INSTRUCTIONS)
        self.assertIn('仅真实存在歧义时须保留疑点',REVIEW_INSTRUCTIONS)
        self.assertIn('建议或验证方案',REVIEW_INSTRUCTIONS)
        self.assertIn('提供帮助与接受帮助、向谁倾诉与听谁倾诉',REVIEW_INSTRUCTIONS)
        self.assertIn('报告作者提出的后续建议无需录入actions',REVIEW_INSTRUCTIONS)
        self.assertNotIn('actor/target为null时核对正文也保留归属疑点',REVIEW_INSTRUCTIONS)

    def test_malformed_review_has_one_protocol_retry_without_regenerating_claims(self):
        from claim_ledger import compile_claims, _evidence_index
        from claim_semantic_review import review_claims
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        compiled = compile_claims(claim_response(),data,prepared,'analysis')
        request = Mock(side_effect=['{"reviews":[]}', json.dumps({'reviews':[
            {'claim_id':'R_1','verdict':'pass','reason':'保持明确存疑。'}]})])
        with tempfile.TemporaryDirectory() as directory:
            record = review_claims(compiled,prepared,_evidence_index(data),request,Path(directory))
            self.assertTrue((Path(directory)/'repair_response.txt').exists())
        self.assertEqual(request.call_count,2)
        self.assertIn('缺项',request.call_args.args[0])
        self.assertEqual(record['reviews'][0]['claim_id'],compiled['claims'][0]['claim_id'])

    def test_repeated_malformed_review_stops_after_one_protocol_retry(self):
        from claim_ledger import compile_claims, _evidence_index
        from claim_semantic_review import review_claims
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        compiled = compile_claims(claim_response(),data,prepared,'analysis')
        request = Mock(return_value='{"reviews":[]}')
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(RuntimeError):
            review_claims(compiled,prepared,_evidence_index(data),request,Path(directory))
        self.assertEqual(request.call_count,2)

    def test_reviewer_receives_same_global_statistics_and_coverage_as_generator(self):
        from claim_ledger import compile_claims, _evidence_index
        from claim_semantic_review import build_review_prompt
        data = source({78: '重复表达', 138: '重复表达'})
        stats = '2026年1月共240条，2月共8条，3月共180条。'
        coverage = '全部300条消息完成抽取，本次报告仅选取部分完整会话。'
        prepared = prepare_final_input('合成', stats, coverage, '旧证据解释', ['旧阶段解释'],
                                       ledger=data, review_questions=QUESTIONS)
        compiled = compile_claims(claim_response(), data, prepared, 'analysis')
        prompt = build_review_prompt(compiled, prepared, _evidence_index(data))
        self.assertIn(stats, prompt)
        self.assertIn(coverage, prompt)
        self.assertIn('同文计数 P_79（全源2次，本发送方2次）', prompt)
        self.assertNotIn('旧证据解释', prompt)
        self.assertNotIn('旧阶段解释', prompt)
        self.assertNotIn('## 本次具体任务', prompt)

    def test_false_counter_check_overrides_pass_and_enters_bounded_repair(self):
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        options = dict(kind='inference', subject='peer', supporting_evidence=['P_79'],
                       counter_review='found', counter_evidence=[{'evidence_id':'P_9',
                           'target':'表达了解意愿','reason':'当时忙碌。'}])
        generate = Mock(side_effect=[claim_response('对方本次表达了解意愿。', **options),
            claim_revision('对方本次表达了解意愿。', **{**options, 'counter_evidence':[],
                                                       'counter_review':'not_found_in_selected'})])
        first = True
        def review(prompt):
            nonlocal first
            drafts, _ = json.JSONDecoder().raw_decode(prompt.split('待复核结论及逐字引用\n',1)[1])
            row = {'claim_id':'R_1','verdict':'pass','reason':'判断有来源。'}
            if first:
                row['counter_checks'] = [{'evidence_id':drafts[0]['counter_evidence'][0]['evidence_id'],
                    'directly_contradicts':False,'reason':'忙碌不能反驳之后的了解意愿。'}]
                first = False
            return json.dumps({'reviews':[row]})
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_claims(prepared,data,'analysis',generate,attempt_dir=Path(tmp),review_request=review)
        self.assertEqual(generate.call_count,2)
        self.assertEqual(result['claims'][0]['counter_evidence'],[])

    def test_semantic_revision_requires_a_complete_failed_claim(self):
        from claim_ledger import apply_claim_revisions
        claim = json.loads(claim_response('对方每次都会回应。', kind='inference',
            counter_evidence=[{'evidence_id':'E_example','target':'每次都会回应','reason':'一次明确拒绝。'}],
            counter_review='found').split('\n<!--')[0])['claims'][0]
        with self.assertRaisesRegex(ValueError, '完整'):
            apply_claim_revisions({'revisions':[{'claim_number':1,'changes':{
                'text':'对方有时回应，也出现过拒绝。'}}]}, {'claims':[claim]}, {1})

    def test_short_review_ids_resolve_to_canonical_claim_ids(self):
        from claim_ledger import compile_claims, _evidence_index
        from claim_semantic_review import review_claims
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        compiled = compile_claims(claim_response(),data,prepared,'analysis')
        request = Mock(return_value=json.dumps({'reviews':[{'claim_id':'R_1','verdict':'pass','reason':'明确存疑。'}]}))
        with tempfile.TemporaryDirectory() as tmp:
            record = review_claims(compiled,prepared,_evidence_index(data),request,Path(tmp))
            self.assertEqual(record['reviews'][0]['claim_id'],compiled['claims'][0]['claim_id'])
            self.assertIn('"review_id": "R_1"',request.call_args.args[0])
            request.return_value = json.dumps({'reviews':[{'claim_id':'R_999','verdict':'pass','reason':'未知编号。'}]})
            with self.assertRaises(RuntimeError):
                review_claims(compiled,prepared,_evidence_index(data),request,Path(tmp))

    def test_review_may_confirm_a_known_limitation_is_not_a_counter(self):
        from claim_semantic_review import REVIEW_INSTRUCTIONS, validate_reviews
        self.assertIn('仅对counter_evidence中的实际反证，directly_contradicts为false时须标revise', REVIEW_INSTRUCTIONS)
        self.assertIn('limiting_evidence中的已登记限制可额外检查，其directly_contradicts须为false', REVIEW_INSTRUCTIONS)
        claims = [{'claim_id':'C_sample','counter_evidence':[], 'limiting_evidence':[{'evidence_id':'E_limit'}]}]
        row = {'claim_id':'C_sample','verdict':'pass','reason':'限制归类一致。', 'counter_checks':[
            {'evidence_id':'E_limit','directly_contradicts':False,'reason':'仅限制条件。'}]}
        validate_reviews(claims,[row])
        row['counter_checks'][0]['directly_contradicts'] = True
        with self.assertRaises(RuntimeError):
            validate_reviews(claims,[row])
        row['counter_checks'][0].update(evidence_id='E_unknown',directly_contradicts=False)
        with self.assertRaises(RuntimeError):
            validate_reviews(claims,[row])
        claims[0]['counter_evidence'] = [{'evidence_id':'E_counter'}]
        row['counter_checks'] = [
            {'evidence_id':'E_limit','directly_contradicts':False,'reason':'仅限制条件。'},
            {'evidence_id':'E_counter','directly_contradicts':True,'reason':'明确反驳当前命题。'},
        ]
        validate_reviews(claims,[row])
        row['counter_checks'].pop()
        with self.assertRaises(RuntimeError):
            validate_reviews(claims,[row])

    def test_semantic_revision_changes_only_failed_claims(self):
        from claim_ledger import apply_claim_revisions
        base = {'claims':[{'text':'第一条','kind':'fact'},{'text':'第二条','kind':'inference'}]}
        revised = apply_claim_revisions({'revisions':[{'claim_number':2,'changes':{'text':'收缩后的第二条','kind':'inference'}}]},base,{2})
        self.assertEqual(revised['claims'][0],base['claims'][0])
        self.assertEqual(revised['claims'][1],{'text':'收缩后的第二条','kind':'inference'})
        self.assertEqual(base['claims'][1]['text'],'第二条')
        for edits in ([], [{'claim_number':1,'changes':{'text':'改动已通过条目'}}],
                      [{'claim_number':2,'changes':{'new_field':'无效'}}],
                      [{'claim_number':2,'changes':{}},{'claim_number':2,'changes':{}}]):
            with self.assertRaises(ValueError):
                apply_claim_revisions({'revisions':edits},base,{2})

    def test_output_contract_states_uncertain_scope_and_all_narrative_fields(self):
        from claim_ledger import claim_contract
        contract = claim_contract('analysis')
        self.assertIn('uncertain 的 scope 固定为 local，confidence 固定为 low', contract)
        self.assertIn('text、context_review、uncertainty、reason 和 target 均须为纯文本', contract)

    def test_structural_repair_does_not_consume_the_semantic_repair(self):
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        options = dict(kind='inference', subject='peer', supporting_evidence=['P_79'],
                       counter_review='not_found_in_selected')
        generate = Mock(side_effect=['invalid json', claim_response('对方一贯愿意了解情况。', **options),
                                     claim_revision('这次对方表示愿意了解情况。', **options)])
        outcomes = iter([('revise','局部材料不足以证明一贯表现。'), ('pass','已限定到本次表达。')])
        def review(prompt):
            verdict, reason = next(outcomes)
            cid = re.findall(r'C_[0-9a-f]{32}', prompt)[0]
            return json.dumps({'reviews':[{'claim_id':cid,'verdict':verdict,'reason':reason}]})
        verifier = Mock(side_effect=review)
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_claims(prepared,data,'analysis',generate,attempt_dir=Path(tmp),review_request=verifier)
        self.assertEqual(generate.call_count,3)
        self.assertEqual(verifier.call_count,2)
        self.assertEqual(result['semantic_review']['reviews'][0]['verdict'],'pass')

    def test_repeated_structural_failure_stops_after_one_structural_repair(self):
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        generate, verifier = Mock(return_value='invalid json'), Mock()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            generate_claims(prepared,data,'analysis',generate,attempt_dir=Path(tmp),review_request=verifier)
        self.assertEqual(generate.call_count,2)
        verifier.assert_not_called()

    def test_repeated_semantic_failure_stops_after_one_semantic_repair(self):
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        generate = Mock(side_effect=[claim_response(), claim_revision(uncertainty='结论仍需逐字核验。')])
        def review(prompt):
            cid = re.findall(r'C_[0-9a-f]{32}', prompt)[0]
            return json.dumps({'reviews':[{'claim_id':cid,'verdict':'revise','reason':'结论仍需修改。'}]})
        verifier = Mock(side_effect=review)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError,'独立语义复核'):
            generate_claims(prepared,data,'analysis',generate,attempt_dir=Path(tmp),review_request=verifier)
        self.assertEqual(generate.call_count,2)
        self.assertEqual(verifier.call_count,2)

    def test_review_covers_explanations_that_will_appear_in_the_report(self):
        from claim_ledger import compile_claims, _evidence_index
        from claim_semantic_review import build_review_prompt
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        compiled = compile_claims(claim_response(context_review='解释层也需要逐字核验',
                                                uncertainty='效果是否实现仍待核验'), data, prepared, 'analysis')
        prompt = build_review_prompt(compiled, prepared, _evidence_index(data))
        self.assertIn('解释层也需要逐字核验', prompt)
        self.assertIn('效果是否实现仍待核验', prompt)

    def test_failed_counter_review_remains_actionable_but_cannot_pass_without_checks(self):
        from claim_semantic_review import validate_reviews
        claims = [{'claim_id': 'C_example', 'counter_evidence': [{'evidence_id': 'E_example'}]}]
        revise = [{'claim_id': 'C_example', 'verdict': 'revise', 'reason': '该引用不能反驳当前命题。'}]
        validate_reviews(claims, revise)
        validate_reviews(claims, [{**revise[0], 'counter_checks': [
            {'evidence_id': 'E_different', 'directly_contradicts': False, 'reason': '需先修改命题和引用。'}]}])
        with self.assertRaisesRegex(RuntimeError, '反证'):
            validate_reviews(claims, [{**revise[0], 'verdict':'pass'}])
        with self.assertRaisesRegex(RuntimeError, '反证'):
            validate_reviews(claims, [{**revise[0], 'verdict': 'pass', 'counter_checks': [
                {'evidence_id': 'E_example', 'directly_contradicts': False, 'reason': '只有限制作用。'}]}])

    def test_normal_workflow_plans_and_reviews_when_original_material_is_incomplete(self):
        import analysis_artifacts
        import analysis_service
        import llm_client
        from artifact_audit import audit_artifacts
        from claim_semantic_review import REVIEW_INSTRUCTIONS
        stages = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / 'input.json'
            input_path.write_bytes((Path(__file__).parent / 'fixtures/web_smoke.json').read_bytes())
            artifacts = analysis_artifacts.build_artifacts(input_path, root / 'out')
            query = artifacts['messages'][0]['content']
            def response(**kwargs):
                prompt = kwargs['user_prompt']
                if not stages:
                    stages.append('chunk')
                    return json.dumps({key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')})
                if '## 本次任务：提出待复核命题' in prompt:
                    stages.append('plan')
                    return json.dumps({'questions':[{'proposition':'本次交流的情况', 'support_queries':[query], 'challenge_queries':[query]}]})
                if kwargs['system_prompt'] == REVIEW_INSTRUCTIONS:
                    stages.append('review')
                    return json.dumps({'reviews':[{'claim_id':cid,'verdict':'pass','reason':'明确存疑，未越过材料范围。'}
                                                  for cid in dict.fromkeys(re.findall(r'C_[0-9a-f]{32}', prompt))]})
                kind = 'persona' if '## 本次具体任务：对方人物侧写' in prompt else 'analysis'
                stages.append(kind)
                return claim_response(report_kind=kind)
            config = SimpleNamespace(api_base='https://test.invalid', api_key='test', model='test')
            with patch('claim_retrieval.needs_retrieval', return_value=True), patch.object(llm_client, 'call_llm', side_effect=response):
                analysis_service.analyze_conversation(artifacts, 'skill', 'skill', config, 1)
            self.assertEqual(stages, ['chunk','plan','analysis','review','persona','review'])
            self.assertTrue(audit_artifacts(artifacts, require_persona=True, require_claims=True).ok)

    def test_semantic_failure_repairs_claim_and_requires_a_new_complete_review(self):
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        options = dict(kind='inference', subject='peer', supporting_evidence=['P_79'],
                       counter_review='not_found_in_selected')
        generate = Mock(side_effect=[claim_response('对方一贯认真了解情况。', **options),
                                     claim_revision('这次对方表示愿意了解情况。', **options)])
        responses = iter([('revise', '局部材料不能证明一贯表现。'), ('pass', '判断限定这次交流，原话支持。')])
        def review(prompt):
            verdict, reason = next(responses)
            cid = re.findall(r'C_[0-9a-f]{32}', prompt)[0]
            return json.dumps({'reviews': [{'claim_id': cid, 'verdict': verdict, 'reason': reason}]}, ensure_ascii=False)
        verifier = Mock(side_effect=review)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = generate_claims(prepared, data, 'analysis', generate, attempt_dir=root,
                                     review_request=verifier)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(verifier.call_count, 2)
            self.assertIn('局部材料不能证明一贯表现', generate.call_args.args[0])
            self.assertIn('第 1 条结论（', generate.call_args.args[0])
            self.assertIn('"support_speakers": ["other"]', generate.call_args.args[0])
            self.assertIn('"positions": [79]', generate.call_args.args[0])
            self.assertIn('我愿意了解', verifier.call_args.args[0])
            self.assertEqual(result['semantic_review']['reviews'][0]['verdict'], 'pass')
            (root / 'input.md').write_text(prepared.prompt, encoding='utf-8')
            (root / 'report.md').write_text(render_claim_report(result, '合成'), encoding='utf-8')
            original_hash = result['semantic_review']['claims_sha256']
            result['semantic_review']['claims_sha256'] = 'tampered'
            (root / 'claims.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '语义复核'):
                audit_claim_report(root / 'claims.json', root / 'report.md', data, '合成',
                                   root / 'input.md', 'analysis')
            result['semantic_review']['claims_sha256'] = original_hash
            result.pop('semantic_review')
            (root / 'claims.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '语义复核'):
                audit_claim_report(root / 'claims.json', root / 'report.md', data, '合成',
                                   root / 'input.md', 'analysis')

    def test_incomplete_review_cannot_be_treated_as_passed(self):
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, '语义复核'):
            generate_claims(prepared, data, 'analysis', Mock(return_value=claim_response()),
                            attempt_dir=Path(tmp), review_request=Mock(return_value='{"reviews":[]}'))
