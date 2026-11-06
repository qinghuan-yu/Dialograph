"""命题驱动的双向字面检索、完整会话回填及请求上限；只用合成材料。"""
import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from analyze import parse_chat_messages
from evidence_grounding import ground_document, grounding_issues, GROUNDING_VERSION, LEDGER_VERSION


def source(text_overrides=None, *, session_break=140):
    start = datetime(2021, 3, 2, 9)
    texts = ['普通补充' + str(n) for n in range(140)]
    texts[5] = '能听我说一件事吗？'
    texts[8] = '现在有点忙，晚点回复你。'
    texts[51] = '刚才没认真回复，是我疏忽了。'
    texts[78] = '你遇到什么事了？我愿意了解。'
    texts[120] = '问题说清楚了，我已经联系相关人员处理。'
    for number, text in (text_overrides or {}).items():
        texts[number] = text
    rows = [{'type': '文本消息', 'isSend': n % 2,
             'formattedTime': (start + timedelta(seconds=n * 10, days=int(n >= session_break))).strftime('%Y-%m-%d %H:%M:%S'),
             'content': text} for n, text in enumerate(texts)]
    messages = parse_chat_messages(rows)['messages']
    doc = {'chunk_index': 1, 'parse_status': 'ok', 'message_range': {'start': 1, 'end': len(messages)},
           'time_range': {'start': messages[0]['time_str'], 'end': messages[-1]['time_str']},
           **{key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')}}
    doc['events'] = [{'summary': '待复核节点', 'evidence': texts[n], 'confidence': 'low',
                      'source_message_ids': [messages[n]['message_id']]} for n in (5, 8, 51, 120)]
    doc = ground_document(doc, messages)
    return {'schema_version': LEDGER_VERSION, 'grounding_version': GROUNDING_VERSION,
            'chunk_count': 1, 'source_id': messages[0]['source_id'], 'source_messages': messages,
            'documents': [doc], 'grounding_issues': grounding_issues([doc])}


QUESTIONS = [{'proposition': '对方没有尝试了解情况。',
              'support_queries': ['晚点回复'], 'challenge_queries': ['愿意了解']}]


def competing_sessions():
    start = datetime(2022, 8, 1, 9)
    groups = [
        ['第一次讨论方案并完成修改'] + ['补充说明甲' + str(n) for n in range(15)],
        ['再次讨论方案、修改、提交、检查、完成、更新'] + ['补充说明乙' + str(n) for n in range(15)],
        ['发生意外后主动承担并妥善处理'] + ['处理进展' + str(n) for n in range(40)],
    ]
    rows = []
    starts = []
    for day, texts in enumerate(groups):
        starts.append(len(rows))
        rows.extend({'type': '文本消息', 'isSend': n % 2,
                     'formattedTime': (start + timedelta(days=day, seconds=n * 10)).strftime('%Y-%m-%d %H:%M:%S'),
                     'content': text} for n, text in enumerate(texts))
    messages = parse_chat_messages(rows)['messages']
    doc = source()['documents'][0]
    doc = {key: value for key, value in doc.items()
           if key in ('chunk_index', 'parse_status', 'message_range', 'time_range',
                      'events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')}
    doc['message_range'] = {'start': 1, 'end': len(messages)}
    doc['time_range'] = {'start': messages[0]['time_str'], 'end': messages[-1]['time_str']}
    doc['events'] = [{'summary': '可复核过程', 'evidence': messages[n]['content'], 'confidence': 'low',
                      'source_message_ids': [messages[n]['message_id']]} for n in starts]
    doc = ground_document(doc, messages)
    return {'schema_version': LEDGER_VERSION, 'grounding_version': GROUNDING_VERSION,
            'chunk_count': 1, 'source_id': messages[0]['source_id'], 'source_messages': messages,
            'documents': [doc], 'grounding_issues': grounding_issues([doc])}


class ClaimRetrievalTests(unittest.TestCase):
    def test_retrieved_text_shows_global_repetition_count_even_outside_selected_session(self):
        from report_context import select_evidence_context, source_repetition_notes, ReviewSession
        repeated = '一段在多个场景原样发送的文字。'
        data = source({78: repeated, 138: repeated, 139: repeated}, session_break=80)
        result = select_evidence_context('', data, select_from_ledger=True,
            review_sessions=[ReviewSession(0, 79, frozenset({'Q1'}), 1)])
        notes = source_repetition_notes(data, result.source_positions)
        self.assertIn('同文计数 P_79（全源3次，本发送方2次）', notes)
        self.assertIn('P_79\n' + repeated, result.text)
        self.assertNotIn('P_139', result.text)
        self.assertNotIn('P_140', result.text)
        self.assertLessEqual(len(result.text.encode('utf-8')), 48000)
        self.assertLessEqual(len(notes.encode('utf-8')), 4000)
        self.assertEqual(source_repetition_notes(data, result.source_positions, max_bytes=1), '')

    def test_retrieved_reports_use_raw_sources_without_secondhand_interpretations(self):
        from report_generation import prepare_final_input, prepare_persona_input
        data = source()
        args = ('合成', '统计摘要', '覆盖说明', '旧摘要推断', ['旧阶段推断'])
        relation = prepare_final_input(*args, ledger=data, review_questions=QUESTIONS)
        persona = prepare_persona_input(*args, '关系报告的二手推断', ledger=data, review_questions=QUESTIONS)
        for prepared in (relation, persona):
            for old in ('待复核节点', '旧摘要推断', '旧阶段推断', '关系报告的二手推断'):
                self.assertNotIn(old, prepared.prompt)
            self.assertNotIn(QUESTIONS[0]['proposition'], prepared.prompt)
            self.assertIn('你遇到什么事了？我愿意了解。', prepared.prompt)
            self.assertIn('P_79', prepared.prompt)
            self.assertIn('涉及失误或冲突时，先写实际发生的行为、主体与影响', prepared.prompt)
            self.assertIn('样本数量、情境描述及总结须与各条具体结论一致', prepared.prompt)
            self.assertTrue(prepared.evidence_ids)
            self.assertTrue(all(eid.startswith('M_') for eid in prepared.evidence_ids))
            self.assertEqual(prepared.context_ids, prepared.evidence_ids)
            self.assertNotIn(data['documents'][0]['events'][0]['evidence_id'], prepared.prompt)
        self.assertEqual(relation.retrieval, persona.retrieval)
        self.assertEqual(relation.retrieval['questions'], QUESTIONS)
        self.assertEqual(relation.source_positions, persona.source_positions)

    def test_omitted_evidence_id_can_resolve_to_its_visible_complete_source(self):
        from claim_ledger import compile_claims
        from claim_fixtures import claim_response
        from report_generation import prepare_final_input
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        item = data['documents'][0]['events'][0]
        eid = item['evidence_id']
        mid = item['grounding']['source_refs'][0]['message_id']
        prepared.evidence_ids.discard(eid)
        prepared.context_ids.discard(eid)
        response = claim_response('用户提出交流请求。', kind='fact', subject='user',
                                  supporting_evidence=[eid], counter_review='not_found_in_selected')
        compiled = compile_claims(response, data, prepared, 'analysis')
        self.assertEqual(compiled['claims'][0]['supporting_evidence'], [mid])
        data['documents'][0]['events'][0]['evidence'] = '被篡改的引文'
        with self.assertRaises(ValueError):
            compile_claims(response, data, prepared, 'analysis')

    def test_session_combination_can_replace_an_early_expensive_choice(self):
        from report_context import ReviewSession, select_evidence_context
        data = competing_sessions()
        expensive = ReviewSession(32, 72, frozenset({'a', 'b'}), 2, ((1, 1.0), (2, 1.0)))
        first = ReviewSession(0, 15, frozenset({'a', 'c'}), 1.8, ((1, 0.9), (3, 0.9)))
        second = ReviewSession(16, 31, frozenset({'b'}), 0.9, ((2, 0.9),))
        budget = len(select_evidence_context('', data, select_from_ledger=True,
                                            review_sessions=[expensive]).text.encode('utf-8')) + 30
        result = select_evidence_context('', data, select_from_ledger=True, max_bytes=budget,
                                         review_sessions=[first, second, expensive])
        self.assertTrue(set(range(1, 33)) <= result.source_positions)
        self.assertNotIn(33, result.source_positions)

    def test_rephrased_queries_with_identical_hits_do_not_change_selection(self):
        from claim_retrieval import retrieve_claim_context
        data = competing_sessions()
        questions = [
            {'proposition': '方案', 'support_queries': ['再次讨论'], 'challenge_queries': ['第一次讨论']},
            {'proposition': '意外处理', 'support_queries': ['主动承担'], 'challenge_queries': ['妥善处理']},
        ]
        first = retrieve_claim_context(data, questions, max_bytes=6000)
        redundant = copy.deepcopy(questions)
        redundant[0]['support_queries'] += ['再次讨论方案', '提交、检查', '完成、更新']
        second = retrieve_claim_context(data, redundant, max_bytes=6000)
        self.assertEqual(first.source_positions, second.source_positions)

    def test_raw_reference_requires_complete_context_even_for_local_fact(self):
        from claim_ledger import compile_claims
        from claim_fixtures import claim_response
        from report_generation import prepare_final_input
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        mid = data['source_messages'][78]['message_id']
        prepared.context_ids.discard(mid)
        response = claim_response('对方说愿意了解。', kind='fact', subject='peer',
                                  supporting_evidence=[mid], counter_review='not_found_in_selected')
        with self.assertRaisesRegex(ValueError, '原文'):
            compile_claims(response, data, prepared, 'analysis')

    def test_raw_message_reference_without_extracted_anchor_is_citable_and_auditable(self):
        from claim_retrieval import retrieve_claim_context
        from claim_ledger import compile_claims, audit_claim_report, render_claim_report
        from claim_fixtures import claim_response
        from report_generation import prepare_final_input
        from report_references import ReferenceCatalog
        data = source()
        data['documents'][0]['events'] = []
        data['documents'] = [ground_document(data['documents'][0], data['source_messages'])]
        data['grounding_issues'] = grounding_issues(data['documents'])
        result = retrieve_claim_context(data, QUESTIONS)
        message = data['source_messages'][78]
        self.assertIn(message['message_id'], result.context_ids)
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        response = claim_response('对方在这次交流中表示愿意了解情况。', kind='inference',
                                  subject='peer', supporting_evidence=[message['message_id']],
                                  counter_review='not_found_in_selected')
        compiled = compile_claims(response, data, prepared, 'analysis')
        self.assertEqual(compiled['claims'][0]['coverage']['positions'], [79])
        self.assertEqual(ReferenceCatalog(data).read(message['message_id'])['kind'], 'message')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'claims.json').write_text(json.dumps(compiled, ensure_ascii=False), encoding='utf-8')
            (root / 'input.md').write_text(prepared.prompt, encoding='utf-8')
            (root / 'report.md').write_text(render_claim_report(compiled, '合成'), encoding='utf-8')
            self.assertEqual(audit_claim_report(root / 'claims.json', root / 'report.md', data,
                                               '合成', root / 'input.md', 'analysis'), 1)
        corrupted = copy.deepcopy(data)
        corrupted['source_messages'][78]['content'] = '替换后的不同原文'
        with self.assertRaises(ValueError):
            compile_claims(response, corrupted, prepared, 'analysis')

    def test_visible_neighbor_can_be_cited_by_position_and_resolves_to_stable_message_id(self):
        from claim_ledger import compile_claims
        from claim_fixtures import claim_response
        from report_generation import prepare_final_input
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        response = claim_response('对方在这次交流中提供补充说明。', kind='fact', subject='peer',
                                  supporting_evidence=['P_81'], counter_review='not_found_in_selected')
        compiled = compile_claims(response, data, prepared, 'analysis')
        self.assertEqual(compiled['claims'][0]['supporting_evidence'], [data['source_messages'][80]['message_id']])
        self.assertIn('P_81', prepared.prompt)
        with self.assertRaises(ValueError):
            compile_claims(response.replace('P_81', 'P_9999'), data, prepared, 'analysis')
        tampered = copy.deepcopy(prepared)
        tampered.retrieval['source_aliases']['P_81'] = data['source_messages'][78]['message_id']
        with self.assertRaisesRegex(ValueError, '原文短引用'):
            compile_claims(response, data, tampered, 'analysis')

    def test_queries_for_one_proposition_do_not_starve_another_proposition(self):
        from claim_retrieval import retrieve_claim_context
        data = competing_sessions()
        minimal = [
            {'proposition': '协作过程', 'support_queries': ['第一次讨论'], 'challenge_queries': ['第一次讨论']},
            {'proposition': '处理意外', 'support_queries': ['主动承担'], 'challenge_queries': ['发生意外']},
        ]
        budget = len(retrieve_claim_context(data, minimal).text.encode('utf-8')) + 30
        broad = copy.deepcopy(minimal)
        broad[0]['support_queries'] += ['再次讨论', '修改', '提交', '检查', '完成', '更新']
        result = retrieve_claim_context(data, broad, max_bytes=budget)
        self.assertIn(33, result.source_positions, '稀缺命题的完整过程被同一命题的查询数量挤掉')
        self.assertIn(73, result.source_positions)
        self.assertLessEqual(len(result.text.encode('utf-8')), budget)

    def test_retrieval_distinguishes_no_match_and_budget_omission(self):
        from claim_retrieval import retrieve_claim_context
        data = source()
        questions = copy.deepcopy(QUESTIONS)
        questions[0]['support_queries'].append('没有出现的字面短语')
        result = retrieve_claim_context(data, questions, max_bytes=1000)
        by_term = {row['query']: row for row in result.retrieval['queries']}
        self.assertEqual(by_term['没有出现的字面短语']['status'], 'no_match')
        self.assertEqual(by_term['愿意了解']['status'], 'budget_exceeded')
        self.assertEqual(result.retrieval['session_decisions'][0]['status'], 'budget_exceeded')

    def test_raw_source_hit_recovers_later_response_and_entire_session(self):
        from claim_retrieval import retrieve_claim_context
        data = source()
        result = retrieve_claim_context(data, QUESTIONS)
        self.assertIn('我愿意了解', result.text)
        for n in (0, 5, 8, 51, 78, 120, 139):
            self.assertIn(data['source_messages'][n]['content'], result.text)
        self.assertEqual(len([eid for eid in result.context_ids if eid.startswith('E_')]), 4)
        self.assertIn(data['source_messages'][78]['message_id'], result.context_ids)
        self.assertEqual(result.retrieval['source_id'], data['source_id'])
        self.assertTrue(all(row['covered'] for row in result.retrieval['queries']))

    def test_budget_cannot_shorten_session_and_certify_its_quotes(self):
        from claim_retrieval import retrieve_claim_context
        result = retrieve_claim_context(source(), QUESTIONS, max_bytes=3000)
        self.assertFalse(result.context_ids)
        self.assertNotIn('核验引文：', result.text)
        self.assertTrue(any(not row['covered'] for row in result.retrieval['queries']))
        self.assertLessEqual(len(result.text.encode('utf-8')), 3000)

    def test_model_categories_and_confidence_do_not_change_query_selection(self):
        from claim_retrieval import retrieve_claim_context
        data = source()
        other = copy.deepcopy(data)
        doc = other['documents'][0]
        rows = [{'against': row['summary'], 'evidence': row['evidence'], 'confidence': 'high',
                 'source_message_ids': row['source_message_ids']} for row in doc['events']]
        doc['events'], doc['counter_evidence'] = [], rows
        other['documents'] = [ground_document(doc, other['source_messages'])]
        other['grounding_issues'] = grounding_issues(other['documents'])
        first = retrieve_claim_context(data, QUESTIONS)
        second = retrieve_claim_context(other, QUESTIONS)
        self.assertEqual(first.retrieval['source_positions'], second.retrieval['source_positions'])

    def test_failed_search_is_not_a_counter_evidence_judgment(self):
        from claim_retrieval import retrieve_claim_context
        questions = [{'proposition': '是否支持', 'support_queries': ['不存在的精确短语'],
                      'challenge_queries': ['也不存在的短语']}]
        result = retrieve_claim_context(source(), questions)
        self.assertFalse(result.context_ids)
        self.assertTrue(all(row['hits'] == 0 and not row['covered'] for row in result.retrieval['queries']))
        self.assertIn('未命中不代表不存在', result.text)

    def test_mismatched_source_is_rejected_before_search(self):
        from claim_retrieval import retrieve_claim_context
        data = source()
        data['source_messages'][100]['source_id'] = 'S_' + 'f' * 64
        with self.assertRaises(ValueError):
            retrieve_claim_context(data, QUESTIONS)

    def test_same_message_with_multiple_evidence_rows_is_one_physical_hit(self):
        from claim_retrieval import retrieve_claim_context
        data = source()
        doc = data['documents'][0]
        doc['persona_signals'] = [{'trait': '待复核', 'signal': '待复核', 'stability': 'local',
                                  'evidence': doc['events'][0]['evidence'], 'confidence': 'high',
                                  'source_message_ids': doc['events'][0]['source_message_ids']}]
        data['documents'] = [ground_document(doc, data['source_messages'])]
        data['grounding_issues'] = grounding_issues(data['documents'])
        questions = [{'proposition': '这次求助', 'support_queries': ['听我说'], 'challenge_queries': ['普通补充']}]
        result = retrieve_claim_context(data, questions)
        row = next(row for row in result.retrieval['queries'] if row['query'] == '听我说')
        self.assertEqual(row['hits'], 1)

    def test_plan_requires_both_search_directions_but_not_actual_counters(self):
        from claim_retrieval import parse_review_questions
        self.assertEqual(parse_review_questions(json.dumps({'questions': QUESTIONS})), QUESTIONS)
        broken = copy.deepcopy(QUESTIONS)
        broken[0]['challenge_queries'] = []
        with self.assertRaises(ValueError):
            parse_review_questions(json.dumps({'questions': broken}))

    def test_plan_rejects_unbounded_questions_and_query_lists(self):
        from claim_retrieval import parse_review_questions
        with self.assertRaises(ValueError):
            parse_review_questions(json.dumps({'questions': QUESTIONS * 9}))
        too_many = copy.deepcopy(QUESTIONS)
        too_many[0]['support_queries'] = ['词' + str(n) for n in range(9)]
        with self.assertRaises(ValueError):
            parse_review_questions(json.dumps({'questions': too_many}))

    def test_request_cap_is_persistent_and_applies_before_transport(self):
        from llm_usage import BoundedRequests, RequestBudgetExceeded
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'budget.json'
            transport = Mock(side_effect=[TimeoutError('模拟传输失败'), 'ok'])
            limited = BoundedRequests(path, 2)
            with self.assertRaises(TimeoutError):
                limited.call(transport, 'first')
            self.assertEqual(limited.call(transport, 'second'), 'ok')
            again = BoundedRequests(path, 2)
            with self.assertRaises(RequestBudgetExceeded):
                again.call(transport, 'third')
            self.assertEqual(transport.call_count, 2)

    def test_two_stage_report_uses_retrieved_input_and_saves_provenance(self):
        from claim_retrieval import generate_reviewed_claims
        from chunk_evidence import format_evidence_summary
        from claim_fixtures import claim_response
        import report_generation
        data = source()
        summary = format_evidence_summary(data['documents'])
        initial = report_generation.prepare_final_input('合成', '', '', summary, [], ledger=data)
        initial.source_positions = frozenset()
        plan, final = Mock(return_value=json.dumps({'questions': QUESTIONS})), Mock(return_value=claim_response())
        def rebuild(questions):
            return report_generation.prepare_final_input('合成', '', '', summary, [], ledger=data,
                                                        review_questions=questions)
        with tempfile.TemporaryDirectory() as tmp:
            prepared, result = generate_reviewed_claims(initial, data, 'analysis', final, rebuild,
                plan, attempt_dir=Path(tmp), plan_identity='same-system-and-model')
            self.assertEqual(plan.call_count, 1)
            self.assertEqual(final.call_count, 1)
            self.assertIn('我愿意了解', prepared.prompt)
            self.assertIn('retrieval', result)
            self.assertEqual(prepared.retrieval, result['retrieval'])
            self.assertEqual(final.call_args.args[0], prepared.prompt)

    def test_planner_receives_raw_source_omitted_from_initial_report(self):
        from claim_retrieval import generate_reviewed_claims
        from claim_fixtures import claim_response
        from report_generation import prepare_final_input
        data = source()
        initial = prepare_final_input('合成', '', '', '', [], ledger=data)
        self.assertNotIn(data['source_messages'][-1]['content'], initial.prompt)
        plan = Mock(return_value=json.dumps({'questions': QUESTIONS}))
        with tempfile.TemporaryDirectory() as tmp:
            generate_reviewed_claims(initial, data, 'analysis', Mock(return_value=claim_response()),
                lambda questions: prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=questions),
                plan, attempt_dir=Path(tmp), plan_identity='same-source-model')
        prompt = plan.call_args.args[0]
        self.assertIn(data['source_messages'][-1]['content'], prompt)
        self.assertIn(data['source_messages'][78]['content'], prompt)
        self.assertIn('原子命题', prompt)

    def test_final_review_failure_does_not_start_a_third_request(self):
        from claim_retrieval import generate_reviewed_claims
        from chunk_evidence import format_evidence_summary
        import report_generation
        data = source()
        summary = format_evidence_summary(data['documents'])
        initial = report_generation.prepare_final_input('合成', '', '', summary, [], ledger=data)
        initial.source_positions = frozenset()
        plan, final = Mock(return_value=json.dumps({'questions': QUESTIONS})), Mock(return_value='{"claims": []}')
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            generate_reviewed_claims(initial, data, 'analysis', final,
                lambda questions: report_generation.prepare_final_input('合成', '', '', summary, [],
                    ledger=data, review_questions=questions), plan,
                attempt_dir=Path(tmp), plan_identity='same-system-and-model')
        self.assertEqual(plan.call_count + final.call_count, 2)

    def test_retrieval_provenance_cannot_be_changed_after_generation(self):
        from claim_ledger import audit_claim_report, compile_claims, render_claim_report
        from claim_fixtures import claim_response
        from report_generation import prepare_final_input
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        result = compile_claims(claim_response(), data, prepared, 'analysis')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claim_path, report_path, input_path = [root / name for name in ('claims.json', 'report.md', 'input.md')]
            report_path.write_text(render_claim_report(result, '合成'), encoding='utf-8')
            input_path.write_text(prepared.prompt, encoding='utf-8')
            result['retrieval']['queries'][0]['included_hits'] = 999
            claim_path.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '检索记录'):
                audit_claim_report(claim_path, report_path, data, '合成', input_path, 'analysis')

    def test_two_reports_share_the_same_retrieved_material_prefix(self):
        from report_generation import prepare_final_input, prepare_persona_input
        data = source()
        analysis = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        persona = prepare_persona_input('合成', '', '', '', [], '二手报告', ledger=data, review_questions=QUESTIONS)
        self.assertEqual(analysis.prompt.split('\n## 本次具体任务：')[0],
                         persona.prompt.split('\n## 本次具体任务：')[0])
        self.assertEqual(analysis.retrieval, persona.retrieval)
        self.assertEqual(analysis.context_ids, persona.context_ids)

    def test_saved_prompt_with_retrieval_cannot_silently_drop_its_record(self):
        from claim_ledger import compile_claims
        from claim_fixtures import claim_response
        from report_generation import prepare_final_input
        data = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=QUESTIONS)
        prepared.retrieval = None
        with self.assertRaisesRegex(ValueError, '检索记录'):
            compile_claims(claim_response(), data, prepared, 'analysis')

    def test_planner_cache_reuses_only_same_prompt_and_model_identity(self):
        from claim_retrieval import generate_reviewed_claims
        from claim_fixtures import claim_response
        from report_generation import prepare_final_input
        data = source()
        initial = prepare_final_input('合成', '', '', '', [], ledger=data)
        initial.source_positions = frozenset()
        plan = Mock(return_value=json.dumps({'questions': QUESTIONS}))
        final = Mock(return_value=claim_response())
        def rebuild(questions):
            return prepare_final_input('合成', '', '', '', [], ledger=data, review_questions=questions)
        with tempfile.TemporaryDirectory() as tmp:
            for identity in ('first-model', 'first-model', 'changed-model'):
                generate_reviewed_claims(initial, data, 'analysis', final, rebuild, plan,
                    attempt_dir=Path(tmp), plan_identity=identity)
        self.assertEqual(plan.call_count, 2)
        self.assertEqual(final.call_count, 3)
