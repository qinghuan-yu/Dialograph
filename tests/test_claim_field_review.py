"""逐字段复核的覆盖、指代与失败门槛，全部使用合成材料。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))


def review_data():
    return {
        'statement_fields': {'text': '对方表示准备问小赵。', 'uncertainty': '执行结果待确认。'},
        'actions': [{'action_index': 1, 'quote': '我会问小赵', 'quote_actor': '我',
                     'quote_target': '小赵', 'state': 'planned'}],
        'counter_evidence': [{'evidence_id': 'P_2'}],
    }


def passing_checks(data):
    return {
        'field_checks': [dict(field=field, content_supported=True, citations_complete=True,
                              certainty_preserved=True, outcome_context_complete=True,
                              reason='合成依据一致。') for field in data['statement_fields']],
        'action_checks': [dict(action_index=number, actor_supported=True, target_supported=True,
                               state_supported=True, reason='原话逐字指代和意向状态一致。')
                          for number in range(1, len(data['actions']) + 1)],
        'counter_checks': [dict(evidence_id=row['evidence_id'], directly_contradicts=True,
                                reason='对固定命题提供相反事实。') for row in data['counter_evidence']],
    }


class ClaimFieldReviewTests(unittest.TestCase):
    def test_different_claims_share_the_complete_selected_source_prefix(self):
        from os.path import commonprefix
        from claim_field_review import build_field_review_input
        from claim_fixtures import claim_response
        from claim_ledger import compile_claims
        from report_generation import prepare_final_input
        from test_claim_retrieval import source
        ledger = source({78: '我会问小赵', 120: '另一场独立交流'}, session_break=80)
        questions = [{'proposition': '合成命题', 'support_queries': ['问小赵'],
                      'challenge_queries': ['独立交流']}]
        prepared = prepare_final_input('合成', '', '', '', [], ledger=ledger, review_questions=questions)
        prompts = []
        for position in (79, 121):
            raw = claim_response('该原话仅作局部观察。', kind='fact', supporting_evidence=[f'P_{position}'],
                                 counter_review='not_found_in_selected')
            compiled = compile_claims(raw, ledger, prepared, 'analysis')
            prompts.append(build_field_review_input(compiled['claims'][0], prepared, ledger)[0])
        prefix = commonprefix(prompts)
        for message in ledger['source_messages']:
            self.assertIn(json.dumps({key: message[key] for key in
                ('position', 'sender', 'time_str', 'type', 'content')}, ensure_ascii=False), prefix)
        self.assertNotIn('本条草稿及登记引用', prefix)

    def test_counter_failure_feedback_restores_the_generators_evidence_id(self):
        from claim_field_review import review_claim_fields
        from claim_semantic_review import SemanticReviewError
        from claim_fixtures import claim_response
        from claim_ledger import compile_claims
        from report_generation import prepare_final_input
        from test_report_context_budget import ledger
        source = ledger()
        support = source['documents'][0]['events'][0]['evidence_id']
        counter = source['documents'][0]['counter_evidence'][0]['evidence_id']
        prepared = prepare_final_input('合成', '', '', support + '\n' + counter, [], ledger=source)
        response = claim_response('对方在本次协作中给予回应。', kind='inference',
            supporting_evidence=[support], counter_review='found',
            counter_evidence=[{'evidence_id': counter, 'target': '给予回应', 'reason': '合成反向证据。'}])
        compiled = compile_claims(response, source, prepared, 'analysis')

        def respond(prompt):
            data = json.loads(prompt.split('\n本条草稿及登记引用\n', 1)[1])
            self.assertEqual(data['counter_evidence'][0]['evidence_id'], 'Q_2')
            checks = passing_checks(data)
            checks['counter_checks'][0].update(directly_contradicts=False, reason='只限制情境，没有相反事实。')
            return json.dumps(checks)

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SemanticReviewError) as failure:
                review_claim_fields(compiled, prepared, source, respond, Path(temporary), cache={})
        self.assertIn('evidence_id=' + counter, str(failure.exception))
        self.assertNotIn('evidence_id=Q_2', str(failure.exception))

    def test_revision_feedback_identifies_the_failed_field_and_check(self):
        from claim_field_review import review_claim_fields
        from claim_semantic_review import SemanticReviewError
        from claim_fixtures import claim_response
        from claim_ledger import compile_claims
        from report_generation import prepare_final_input
        from test_claim_retrieval import source, QUESTIONS
        ledger = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=ledger, review_questions=QUESTIONS)
        compiled = compile_claims(claim_response(), ledger, prepared, 'analysis')

        def respond(prompt):
            data = json.loads(prompt.split('\n本条草稿及登记引用\n', 1)[1])
            checks = passing_checks(data)
            row = next(row for row in checks['field_checks'] if row['field'] == 'context_review')
            row.update(content_supported=False, reason='回应顺序需要修正。')
            return json.dumps(checks)

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SemanticReviewError) as failure:
                review_claim_fields(compiled, prepared, ledger, respond, Path(temporary), cache={})
        self.assertIn('context_review', str(failure.exception))
        self.assertIn('content_supported', str(failure.exception))
        self.assertIn('回应顺序需要修正', str(failure.exception))

    def test_legacy_checks_remain_auditable_but_cannot_satisfy_the_new_input_gate(self):
        from claim_field_review import (FIELD_REVIEW_MARKER, LEGACY_FIELD_REVIEW_MARKER,
                                        LEGACY_FIELD_REVIEW_VERSION, statement_fields)
        from claim_semantic_review import audit_semantic_review
        from claim_fixtures import claim_response
        from claim_ledger import compile_claims
        from evidence_grounding import digest
        from report_generation import prepare_final_input
        from test_claim_retrieval import source, QUESTIONS
        ledger = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=ledger, review_questions=QUESTIONS)
        prepared.prompt += '\n' + LEGACY_FIELD_REVIEW_MARKER
        compiled = compile_claims(claim_response(), ledger, prepared, 'analysis')
        claim = compiled['claims'][0]
        checks = passing_checks({'statement_fields': statement_fields(claim), 'actions': [],
                                 'counter_evidence': []})
        for row in checks['field_checks']:
            row.pop('certainty_preserved')
            row.pop('outcome_context_complete')
        compiled['semantic_review'] = {
            'schema_version': LEGACY_FIELD_REVIEW_VERSION,
            'claims_sha256': digest(compiled['claims']), 'input_sha256': compiled['input_sha256'],
            'reviews': [dict(claim_id=claim['claim_id'], material_sha256='a' * 64, **checks)],
        }
        audit_semantic_review(compiled, prepared.prompt)
        with self.assertRaisesRegex(ValueError, '逐字段'):
            audit_semantic_review(compiled, prepared.prompt.replace(LEGACY_FIELD_REVIEW_MARKER,
                                                                   FIELD_REVIEW_MARKER))

    def test_current_checks_reject_missing_or_non_boolean_strength_and_outcome_flags(self):
        from claim_field_review import parse_field_checks
        data = review_data()
        for flag in ('certainty_preserved', 'outcome_context_complete'):
            for value in (None, 'true', 1):
                checks = passing_checks(data)
                if value is None:
                    checks['field_checks'][0].pop(flag)
                else:
                    checks['field_checks'][0][flag] = value
                with self.subTest(flag=flag, value=value), self.assertRaises(RuntimeError):
                    parse_field_checks(json.dumps(checks), data)

    def test_strength_and_known_outcome_failures_require_revision(self):
        from claim_field_review import parse_field_checks
        data = review_data()
        for flag in ('certainty_preserved', 'outcome_context_complete'):
            checks = passing_checks(data)
            for row in checks['field_checks']:
                row.update(certainty_preserved=True, outcome_context_complete=True)
            checks['field_checks'][0][flag] = False
            with self.subTest(flag=flag):
                result = parse_field_checks(json.dumps(checks), data)
                self.assertEqual(result['verdict'], 'revise')
                self.assertEqual(len(result['failures']), 1)

    def test_generation_revises_only_failed_claims_and_audits_the_field_gate(self):
        from claim_field_review import FIELD_REVIEW_MARKER, FIELD_REVIEW_VERSION
        from claim_fixtures import claim_response, claim_revision
        from claim_ledger import generate_claims, audit_claim_report, render_claim_report
        from report_generation import prepare_final_input
        from test_claim_retrieval import source, QUESTIONS
        ledger = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=ledger, review_questions=QUESTIONS)
        first = json.loads(claim_response().split('\n<!--')[0])['claims'][0]
        second = dict(first, text='该判断仍待确认。')
        generate = Mock(side_effect=[json.dumps({'claims': [first, second]}),
                                    claim_revision('已缩小的判断仍待确认。', claim_number=2)])
        def respond(prompt):
            data = json.loads(prompt.split('\n本条草稿及登记引用\n', 1)[1])
            checks = passing_checks(data)
            if data['statement_fields']['text'] == second['text']:
                checks['field_checks'][0]['content_supported'] = False
                checks['field_checks'][0]['reason'] = '须缩小该合成判断。'
            return json.dumps(checks)
        reviewer = Mock(side_effect=respond)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = generate_claims(prepared, ledger, 'analysis', generate, attempt_dir=root,
                                     review_request=reviewer, field_review=True)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(reviewer.call_count, 3)
            self.assertEqual(result['claims'][0]['text'], first['text'])
            self.assertIn(FIELD_REVIEW_MARKER, prepared.prompt)
            self.assertEqual(result['semantic_review']['schema_version'], FIELD_REVIEW_VERSION)
            (root / 'input.md').write_text(prepared.prompt, encoding='utf-8')
            (root / 'report.md').write_text(render_claim_report(result, '合成'), encoding='utf-8')
            (root / 'claims.json').write_text(json.dumps(result), encoding='utf-8')
            self.assertEqual(audit_claim_report(root / 'claims.json', root / 'report.md', ledger, '合成',
                                              root / 'input.md', 'analysis'), 2)
            result['semantic_review']['schema_version'] = 'claim-semantic-review-v1'
            (root / 'claims.json').write_text(json.dumps(result), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '逐字段'):
                audit_claim_report(root / 'claims.json', root / 'report.md', ledger, '合成',
                                   root / 'input.md', 'analysis')

    def test_review_reuses_identical_material_and_rechecks_changed_text_or_context(self):
        from claim_field_review import review_claim_fields, audit_field_review
        from claim_fixtures import claim_response
        from claim_ledger import compile_claims
        from report_generation import prepare_final_input
        from test_claim_retrieval import source, QUESTIONS
        ledger = source()
        prepared = prepare_final_input('合成', '统计甲', '覆盖甲', '', [], ledger=ledger,
                                       review_questions=QUESTIONS)
        compiled = compile_claims(claim_response(), ledger, prepared, 'analysis')
        def respond(prompt):
            data = json.loads(prompt.split('\n本条草稿及登记引用\n', 1)[1])
            return json.dumps(passing_checks(data))
        request = Mock(side_effect=respond)
        cache = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = review_claim_fields(compiled, prepared, ledger, request, root / 'first', cache=cache)
            audit_field_review(compiled, record)
            repeated = review_claim_fields(compiled, prepared, ledger, request, root / 'repeat', cache=cache)
            self.assertEqual(record, repeated)
            self.assertEqual(request.call_count, 1)
            self.assertTrue((root / 'repeat' / 'claim_01' / 'cache_origin.json').exists())
            changed = compile_claims(claim_response('该材料的判断仍待验证。'), ledger, prepared, 'analysis')
            review_claim_fields(changed, prepared, ledger, request, root / 'changed', cache=cache)
            self.assertEqual(request.call_count, 2)
            prepared.prompt = prepared.prompt.replace('统计甲', '统计乙')
            review_claim_fields(changed, prepared, ledger, request, root / 'context', cache=cache)
            self.assertEqual(request.call_count, 3)
        tampered = copy.deepcopy(record)
        tampered['reviews'][0]['field_checks'].pop()
        with self.assertRaises(ValueError):
            audit_field_review(compiled, tampered)

    def test_failed_fields_block_publication_and_invalid_protocol_has_one_repair(self):
        from claim_field_review import review_claim_fields
        from claim_semantic_review import SemanticReviewError
        from claim_fixtures import claim_response
        from claim_ledger import compile_claims
        from report_generation import prepare_final_input
        from test_claim_retrieval import source, QUESTIONS
        ledger = source()
        prepared = prepare_final_input('合成', '', '', '', [], ledger=ledger, review_questions=QUESTIONS)
        compiled = compile_claims(claim_response(), ledger, prepared, 'analysis')
        data = {'statement_fields': dict(text='', context_review='', uncertainty=''),
                'actions': [], 'counter_evidence': []}
        checks = passing_checks(data)
        checks['field_checks'][0]['citations_complete'] = False
        checks['field_checks'][0]['reason'] = '追问原话没有登记为本条引用。'
        request = Mock(side_effect=['{}', json.dumps(checks)])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(SemanticReviewError, '追问原话') as failure:
                review_claim_fields(compiled, prepared, ledger, request, root, cache={})
            self.assertEqual(failure.exception.claim_numbers, {1})
            self.assertTrue((root / 'claim_01' / 'repair_response.txt').exists())
            self.assertEqual(request.call_count, 2)
            invalid = Mock(return_value='{}')
            with self.assertRaisesRegex(RuntimeError, '逐字段复核'):
                review_claim_fields(compiled, prepared, ledger, invalid, root / 'invalid', cache={})
            self.assertEqual(invalid.call_count, 2)

    def test_review_view_preserves_literal_pronouns_and_all_selected_sessions(self):
        from claim_field_review import build_field_review_input
        from claim_ledger import compile_claims
        from claim_fixtures import claim_response
        from report_generation import prepare_final_input
        from test_claim_retrieval import source
        ledger = source({78: '我会问小赵', 120: '另一场独立交流'}, session_break=80)
        questions = [{'proposition': '合成命题', 'support_queries': ['问小赵'],
                      'challenge_queries': ['问小赵']},
                     {'proposition': '另一命题', 'support_queries': ['独立交流'],
                      'challenge_queries': ['独立交流']}]
        prepared = prepare_final_input('合成', '全局统计', '完整抽取覆盖', '', [],
                                       ledger=ledger, review_questions=questions)
        raw = claim_response('对方表示准备问小赵。', kind='fact', subject='peer',
            supporting_evidence=['P_79'], counter_review='not_found_in_selected',
            actions=[dict(evidence_id='P_79', quote='我会问小赵', actor='我', target='小赵',
                          state='planned', confirmation=None)])
        compiled = compile_claims(raw, ledger, prepared, 'analysis')
        before = copy.deepcopy(compiled)
        prompt, data, aliases = build_field_review_input(compiled['claims'][0], prepared, ledger)
        self.assertIn('全局统计', prompt)
        self.assertIn('完整抽取覆盖', prompt)
        self.assertEqual(data['counter_review'], 'not_found_in_selected')
        self.assertEqual(data['actions'][0]['quote_actor'], '我')
        self.assertEqual(data['actions'][0]['quote_target'], '小赵')
        self.assertNotIn('actor', data['actions'][0])
        self.assertEqual(data['registered_citations'][0]['source']['sender'], 'other')
        self.assertEqual(data['registered_citations'][0]['id'], 'P_79')
        self.assertEqual(list(aliases.values()), ['P_79'])
        context = json.loads(prompt.split('复核原文材料\n', 1)[1].split('\n本条定位及同文邻文\n', 1)[0])
        focus = json.loads(prompt.split('\n本条定位及同文邻文\n', 1)[1].split('\n本条草稿及登记引用\n', 1)[0])
        self.assertEqual([row['position'] for row in context['messages']], list(range(1, 141)))
        self.assertEqual(focus['citation_session_positions'], list(range(1, 81)))
        self.assertEqual(context['omitted_selected_messages'], 0)
        self.assertEqual(compiled, before)

    def test_repetition_supplement_includes_real_neighbors_and_explicit_omissions(self):
        from claim_field_review import repetition_context
        from test_claim_retrieval import source
        repeated = '这是独立编写的重复长文，用于检查复用文字的邻文与覆盖范围。' * 4
        ledger = source({8: repeated, 100: repeated}, session_break=80)
        citations = [{'quote': repeated}]
        context = repetition_context(ledger['source_messages'], citations)
        self.assertEqual(context['matching_positions'], [9, 101])
        self.assertEqual(context['omitted_occurrences'], 0)
        positions = {row['position'] for row in context['messages']}
        self.assertTrue({8, 9, 10, 100, 101, 102} <= positions)
        limited = repetition_context(ledger['source_messages'], citations, max_bytes=1)
        self.assertEqual(limited['messages'], [])
        self.assertEqual(limited['omitted_occurrences'], 2)

    def test_any_failed_dimension_requires_revision_without_a_model_summary_vote(self):
        from claim_field_review import parse_field_checks
        data = review_data()
        for group, flag in [('field_checks', 'content_supported'), ('field_checks', 'citations_complete'),
                            ('action_checks', 'actor_supported'), ('action_checks', 'target_supported'),
                            ('action_checks', 'state_supported'), ('counter_checks', 'directly_contradicts')]:
            checks = passing_checks(data)
            checks[group][0][flag] = False
            with self.subTest(group=group, flag=flag):
                result = parse_field_checks(json.dumps(checks), data)
                self.assertEqual(result['verdict'], 'revise')
                self.assertEqual(len(result['failures']), 1)
        self.assertEqual(parse_field_checks(json.dumps(passing_checks(data)), data)['verdict'], 'pass')

    def test_missing_duplicate_unknown_or_mistyped_checks_are_rejected(self):
        from claim_field_review import parse_field_checks
        data = review_data()
        invalid = []
        for group in ('field_checks', 'action_checks', 'counter_checks'):
            missing = passing_checks(data)
            missing[group].pop()
            invalid.append(missing)
            duplicate = passing_checks(data)
            duplicate[group].append(copy.deepcopy(duplicate[group][0]))
            invalid.append(duplicate)
        unknown = passing_checks(data)
        unknown['field_checks'][0]['field'] = 'invented'
        invalid.append(unknown)
        boolean_index = passing_checks(data)
        boolean_index['action_checks'][0]['action_index'] = True
        invalid.append(boolean_index)
        string_flag = passing_checks(data)
        string_flag['field_checks'][0]['content_supported'] = 'true'
        invalid.append(string_flag)
        missing_reason = passing_checks(data)
        missing_reason['action_checks'][0]['reason'] = ' '
        invalid.append(missing_reason)
        for checks in invalid:
            with self.subTest(checks=checks), self.assertRaisesRegex(RuntimeError, '逐字段复核'):
                parse_field_checks(json.dumps(checks), data)

    def test_protocol_rejects_extra_verdict_and_non_object_json(self):
        from claim_field_review import parse_field_checks
        data = review_data()
        for response in ('[]', 'null', 'broken', json.dumps(dict(passing_checks(data), verdict='pass'))):
            with self.subTest(response=response), self.assertRaisesRegex(RuntimeError, '逐字段复核'):
                parse_field_checks(response, data)


if __name__ == '__main__':
    unittest.main()
