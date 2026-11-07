"""逐字段出处必须逐字可核对，引用是否登记由程序再次检查。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from claim_field_review import parse_field_checks


def sample():
    data = {'statement_fields': {'text': '对方说会修改，用户发现单位不一致。'},
            'actions': [], 'counter_evidence': [],
            'registered_citations': [{'id': 'P_1', 'quote': '我会修改',
                                      'source': {'message_id': 'M_first'}}],
            '_basis_sources': {'P_1': {'quote': '我会修改', 'message_id': 'M_first'},
                               'P_2': {'quote': '单位不一致', 'message_id': 'M_second'}}}
    checks = {'field_checks': [dict(field='text', content_supported=True, citations_complete=True,
        certainty_preserved=True, outcome_context_complete=True, reason='两句话都有出处。',
        basis=[{'statement': '对方说会修改', 'sources': [{'id': 'P_1', 'quote': '我会修改'}]},
               {'statement': '用户发现单位不一致', 'sources': [{'id': 'P_2', 'quote': '单位不一致'}]}])],
        'action_checks': [], 'counter_checks': []}
    return data, checks


def parse(checks, data):
    return parse_field_checks(json.dumps(checks, ensure_ascii=False), data,
                              schema_version='claim-field-review-v3')


class ClaimReviewBasisTests(unittest.TestCase):
    def test_review_view_indexes_only_visible_sources_without_duplicating_prompt(self):
        from claim_field_review import build_field_review_input
        from claim_fixtures import claim_response
        from claim_ledger import compile_claims
        from report_generation import prepare_final_input
        from test_claim_retrieval import source
        ledger = source({78: '我会修改', 120: '单位不一致'}, session_break=80)
        questions = [{'proposition': '合成命题', 'support_queries': ['我会修改'],
                      'challenge_queries': ['不存在的原话']}]
        prepared = prepare_final_input('合成', '', '', '', [], ledger=ledger, review_questions=questions)
        compiled = compile_claims(claim_response('对方说会修改。', kind='fact',
            supporting_evidence=['P_79'], counter_review='not_found_in_selected'), ledger, prepared, 'analysis')
        prompt, data, _ = build_field_review_input(compiled['claims'][0], prepared, ledger)
        self.assertIn('P_79', data.get('_basis_sources', {}))
        self.assertNotIn('P_121', data['_basis_sources'])
        self.assertNotIn('_basis_sources', prompt)

    def test_generated_basis_uses_stable_ids_and_audit_rechecks_actual_quotes(self):
        from claim_field_review import review_claim_fields, audit_field_review
        from claim_compiler import _evidence_index
        from claim_fixtures import claim_response
        from claim_ledger import compile_claims
        from report_generation import prepare_final_input
        from test_claim_retrieval import source
        from test_claim_field_review import passing_checks
        ledger = source({78: '我会修改'}, session_break=80)
        questions = [{'proposition': '合成命题', 'support_queries': ['我会修改'],
                      'challenge_queries': ['不存在的原话']}]
        prepared = prepare_final_input('合成', '', '', '', [], ledger=ledger, review_questions=questions)
        compiled = compile_claims(claim_response('对方说会修改。', kind='fact',
            supporting_evidence=['P_79'], counter_review='not_found_in_selected'), ledger, prepared, 'analysis')

        def review(prompt):
            data = json.JSONDecoder().raw_decode(prompt.split('\n本条草稿及登记引用\n', 1)[1])[0]
            checks = passing_checks(data)
            for row in checks['field_checks']:
                row['basis'] = []
                if row['field'] == 'text':
                    row['basis'] = [{'statement': '对方说会修改',
                                     'sources': [{'id': 'P_79', 'quote': '我会修改'}]}]
            return json.dumps(checks)

        with tempfile.TemporaryDirectory() as temporary:
            record = review_claim_fields(compiled, prepared, ledger, review, Path(temporary), cache={})
        self.assertEqual(record['schema_version'], 'claim-field-review-v3')
        source_id = record['reviews'][0]['field_checks'][0]['basis'][0]['sources'][0]['id']
        self.assertEqual(source_id, ledger['source_messages'][78]['message_id'])
        audit_field_review(compiled, record, evidence_index=_evidence_index(ledger))
        tampered = deepcopy(record)
        tampered['reviews'][0]['field_checks'][0]['basis'][0]['sources'][0]['quote'] = '我已修改'
        with self.assertRaisesRegex(ValueError, '出处|来源|引语'):
            audit_field_review(compiled, tampered, evidence_index=_evidence_index(ledger))
        for invalid in (None, {}, [None], [{'statement': '对方说会修改', 'sources': None}]):
            malformed = deepcopy(record)
            malformed['reviews'][0]['field_checks'][0]['basis'] = invalid
            with self.subTest(basis=invalid), self.assertRaises(ValueError):
                audit_field_review(compiled, malformed, evidence_index=_evidence_index(ledger))

    def test_visible_but_uncited_fact_blocks_an_all_true_model_review(self):
        data, checks = sample()
        result = parse(checks, data)
        self.assertEqual(result['verdict'], 'revise')
        self.assertIn('P_2', result['failures'][0]['reason'])
        self.assertEqual(result['failures'][0]['field'], 'text')
        self.assertFalse(result['failures'][0]['citations_complete'])
        self.assertEqual(result['checks'], checks, '保留原模型意见，派生错误另行汇总。')

    def test_registered_source_passes_and_unknown_result_needs_no_invented_source(self):
        data, checks = sample()
        data['registered_citations'].append({'id': 'P_2', 'quote': '单位不一致',
                                             'source': {'message_id': 'M_second'}})
        self.assertEqual(parse(checks, data)['verdict'], 'pass')
        data['statement_fields']['text'] = '执行结果仍待确认。'
        checks['field_checks'][0]['basis'] = []
        self.assertEqual(parse(checks, data)['verdict'], 'pass')
        checks['field_checks'][0]['basis'] = [{'statement': '执行结果仍待确认', 'sources': []}]
        self.assertEqual(parse(checks, data)['verdict'], 'pass')

    def test_same_message_alias_requires_the_quote_to_be_inside_registered_excerpt(self):
        data, checks = sample()
        data['registered_citations'].append({'id': 'Q_2', 'quote': '单位不一致',
                                             'source': {'message_id': 'M_second'}})
        self.assertEqual(parse(checks, data)['verdict'], 'pass')
        data['registered_citations'][-1]['quote'] = '单位'
        self.assertEqual(parse(checks, data)['verdict'], 'revise')

    def test_context_only_source_is_checked_but_does_not_add_an_asserted_fact(self):
        data, checks = sample()
        data['statement_fields']['text'] = '对方说会修改。'
        checks['field_checks'][0]['basis'] = [{'statement': '对方说会修改', 'sources': [
            {'id': 'P_1', 'quote': '我会修改', 'usage': 'assertion'},
            {'id': 'P_2', 'quote': '单位不一致', 'usage': 'context'}]}]
        self.assertEqual(parse(checks, data)['verdict'], 'pass')
        checks['field_checks'][0]['basis'][0]['sources'][1]['usage'] = 'assertion'
        self.assertEqual(parse(checks, data)['verdict'], 'revise')
        checks['field_checks'][0]['basis'][0]['sources'][1]['usage'] = 'ignore'
        with self.assertRaises(RuntimeError):
            parse(checks, data)

    def test_wrong_quote_unknown_source_and_fabricated_statement_are_protocol_errors(self):
        data, good = sample()
        variants = []
        wrong_quote = deepcopy(good)
        wrong_quote['field_checks'][0]['basis'][0]['sources'][0]['quote'] = '我已修改'
        variants.append(wrong_quote)
        invisible = deepcopy(good)
        invisible['field_checks'][0]['basis'][0]['sources'][0]['id'] = 'P_900'
        variants.append(invisible)
        invented = deepcopy(good)
        invented['field_checks'][0]['basis'][0]['statement'] = '对方已经修改'
        variants.append(invented)
        omitted = deepcopy(good)
        omitted['field_checks'][0].pop('basis')
        variants.append(omitted)
        for checks in variants:
            with self.subTest(checks=checks), self.assertRaises(RuntimeError):
                parse(checks, data)

    def test_v2_remains_auditable_without_a_basis_but_v3_cannot_downgrade(self):
        data, checks = sample()
        checks['field_checks'][0].pop('basis')
        legacy = parse_field_checks(json.dumps(checks), data, schema_version='claim-field-review-v2')
        self.assertEqual(legacy['verdict'], 'pass')
        with self.assertRaises(RuntimeError):
            parse(checks, data)

    def test_protocol_feedback_reports_field_and_counter_coverage_together(self):
        data, checks = sample()
        checks['field_checks'].append(dict(checks['field_checks'][0], field='invented'))
        checks['counter_checks'] = [{'evidence_id': 'P_2', 'directly_contradicts': True, 'reason': '额外发现。'}]
        with self.assertRaises(RuntimeError) as failure:
            parse(checks, data)
        self.assertIn('field_checks', str(failure.exception))
        self.assertIn('counter_checks', str(failure.exception))
