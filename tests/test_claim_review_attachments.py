"""可见附件标签仅作话题背景，不能成为附件内容或动作结果的证据。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from claim_field_review import (FIELD_REVIEW_MARKER, build_field_review_input,
                                parse_field_checks, review_claim_fields)
from claim_fixtures import claim_response
from claim_ledger import compile_claims, audit_claim_report, render_claim_report
from report_generation import prepare_final_input
from test_claim_field_review import passing_checks
from test_claim_retrieval import source


def fixture():
    ledger = source({77: '[文件] 练习.md', 78: '我会修改', 125: '[图片]'},
                    type_overrides={77: '文件消息', 125: '图片消息'}, session_break=80)
    questions = [{'proposition': '合成命题', 'support_queries': ['我会修改'],
                  'challenge_queries': ['不存在的原话']}]
    prepared = prepare_final_input('合成', '', '', '', [], ledger=ledger, review_questions=questions)
    prepared.prompt += '\n' + FIELD_REVIEW_MARKER
    response = claim_response('对方在文件讨论中表示会修改。', kind='fact',
                              supporting_evidence=['P_79'], counter_review='not_found_in_selected')
    compiled = compile_claims(response, ledger, prepared, 'analysis')
    return ledger, prepared, compiled


def response(data):
    checks = passing_checks(data)
    checks['field_checks'][0]['basis'] = [{
        'statement': '对方在文件讨论中表示会修改', 'sources': [
            {'id': 'P_79', 'quote': '我会修改', 'usage': 'assertion'},
            {'id': 'P_78', 'quote': '[文件] 练习.md', 'usage': 'context'}]}]
    return checks


class ClaimReviewAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.ledger, self.prepared, self.compiled = fixture()
        self.prompt, self.data, _ = build_field_review_input(
            self.compiled['claims'][0], self.prepared, self.ledger)

    def test_visible_attachment_background_does_not_trigger_a_protocol_retry(self):
        self.assertIn('[文件] 练习.md', self.prompt)
        self.assertNotIn('P_126', self.data['_basis_sources'])
        result = parse_field_checks(json.dumps(response(self.data)), self.data)
        self.assertEqual(result['verdict'], 'pass')

    def test_attachment_contents_and_unselected_sources_still_fail(self):
        for change in ({'quote': '练习已完成'}, {'id': 'P_126', 'quote': '[图片]'}):
            checks = response(self.data)
            checks['field_checks'][0]['basis'][0]['sources'][1].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, '来源|引语'):
                parse_field_checks(json.dumps(checks), self.data)

    def test_attachment_assertion_is_not_registered_and_preserves_semantic_failure(self):
        checks = response(self.data)
        checks['field_checks'][0]['basis'][0]['sources'][1]['usage'] = 'assertion'
        checks['field_checks'][0].update(content_supported=False, reason='文件内容并未解析。')
        result = parse_field_checks(json.dumps(checks), self.data)
        self.assertEqual(result['verdict'], 'revise')
        self.assertEqual(result['checks'], checks)
        self.assertTrue(any(not row['citations_complete'] for row in result['failures']))
        with self.assertRaises(ValueError):
            compile_claims(claim_response('对方完成了练习。', kind='fact',
                supporting_evidence=['P_78'], counter_review='not_found_in_selected'),
                self.ledger, self.prepared, 'analysis')

    def test_saved_attachment_background_is_rechecked_by_report_audit(self):
        calls = []

        def review(prompt):
            calls.append(prompt)
            data, _ = json.JSONDecoder().raw_decode(prompt.split('\n本条草稿及登记引用\n', 1)[1])
            return json.dumps(response(data))

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.compiled['semantic_review'] = review_claim_fields(
                self.compiled, self.prepared, self.ledger, review, directory, cache={})
            self.assertEqual(len(calls), 1)
            source = self.compiled['semantic_review']['reviews'][0]['field_checks'][0]['basis'][0]['sources'][1]
            self.assertEqual(source['id'], self.ledger['source_messages'][77]['message_id'])
            input_path, claim_path, report_path = [directory / name for name in ('input.md', 'claims.json', 'report.md')]
            input_path.write_text(self.prepared.prompt, encoding='utf-8')
            report_path.write_text(render_claim_report(self.compiled, '合成'), encoding='utf-8')
            claim_path.write_text(json.dumps(self.compiled), encoding='utf-8')
            self.assertEqual(audit_claim_report(claim_path, report_path, self.ledger,
                                               '合成', input_path, 'analysis'), 1)
            for change in ({'quote': '练习已完成'}, {'usage': 'assertion'}, {'id': 'M_unknown'}):
                changed = deepcopy(self.compiled)
                changed['semantic_review']['reviews'][0]['field_checks'][0]['basis'][0]['sources'][1].update(change)
                claim_path.write_text(json.dumps(changed), encoding='utf-8')
                with self.subTest(change=change), self.assertRaises(ValueError):
                    audit_claim_report(claim_path, report_path, self.ledger, '合成', input_path, 'analysis')
