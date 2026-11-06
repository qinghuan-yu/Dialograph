"""固定命题的离线人工标注验收；不是模型自动语义裁判。"""
import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from test_report_context_budget import ledger
from claim_fixtures import claim_response
from claim_ledger import ReportInput, compile_claims, render_claim_report

CASES = Path(__file__).parent / 'fixtures/claim_semantic_cases.json'


class EvidenceRoleTests(unittest.TestCase):
    def setUp(self):
        self.data = ledger()
        self.support = self.data['documents'][0]['events'][0]['evidence_id']
        self.counter = self.data['documents'][0]['counter_evidence'][0]['evidence_id']
        self.prepared = ReportInput(self.support + ' ' + self.counter,
                                   {self.support, self.counter}, {self.support, self.counter})

    def build(self, **changes):
        options = dict(text='对方在本次合作中愿意回应。', kind='inference', scope='local',
                       supporting_evidence=[self.support], counter_review='not_found_in_selected',
                       counter_evidence=[], limiting_evidence=[])
        options.update(changes)
        return compile_claims(claim_response(**options), self.data, self.prepared, 'analysis')

    def test_limit_is_kept_separately_when_no_direct_counter_was_found(self):
        result = self.build(limiting_evidence=[{'evidence_id': self.counter, 'reason': '这条材料只限制可约见的时间。'}])
        item = result['claims'][0]
        self.assertEqual(item['counter_evidence'], [])
        self.assertEqual(item['counter_review'], 'not_found_in_selected')
        self.assertEqual(item['limiting_evidence'][0]['evidence_id'], self.counter)
        report = render_claim_report(result, '合成')
        self.assertIn('情境限制', report)
        self.assertIn('可约见的时间', report)
        self.assertNotIn('反证：' + self.counter, report)

    def test_counter_requires_target_within_current_claim_and_a_reason(self):
        result = self.build(counter_review='found', counter_evidence=[{
            'evidence_id': self.counter, 'target': '愿意回应', 'reason': '这段原文明确表达了不愿回应。'}])
        self.assertEqual(result['claims'][0]['counter_evidence'][0]['target'], '愿意回应')
        self.assertIn('削弱“愿意回应”', render_claim_report(result, '合成'))
        for entry in ({'evidence_id': self.counter, 'target': '不存在的判断', 'reason': '说明'},
                      {'evidence_id': self.counter, 'target': '愿意回应', 'reason': ''},
                      {'evidence_id': self.counter, 'reason': '缺少目标'}):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                self.build(counter_review='found', counter_evidence=[entry])

    def test_same_reference_cannot_be_both_counter_and_limitation(self):
        with self.assertRaisesRegex(ValueError, '重复|重叠'):
            self.build(counter_review='found', counter_evidence=[{
                'evidence_id': self.counter, 'target': '愿意回应', 'reason': '反向材料'}],
                limiting_evidence=[{'evidence_id': self.counter, 'reason': '只限制范围'}])

    def test_unexplained_legacy_counter_ids_are_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, '反证'):
            self.build(counter_review='found', counter_evidence=[self.counter])

    def test_limitation_keeps_source_and_context_gates(self):
        with self.assertRaisesRegex(ValueError, '入选'):
            self.build(limiting_evidence=[{'evidence_id': 'E_' + 'f' * 32, 'reason': '无来源'}])
        self.prepared.context_ids.remove(self.counter)
        with self.assertRaisesRegex(ValueError, '上下文'):
            self.build(limiting_evidence=[{'evidence_id': self.counter, 'reason': '缺少语境'}])

    def test_limitations_do_not_inflate_support_scope(self):
        with self.assertRaisesRegex(ValueError, '范围'):
            self.build(scope='long_term', limiting_evidence=[{'evidence_id': self.counter, 'reason': '限制'}])

    def test_audit_preserves_new_roles_and_never_upgrades_v1_in_place(self):
        from claim_ledger import audit_claim_report
        import report_generation
        self.prepared = report_generation.prepare_final_input('合成', '', '',
            '### 事件\n- ' + self.support + '\n### 反证\n- ' + self.counter, [], ledger=self.data)
        result = self.build(limiting_evidence=[{'evidence_id': self.counter, 'reason': '仅限制适用场景。'}])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp, rp, ip = (root / name for name in ('claims.json', 'report.md', 'input.md'))
            cp.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            rp.write_text(render_claim_report(result, '合成'), encoding='utf-8')
            ip.write_text(self.prepared.prompt, encoding='utf-8')
            self.assertEqual(audit_claim_report(cp, rp, self.data, '合成', ip, 'analysis'), 1)
            result['schema_version'] = 'chat-claims-v1'
            cp.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            before = [p.read_bytes() for p in (cp, rp, ip)]
            with self.assertRaisesRegex(ValueError, '旧版'):
                audit_claim_report(cp, rp, self.data, '合成', ip, 'analysis')
            self.assertEqual(before, [p.read_bytes() for p in (cp, rp, ip)])


class OfflineAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.cases = json.loads(CASES.read_text(encoding='utf-8'))['cases']
        self.predictions = [{'case_id': case['id'], 'proposition': case['proposition'],
                             **copy.deepcopy(case['expected']),
                             'available_positions': [row['position'] for row in case['messages']]}
                            for case in self.cases]

    def test_gold_annotations_and_complete_materials_pass(self):
        from evaluate_claim_semantics import evaluate_cases
        result = evaluate_cases(self.cases, self.predictions)
        self.assertEqual(result['cases'], 13)
        self.assertEqual(result['passed'], 13)

    def test_cancelled_meeting_is_not_silently_counted_as_counter_to_care(self):
        from evaluate_claim_semantics import evaluate_cases
        self.predictions[0]['roles']['3'] = 'counter'
        result = evaluate_cases(self.cases, self.predictions)
        self.assertEqual(result['passed'], 12)
        self.assertIn('roles', result['results'][0]['failed_checks'])

    def test_missing_later_response_fails_coverage_even_with_correct_roles(self):
        from evaluate_claim_semantics import evaluate_cases
        case = next(row for row in self.predictions if row['case_id'] == 'brief_start_followed_by_support')
        case['available_positions'] = [1, 2, 3]
        result = evaluate_cases(self.cases, self.predictions)
        row = next(row for row in result['results'] if row['case_id'] == case['case_id'])
        self.assertEqual(row['missing_positions'], [4, 5])
        self.assertIn('material_coverage', row['failed_checks'])

    def test_cannot_change_proposition_or_cherry_pick_cases(self):
        from evaluate_claim_semantics import evaluate_cases
        self.predictions[0]['proposition'] = '对方总能见面。'
        result = evaluate_cases(self.cases, self.predictions[:-1])
        self.assertIn('proposition', result['results'][0]['failed_checks'])
        self.assertIn('missing_prediction', result['results'][-1]['failed_checks'])

    def test_cli_returns_failure_and_writes_reviewable_mismatch(self):
        self.predictions[0]['roles']['3'] = 'counter'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions, output = root / 'predictions.json', root / 'result.json'
            predictions.write_text(json.dumps({'schema_version': 'claim-semantic-predictions-v1',
                                              'predictions': self.predictions}), encoding='utf-8')
            process = subprocess.run([sys.executable, '-X', 'utf8',
                str(Path(__file__).resolve().parents[1] / 'agent/evaluate_claim_semantics.py'),
                '--cases', str(CASES), '--predictions', str(predictions), '--output', str(output)],
                capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(process.returncode, 1, process.stderr)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result['passed'], 12)
            self.assertEqual(result['results'][0]['failed_checks'], ['roles'])


if __name__ == '__main__':
    unittest.main()
