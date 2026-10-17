"""人格恢复评测的计分边界，避免缺失与极性错误被算作命中。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
import persona_metrics as m


class PersonaMetricsTests(unittest.TestCase):
    def test_planned_failures_remain_in_primary_denominator(self):
        cases = [{'id': 'a', 'profile': 'p', 'gold': {'x': 2}},
                 {'id': 'b', 'profile': 'q', 'gold': {'x': 0}},
                 {'id': 'c', 'profile': 'r', 'gold': {'x': 1}}]
        completed = [{'id': 'a', 'pred': {'x': 2}}, {'id': 'c', 'pred': {'x': 1}}]
        rows = m.planned_rows(cases, completed, recovery_ids={'c'})
        self.assertEqual([r['pred']['x'] for r in rows], [2, None, None])
        self.assertAlmostEqual(m.score([r['gold'] for r in rows], [r['pred'] for r in rows])['recall'], 1 / 3)

    def test_missing_and_wrong_polarity_are_not_true_positives(self):
        scores = m.score([{'a': 2, 'b': 0, 'c': 1}], [{'a': 0, 'b': 0, 'c': None}])
        self.assertAlmostEqual(scores['precision'], .5)
        self.assertAlmostEqual(scores['recall'], 1 / 3)
        self.assertAlmostEqual(scores['micro_f1'], .4)
        self.assertAlmostEqual(scores['coverage'], 2 / 3)
        self.assertAlmostEqual(scores['mae_covered'], .5)
        self.assertAlmostEqual(scores['mae_penalized'], 2 / 3)

    def test_abstention_control_has_no_recovery_score(self):
        scores = m.score([{'a': None, 'b': None}], [{'a': 2, 'b': None}])
        self.assertIsNone(scores['micro_f1'])
        self.assertAlmostEqual(scores['unanswerable_assertion_rate'], .5)

    def test_perfect_varied_predictions_have_unit_agreement(self):
        gold = [{'a': 0, 'b': 2}, {'a': 1, 'b': 0}, {'a': 2, 'b': 1}]
        scores = m.score(gold, gold)
        for key in ('micro_f1', 'label_macro_f1', 'weighted_kappa', 'icc_2_1', 'coupling'):
            self.assertAlmostEqual(scores[key], 1)

    def test_constant_agreement_kappa_is_undefined(self):
        scores = m.score([{'a': 1}], [{'a': 1}])
        self.assertIsNone(scores['weighted_kappa'])

    def test_extraction_requires_exact_source_quote(self):
        with self.assertRaises(ValueError):
            m.validate_extraction({'a': {'value': 2, 'quote': '不存在', 'confidence': .9}}, ['a'], '原文')
        result = m.validate_extraction({'a': {'value': 0, 'quote': '原文', 'confidence': .9}}, ['a'], '这是原文')
        self.assertEqual(result['a']['value'], 0)

    def test_extraction_rejects_out_of_ontology_and_boolean(self):
        for value in (3, True, .5):
            with self.assertRaises(ValueError):
                m.validate_extraction({'a': {'value': value, 'quote': '原文', 'confidence': .9}}, ['a'], '原文')

    def test_cluster_bootstrap_preserves_profile_pairing(self):
        rows = [{'profile': 'a', 'gold': {'x': 2}, 'pred': {'x': 2}},
                {'profile': 'a', 'gold': {'x': 2}, 'pred': {'x': 0}},
                {'profile': 'b', 'gold': {'x': 0}, 'pred': {'x': 0}}]
        intervals = m.bootstrap(rows, repeats=100, seed=7)
        self.assertEqual(intervals['clusters'], 2)
        self.assertLessEqual(intervals['micro_f1'][0], 2 / 3)
        self.assertGreaterEqual(intervals['micro_f1'][1], 2 / 3)


if __name__ == '__main__':
    unittest.main()
