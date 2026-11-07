"""人工确认模式保留来源约束，与模型复核记录和缓存严格区分。"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from claim_fixtures import claim_response
from claim_ledger import generate_claims, render_claim_report, audit_claim_report
from report_generation import prepare_final_input
from test_claim_retrieval import source


class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.ledger = source({78: '我会修改'})
        questions = [{'proposition': '修改承诺', 'support_queries': ['我会修改'],
                      'challenge_queries': ['没有找到的原话']}]
        self.prepared = prepare_final_input('合成', '', '', '', [], ledger=self.ledger, review_questions=questions)
        self.response = claim_response('对方承诺修改。', kind='fact', supporting_evidence=['P_79'],
            counter_review='not_found_in_selected', actions=[{'evidence_id': 'P_79', 'quote': '我会修改',
                'actor': '我', 'target': None, 'state': 'planned', 'confirmation': None}])

    def generate(self, folder, request=None):
        reviewer = Mock(side_effect=AssertionError('人工确认模式不进行模型语义复核'))
        compiled = generate_claims(self.prepared, self.ledger, 'analysis', request or Mock(return_value=self.response),
            attempt_dir=folder, review_request=reviewer, field_review=True, review_mode='human')
        reviewer.assert_not_called()
        return compiled

    def test_human_report_is_explicitly_unconfirmed_and_auditable_without_model_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            compiled = self.generate(folder)
            self.assertEqual(compiled['human_review']['status'], 'pending')
            self.assertNotIn('semantic_review', compiled)
            report = render_claim_report(compiled, '合成')
            self.assertIn('待人工确认', report)
            for path, content in (('claims.json', json.dumps(compiled)), ('report.md', report),
                                  ('input.md', self.prepared.prompt)):
                (folder / path).write_text(content, encoding='utf-8')
            self.assertEqual(audit_claim_report(folder / 'claims.json', folder / 'report.md', self.ledger,
                '合成', folder / 'input.md', 'analysis'), 1)
            for key, value in (('status', 'confirmed'), ('claims_sha256', 'invented')):
                tampered = deepcopy(compiled)
                tampered['human_review'][key] = value
                (folder / 'claims.json').write_text(json.dumps(tampered), encoding='utf-8')
                with self.subTest(key=key), self.assertRaises(ValueError):
                    audit_claim_report(folder / 'claims.json', folder / 'report.md', self.ledger,
                                       '合成', folder / 'input.md', 'analysis')

    def test_human_mode_still_rejects_fabricated_quotes_after_one_structure_repair(self):
        broken = self.response.replace('我会修改', '我已修改')
        request = Mock(return_value=broken)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            self.generate(Path(temporary), request)
        self.assertEqual(request.call_count, 2)

    def test_human_record_cannot_be_used_as_model_review(self):
        from claim_semantic_review import audit_semantic_review
        from claim_field_review import FIELD_REVIEW_MARKER
        with tempfile.TemporaryDirectory() as temporary:
            compiled = self.generate(Path(temporary))
        with self.assertRaises(ValueError):
            audit_semantic_review(compiled, FIELD_REVIEW_MARKER)

    def test_report_review_mode_changes_pipeline_identity(self):
        from config import LLMConfig
        from workflow_state import run_identity
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'input.json'
            path.write_text('{}', encoding='utf-8')
            artifacts = {'chat_file': path, 'messages': []}
            config = LLMConfig('https://example.invalid', 'synthetic', 'example', 'test')
            human = replace(config, report_review_mode='human')
            self.assertNotEqual(run_identity(artifacts, '', config), run_identity(artifacts, '', human))

    def test_web_human_mode_produces_both_reports_with_no_semantic_requests(self):
        from web_worker import run_job
        from test_web_app import sample_chat
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = root / 'job'
            job.mkdir()
            (job / 'input.json').write_text(json.dumps(sample_chat()), encoding='utf-8')
            chunk = json.dumps({key: [] for key in
                ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')})
            config = SimpleNamespace(api_base='https://example.invalid', api_key='synthetic', model='test')
            replies = [chunk, claim_response('关系草稿。'), claim_response('人物草稿。', report_kind='persona')]
            events = []
            with patch('web_worker.load_llm_config', return_value=config), \
                 patch('llm_client.call_llm', side_effect=replies) as request:
                result = run_job(root, job, {'mode': 'full', 'review_mode': 'human'}, events.append)
            self.assertEqual(request.call_count, 3)
            self.assertIn('待人工确认', events[-1]['message'])
            for kind in ('analysis', 'persona'):
                report = (job / result['files'][kind]['path']).read_text(encoding='utf-8')
                self.assertIn('待人工确认', report)
                claims = json.loads((job / result['files'][kind + '_claims']['path']).read_text(encoding='utf-8'))
                self.assertEqual(claims['human_review']['status'], 'pending')

    def test_new_web_jobs_default_to_human_and_validate_review_mode(self):
        from web_app import WebService
        from test_web_app import sample_chat
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = WebService(root)
            row = service.import_chat('synthetic.json', json.dumps(sample_chat()).encode())
            with patch('web_app.load_llm_config', return_value=SimpleNamespace(api_key='synthetic')), \
                 patch('web_app.threading.Thread.start'):
                for invalid in ('invalid', [], None):
                    with self.subTest(mode=invalid), self.assertRaises(ValueError):
                        service.start({'chat_id': row['id'], 'mode': 'full', 'review_mode': invalid})
                state = service.start({'chat_id': row['id'], 'mode': 'full'})
            saved = json.loads((service.jobs_dir / state['id'] / 'request.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['review_mode'], 'human')
            self.assertEqual(state['review_mode'], 'human')
