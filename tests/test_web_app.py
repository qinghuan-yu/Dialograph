import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from web_app import WebService, validate_chat
from web_worker import run_job, progress_from_line, ProgressStream
import chunk_pipeline as workflow
import analysis_artifacts
import report_generation
import llm_client
from workflow_state import run_identity


def sample_chat():
    return {'session': {'displayName': '测试对象'}, 'messages': [
        {'type': '文本消息', 'content': '测试消息', 'isSend': i % 2,
         'formattedTime': f'2026-05-01 10:0{i}:00', 'createTime': 1777600800 + i * 60}
        for i in range(4)
    ]}


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'talks').mkdir()
        self.service = WebService(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_wrong_export_shape_and_empty_messages(self):
        for data in [[], {'messages': []}, {'messages': [{'text': 'wrong format'}]},
                     {'session': [], 'messages': sample_chat()['messages']}]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_chat(data)

    def test_upload_is_unique_and_cannot_escape_talks(self):
        raw = json.dumps(sample_chat(), ensure_ascii=False).encode('utf-8')
        first = self.service.import_chat('../../private.json', raw)
        second = self.service.import_chat('../../private.json', raw)
        self.assertNotEqual(first['id'], second['id'])
        self.assertTrue(self.service.chat_path(first['id']).is_relative_to(self.root / 'talks'))
        self.assertEqual(len(list((self.root / 'talks').glob('*.json'))), 2)
        with self.assertRaises(ValueError):
            self.service.chat_path('../.env')

    def test_local_preprocessing_uses_real_backend_without_model_request(self):
        job_dir = self.root / 'analysis' / 'web_jobs' / 'test-job'
        job_dir.mkdir(parents=True)
        source = job_dir / 'input.json'
        source.write_text(json.dumps(sample_chat()), encoding='utf-8')
        events = []
        with patch('llm_client.call_llm', side_effect=AssertionError('must remain local')):
            result = run_job(self.root, job_dir, {'mode': 'preprocess'}, events.append)
        self.assertEqual(result['stats']['total_messages'], 4)
        self.assertEqual(result['stats']['session_count'], 1)
        self.assertIn('stats', result['files'])
        self.assertIn('diagnostics', result['files'])
        self.assertNotIn('analysis', result['files'])
        self.assertEqual(events[-1]['stage'], 'complete')

    def test_progress_does_not_mark_inflight_chunk_as_completed(self):
        self.assertEqual(progress_from_line('- 分析分块 2/5'), {'stage': 'chunks', 'current': 2, 'total': 5, 'completed': 1})
        self.assertEqual(progress_from_line('- 复用已完成分块 2/5')['completed'], 2)
        self.assertEqual(progress_from_line('- 综合全部分块，生成最终报告...')['stage'], 'analysis')

    def test_output_access_rejects_paths_and_unknown_keys(self):
        with self.assertRaises(ValueError):
            self.service.output_path('../other', 'analysis')
        with self.assertRaises(ValueError):
            self.service.output_path('0' * 32, '../../.env')

    def test_complete_pipeline_binds_reports_to_real_coverage_and_audit(self):
        job_dir = self.root / 'analysis' / 'web_jobs' / ('a' * 32)
        job_dir.mkdir(parents=True)
        (job_dir / 'input.json').write_text(json.dumps(sample_chat()), encoding='utf-8')
        chunk = json.dumps({key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')})
        responses = [chunk, '# 关系报告\n测试结果。\n<!-- END_FINAL_REPORT -->',
                     '# 人物侧写\n测试结果。\n<!-- END_PERSONA_REPORT -->']
        config = SimpleNamespace(api_base='https://test.invalid', api_key='test-only', model='test-model')
        with patch('web_worker.load_llm_config', return_value=config), patch('llm_client.call_llm', side_effect=responses) as llm:
            result = run_job(self.root, job_dir, {'mode': 'full'}, lambda event: None)
        self.assertEqual(llm.call_count, 3)
        self.assertEqual(set(result['files']), {'analysis', 'persona', 'evidence', 'summary', 'coverage', 'manifest', 'stats', 'diagnostics', 'preprocess'})
        manifest = json.loads((job_dir / result['files']['manifest']['path']).read_text(encoding='utf-8'))
        self.assertEqual(sum(row['message_count'] for row in manifest), 4)
        with patch('web_worker.load_llm_config', return_value=config), patch('llm_client.call_llm', side_effect=AssertionError('must reuse completed reports')):
            resumed = run_job(self.root, job_dir, {'mode': 'full', 'resume': True}, lambda event: None)
        self.assertEqual(result['files'], resumed['files'])

    def test_resume_keeps_shrunken_run_parameters(self):
        job_dir = self.root / 'analysis' / 'web_jobs' / ('b' * 32)
        job_dir.mkdir(parents=True)
        (job_dir / 'input.json').write_text(json.dumps(sample_chat()), encoding='utf-8')
        config = SimpleNamespace(api_key='test-only')
        artifacts = analysis_artifacts.build_artifacts(job_dir / 'input.json', job_dir / 'output')
        run_dir = artifacts['runs_dir'] / '20261017-000001'
        run_dir.mkdir(parents=True)
        skill = report_generation.load_skill_prompt(Path(workflow.__file__).parent / 'SKILL.md')
        (run_dir / 'run_metadata.json').write_text(json.dumps({
            'chunk_size': 660, 'chunk_max_bytes': 12000, 'chunk_max_prompt_bytes': 18000,
            **run_identity(artifacts, skill, config),
        }), encoding='utf-8')
        with patch('web_worker.load_llm_config', return_value=config), patch('chunk_pipeline.run_single_workflow_with_auto_shrink', side_effect=RuntimeError('test-stop')) as run:
            with self.assertRaisesRegex(RuntimeError, 'test-stop'):
                run_job(self.root, job_dir, {'mode': 'full', 'resume': True}, lambda event: None)
        self.assertEqual(run.call_args.kwargs['settings'].chunk_size, 660)
        self.assertEqual(run.call_args.kwargs['settings'].chunk_max_bytes, 12000)
        self.assertEqual(run.call_args.kwargs['settings'].chunk_max_prompt_bytes, 18000)

    def test_completed_reports_regenerate_after_same_length_input_change(self):
        job_dir = self.root / 'job'
        job_dir.mkdir()
        source = job_dir / 'input.json'
        data = sample_chat()
        source.write_text(json.dumps(data), encoding='utf-8')
        chunk = json.dumps({key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')})
        responses = [chunk, 'relation <!-- END_FINAL_REPORT -->', 'persona <!-- END_PERSONA_REPORT -->']
        config = SimpleNamespace(api_base='https://test.invalid', api_key='test-only', model='test-model')
        with patch('web_worker.load_llm_config', return_value=config):
            with patch('llm_client.call_llm', side_effect=responses):
                run_job(self.root, job_dir, {'mode': 'full'}, lambda event: None)
            data['messages'][0]['content'] = '改变消息'
            source.write_text(json.dumps(data), encoding='utf-8')
            with patch('llm_client.call_llm', side_effect=responses) as llm:
                run_job(self.root, job_dir, {'mode': 'full'}, lambda event: None)
                self.assertEqual(llm.call_count, 3)

    def test_persona_prompt_change_regenerates_only_persona(self):
        job_dir = self.root / 'job'
        job_dir.mkdir()
        (job_dir / 'input.json').write_text(json.dumps(sample_chat()), encoding='utf-8')
        chunk = json.dumps({key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')})
        responses = [chunk, 'relation <!-- END_FINAL_REPORT -->', 'persona <!-- END_PERSONA_REPORT -->']
        config = SimpleNamespace(api_base='https://test.invalid', api_key='test-only', model='test-model')
        original_loader = report_generation.load_skill_prompt
        def changed_prompt(path):
            return original_loader(path) + (' changed' if path.name == 'PERSONA_SKILL.md' else '')
        with patch('web_worker.load_llm_config', return_value=config):
            with patch('llm_client.call_llm', side_effect=responses):
                run_job(self.root, job_dir, {'mode': 'full'}, lambda event: None)
            with patch('report_generation.load_skill_prompt', side_effect=changed_prompt), \
                 patch('llm_client.call_llm', return_value='updated <!-- END_PERSONA_REPORT -->') as llm:
                run_job(self.root, job_dir, {'mode': 'full'}, lambda event: None)
                self.assertEqual(llm.call_count, 1)

    def test_one_job_at_a_time_and_cancel_then_resume(self):
        source = self.service.import_chat('test.json', json.dumps(sample_chat()).encode('utf-8'))
        with patch('web_app.threading.Thread.start'):
            first = self.service.start({'chat_id': source['id'], 'mode': 'preprocess'})
            with self.assertRaises(ValueError):
                self.service.start({'chat_id': source['id'], 'mode': 'preprocess'})
            self.assertEqual(self.service.cancel(first['id'])['status'], 'cancelled')
            resumed = self.service.start({'retry_of': first['id']})
            self.assertEqual(resumed['id'], first['id'])
            self.assertEqual(resumed['status'], 'queued')
            request = json.loads((self.service.jobs_dir / first['id'] / 'request.json').read_text(encoding='utf-8'))
            self.assertTrue(request['resume'])

    def test_progress_masks_secret_even_when_print_is_split(self):
        events = []
        stream = ProgressStream(events.append, 'secret-value')
        stream.write('error secret-')
        stream.write('value\n')
        self.assertNotIn('secret-value', events[0]['message'])
        self.assertIn('[已隐藏]', events[0]['message'])


if __name__ == '__main__':
    unittest.main()
