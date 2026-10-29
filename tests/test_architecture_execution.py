"""架构边界回归：只使用合成输入和请求替身。"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
import chunk_pipeline as workflow
import analysis_artifacts
import report_generation
import llm_client
import web_app
import web_worker
import prompt_budget

CONFIG = SimpleNamespace(api_base='https://example.invalid', api_key='synthetic', model='test')
KEYS = ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')


def synthetic_summaries():
    documents = []
    for index in range(1, 31):
        item = {'evidence': f'原文{index}：' + '虚构内容用于测试。' * 15, 'confidence': 'medium'}
        documents.append({'chunk_index': index, 'parse_status': 'ok',
            'events': [dict(item, summary=f'事件{index}')],
            'relation_signals': [dict(item, model='模型', direction='mixed', signal=f'关系{index}')],
            'persona_signals': [dict(item, trait='维度', stability='local', signal=f'人物{index}')],
            'counter_evidence': [dict(item, against=f'反证{index}')],
            'uncertainties': [{'question': f'疑问{index}', 'reason': '合成未知信息'}]})
    return workflow.format_evidence_summary(documents), [workflow.format_evidence_summary([doc]) for doc in documents]


class BudgetTests(unittest.TestCase):
    def test_final_context_retains_each_category_and_late_evidence(self):
        evidence, phases = synthetic_summaries()
        prompt = report_generation.build_final_prompt('合成', 'stats', 'coverage', evidence, phases)
        self.assertIn('反证30', prompt)
        self.assertIn('疑问30', prompt)
        self.assertIn('chunk_030', prompt)
        self.assertIn('关系30', prompt)
        self.assertIn('人物30', prompt)
        self.assertLessEqual(len(prompt.encode('utf-8')), 30000)
        self.assertIn('省略', prompt)

    def test_persona_uses_same_balancing_and_bounded_context(self):
        evidence, phases = synthetic_summaries()
        prompt = report_generation.build_persona_prompt('合成', '统计' * 20000, '覆盖' * 20000,
            evidence, phases, '关系报告' * 30000)
        self.assertLessEqual(len(prompt.encode('utf-8')), 40000)
        self.assertIn('反证30', prompt)
        self.assertIn('疑问30', prompt)
        self.assertIn('省略', prompt)

    def test_reloaded_phase_file_preserves_budget_and_time_axis(self):
        evidence, phases = synthetic_summaries()
        # 续跑和侧写从一个阶段文件加载，不能把重复标题误当作新类别。
        selected = prompt_budget.budget_phase_summaries(['\n\n'.join(phases)], 12000)
        self.assertLessEqual(len(selected.encode('utf-8')), 12000)
        prompt = report_generation.build_persona_prompt('合成', '统计' * 20000, '覆盖' * 20000,
            evidence, ['\n\n'.join(phases)], '关系报告' * 30000)
        self.assertLessEqual(len(prompt.encode('utf-8')), 40000)
        self.assertIn('反证30', prompt)
        self.assertIn('chunk_030', prompt)


class RequestRecoveryTests(unittest.TestCase):
    def test_bad_request_and_rate_limit_do_not_request_smaller_chunks(self):
        for code in (400, 401, 429, 500, 503):
            with self.subTest(code=code):
                self.assertFalse(workflow.should_retry_with_smaller_prompt(
                    HTTPError(CONFIG.api_base, code, 'synthetic', {}, None)))

    def test_bad_request_fails_without_repeating_the_request(self):
        with patch.object(llm_client, 'call_llm', side_effect=HTTPError(CONFIG.api_base, 400, 'bad', {}, None)) as api, \
             patch.object(workflow.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(HTTPError):
                llm_client.call_llm_with_retry_result(CONFIG, 's', 'u', 1, 100)
        self.assertEqual(api.call_count, 1)

    def test_rate_limit_honors_retry_after_on_the_same_request(self):
        error = HTTPError(CONFIG.api_base, 429, 'rate', {'Retry-After': '7'}, None)
        with patch.object(llm_client, 'call_llm', side_effect=[error, 'ok']) as api, \
             patch.object(workflow.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            result = llm_client.call_llm_with_retry_result(CONFIG, 's', 'u', 1, 100)
        self.assertEqual(result.content, 'ok')
        self.assertEqual(api.call_args_list[0], api.call_args_list[1])
        sleep.assert_called_once_with(7)

    def test_reasoning_only_response_is_not_a_report(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '', 'reasoning_content': 'private reasoning'}}]}
        with patch.object(llm_client.request, 'urlopen') as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            with self.assertRaisesRegex(ValueError, '正文|可用内容'):
                llm_client.call_llm(CONFIG.api_base, 'fake', 'test', 's', 'u', 1)

    def test_persistent_rate_limit_preserves_completed_chunk_without_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.json'
            source.write_bytes((Path(__file__).parent / 'fixtures' / 'web_smoke.json').read_bytes())
            artifacts = analysis_artifacts.build_artifacts(source, root / 'out')
            response = json.dumps({key: [] for key in KEYS})
            error = HTTPError(CONFIG.api_base, 429, 'rate', {}, None)
            with patch.object(llm_client, 'call_llm', side_effect=[response] + [error] * 20) as api, \
                 patch.object(workflow.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(HTTPError):
                    workflow.run_single_workflow_with_auto_shrink(artifacts, 'skill', CONFIG, 1,
                        chunk_size=1, chunk_sleep=0)
            runs = list(artifacts['runs_dir'].iterdir())
            self.assertEqual(len(runs), 1)
            self.assertEqual(api.call_count, 4)
            self.assertTrue((runs[0] / '结构化证据' / 'evidence_001.json').is_file())


class ReportLifecycleTests(unittest.TestCase):
    def prepare(self, root):
        source = root / 'input.json'
        source.write_bytes((Path(__file__).parent / 'fixtures' / 'web_smoke.json').read_bytes())
        artifacts = analysis_artifacts.build_artifacts(source, root / 'out')
        with patch.object(llm_client, 'call_llm', side_effect=[json.dumps({key: [] for key in KEYS}),
                'relation <!-- END_FINAL_REPORT -->', 'old persona <!-- END_PERSONA_REPORT -->']), \
             contextlib.redirect_stdout(io.StringIO()):
            workflow.run_single_workflow(artifacts, 'skill', CONFIG, 1, chunk_sleep=0)
            report_generation.generate_persona_report(artifacts, 'skill', CONFIG, 1)
        return artifacts

    def test_persona_publish_failure_restores_report_and_metadata(self):
        import os
        real_replace = os.replace
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.prepare(Path(tmp))
            paths = [artifacts['persona_file'], artifacts['persona_meta_file']]
            old = [path.read_bytes() for path in paths]
            for failure_step in range(1, 5):
                count = 0
                def fail_once(src, dst):
                    nonlocal count
                    count += 1
                    if count == failure_step:
                        raise OSError('synthetic publish failure')
                    return real_replace(src, dst)
                with self.subTest(step=failure_step), patch('os.replace', side_effect=fail_once), \
                     patch.object(llm_client, 'call_llm', return_value='new persona <!-- END_PERSONA_REPORT -->'), \
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(OSError):
                        report_generation.generate_persona_report(artifacts, 'skill', CONFIG, 1)
                    self.assertEqual([path.read_bytes() for path in paths], old)

    def test_persona_failed_audit_cannot_replace_published_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.prepare(Path(tmp))
            paths = [artifacts['persona_file'], artifacts['persona_meta_file']]
            old = [path.read_bytes() for path in paths]
            audit = SimpleNamespace(ok=False, errors=['synthetic audit failure'])
            with patch.object(report_generation, 'audit_artifacts', return_value=audit), \
                 patch.object(llm_client, 'call_llm', return_value='new persona <!-- END_PERSONA_REPORT -->'), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, '审计'):
                    report_generation.generate_persona_report(artifacts, 'skill', CONFIG, 1)
            self.assertEqual([path.read_bytes() for path in paths], old)

    def test_incompatible_automatic_web_resume_warns_and_starts_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'input.json').write_bytes((Path(__file__).parent / 'fixtures' / 'web_smoke.json').read_bytes())
            artifacts = analysis_artifacts.build_artifacts(root / 'input.json', root / 'output')
            run = artifacts['runs_dir'] / 'old-run'
            run.mkdir(parents=True)
            (run / 'run_metadata.json').write_text('{}', encoding='utf-8')
            events = []
            with patch.object(web_worker, 'load_llm_config', return_value=CONFIG), \
                 patch.object(workflow, 'run_single_workflow_with_auto_shrink', side_effect=RuntimeError('new-run')) as dispatch:
                with self.assertRaisesRegex(RuntimeError, 'new-run'):
                    web_worker.run_job(root, root, {'mode': 'full', 'resume': True}, events.append)
            self.assertIsNone(dispatch.call_args.kwargs['resume_run_id'])
            self.assertTrue(any(event['type'] == 'warning' for event in events))

    def test_malformed_worker_event_releases_job_and_process(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            service = web_app.WebService(Path(tmp))
            job_id = 'a' * 32
            (service.jobs_dir / job_id).mkdir()
            service.jobs[job_id] = {'id': job_id, 'status': 'queued', 'events': []}
            process = MagicMock()
            process.stdout = io.StringIO('{"type":"result"}\n')
            process.stderr = io.StringIO('')
            process.wait.return_value = 0
            with patch.object(web_app.subprocess, 'Popen', return_value=process):
                service.run(job_id)
            self.assertEqual(service.jobs[job_id]['status'], 'failed')
            self.assertEqual(service.processes, {})
            self.assertIn('finished', service.jobs[job_id])


class CancellationRecoveryTests(unittest.TestCase):
    def test_stream_failure_does_not_overwrite_a_user_cancellation(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            service = web_app.WebService(Path(tmp))
            job_id = 'b' * 32
            (service.jobs_dir / job_id).mkdir()
            service.jobs[job_id] = {'id': job_id, 'status': 'queued', 'events': []}
            class BrokenStream(io.StringIO):
                def __next__(self):
                    service.jobs[job_id]['status'] = 'cancelled'
                    raise UnicodeError('synthetic stream failure')
            process = MagicMock()
            process.stdout, process.stderr = BrokenStream(), io.StringIO('')
            process.wait.return_value = 0
            with patch.object(web_app.subprocess, 'Popen', return_value=process):
                service.run(job_id)
            self.assertEqual(service.jobs[job_id]['status'], 'cancelled')
            self.assertEqual(service.processes, {})


class PreprocessingBoundaryTests(unittest.TestCase):
    def test_both_preprocessing_commands_accept_alpha_and_share_output(self):
        import analyze
        import run_analysis
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'talks').mkdir()
            source = root / 'talks' / 'input.md'
            source.write_bytes((Path(__file__).parent / 'fixtures' / 'alpha_smoke.md').read_bytes())
            for module in (analyze, run_analysis):
                with self.subTest(command=module.__name__), \
                     patch.object(module, '__file__', str(root / 'agent' / 'command.py')), \
                     patch.object(sys, 'argv', ['command', 'input']), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(module.main(), 0)
            self.assertEqual(len(list((root / 'analysis').rglob('统计_*.json'))), 1)
            self.assertEqual(len(list((root / 'analysis').rglob('分析prompt_*.md'))), 1)

    def test_full_preprocessing_does_not_create_unused_preview_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.md'
            source.write_bytes((Path(__file__).parent / 'fixtures' / 'alpha_smoke.md').read_bytes())
            artifacts = analysis_artifacts.build_artifacts(source, root / 'out')
            self.assertFalse(artifacts['prompt_file'].exists())
            self.assertNotIn('prompt', artifacts)
            self.assertTrue(artifacts['stats_file'].is_file())

    def test_preview_prompt_is_generated_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.json'
            source.write_bytes((Path(__file__).parent / 'fixtures' / 'canonical_smoke.json').read_bytes())
            artifacts = analysis_artifacts.build_artifacts(source, root / 'out', include_preview_prompt=True)
            self.assertTrue(artifacts['prompt_file'].is_file())
            self.assertIn('prompt', artifacts)
            self.assertEqual(artifacts['prompt_file'].read_text(encoding='utf-8'), artifacts['prompt'])


if __name__ == '__main__':
    unittest.main()
