"""仅用合成输入验证结构优化后的行为边界。"""
from datetime import datetime
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
import analyze
import analysis_artifacts
import chunk_pipeline
import web_app
import workflow_state


class StructureBoundaryTests(unittest.TestCase):
    def test_internal_statistics_require_fields_but_allow_absent_sender_counts(self):
        messages = analyze.parse_chat_messages([{'type': '文本消息', 'content': '合成消息',
            'isSend': 0, 'formattedTime': '2099-07-08 09:10:11'}])['messages']
        stats = analyze.compute_statistics(messages, {})
        self.assertIn('我=0次', analysis_artifacts.format_stats_summary(stats))
        for key in ('session_count', 'session_gap_hours', 'session_initiators', 'session_closers',
                    'avg_session_message_count', 'median_session_duration_minutes',
                    'session_reply_time_my_avg_seconds', 'invalid_time_message_count'):
            with self.subTest(field=key):
                incomplete = dict(stats)
                del incomplete[key]
                with self.assertRaises(KeyError):
                    analysis_artifacts.format_stats_summary(incomplete)

    def test_internal_diagnostics_do_not_hide_missing_fields_as_zero(self):
        diagnostics = analyze.parse_chat_messages([])['diagnostics']
        self.assertIn('原始消息数: 0', analysis_artifacts.format_parse_diagnostics_report(diagnostics))
        for key in ('raw_message_count', 'invalid_time_message_count', 'skip_reasons', 'message_types'):
            with self.subTest(field=key):
                incomplete = dict(diagnostics)
                del incomplete[key]
                with self.assertRaises(KeyError):
                    analysis_artifacts.format_parse_diagnostics_report(incomplete)

    def test_documented_scripts_import_and_run_from_an_isolated_working_directory(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / 'agent'
            scripts.mkdir()
            for file in (project / 'agent').glob('*.py'):
                shutil.copyfile(file, scripts / file.name)
            (root / 'talks').mkdir()
            shutil.copyfile(project / 'tests/fixtures/canonical_smoke.json', root / 'talks/input.json')
            environment = {**os.environ, 'LLM_API_KEY': '', 'DEEPSEEK_API_KEY': '', 'OPENAI_API_KEY': ''}
            for script, argument, code in [('run_workflow.py', '--help', 0), ('web_app.py', '--help', 0),
                    ('evidence_store.py', '--help', 0), ('analyze.py', 'input', 0), ('run_analysis.py', 'input', 0),
                    ('run_single_pass_analysis.py', 'input', 1), ('run_single_pass_persona.py', 'input', 1)]:
                with self.subTest(script=script):
                    result = subprocess.run([sys.executable, '-E', '-X', 'utf8', str(scripts / script), argument],
                        cwd=root, env=environment, capture_output=True, text=True, encoding='utf-8', timeout=30)
                    self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                    self.assertNotIn('ImportError', result.stderr)
                    self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_chunk_settings_shrink_together_without_losing_request_options(self):
        from chunking import ChunkSettings
        settings = ChunkSettings(chunk_size=100, chunk_max_bytes=18000,
                                 chunk_max_prompt_bytes=32000, chunk_timeout=9, chunk_sleep=0.2)
        smaller = settings.smaller()
        self.assertEqual((smaller.chunk_size, smaller.chunk_max_bytes, smaller.chunk_max_prompt_bytes),
                         (55, 12000, 18000))
        self.assertEqual((smaller.chunk_timeout, smaller.chunk_sleep), (9, 0.2))
        self.assertEqual(settings.chunk_size, 100)
        self.assertIsNone(smaller.smaller())

    def test_time_labels_share_parsing_and_preserve_invalid_input_fallbacks(self):
        message = {'time_str': '2099-07-08 09:10:11'}
        analyze._parse_time_text.cache_clear()
        with patch.object(analyze, 'datetime', wraps=datetime) as clock:
            self.assertEqual(analyze.message_date_label(message), '2099-07-08')
            self.assertEqual(analyze.message_month_label(message), '2099-07')
            self.assertEqual(analyze.message_hour(message), 9)
            self.assertEqual(clock.strptime.call_count, 1)
        for value in (None, [], {}, 23):
            self.assertIsNone(analyze.parse_message_time(value))
        self.assertEqual(analyze.message_date_label({'time_str': 'bad-date-text'}), 'bad-date-t')

    def test_new_runtime_modules_participate_in_reuse_fingerprint(self):
        config = SimpleNamespace(model='test', api_base='https://test.invalid')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.json'
            source.write_text('{}', encoding='utf-8')
            artifacts = {'chat_file': source, 'messages': []}
            baseline = workflow_state.run_identity(artifacts, 'skill', config)
            read = Path.read_bytes
            for module in ('chunking.py', 'chunk_evidence.py', 'summary_merge.py'):
                with self.subTest(module=module), patch.object(Path, 'read_bytes',
                        lambda path: read(path) + (b'\n# synthetic edit' if path.name == module else b'')):
                    self.assertNotEqual(workflow_state.run_identity(artifacts, 'skill', config), baseline)

    def test_worker_error_records_only_exception_kind_and_releases_process(self):
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
            state = service.jobs[job_id]
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['failure_kind'], 'KeyError')
            self.assertNotIn('files', state)
            self.assertEqual(service.processes, {})

    def test_non_chunk_errors_never_trigger_automatic_restarts(self):
        with patch.object(chunk_pipeline, 'run_single_workflow', side_effect=ValueError('synthetic audit failure')) as run:
            with self.assertRaisesRegex(ValueError, 'audit failure'):
                chunk_pipeline.run_single_workflow_with_auto_shrink({}, 'skill', object(), 1)
            self.assertEqual(run.call_count, 1)

    def test_preview_shared_fallback_is_lazy_and_does_not_retry_unrelated_errors(self):
        import preview_support
        config = object()
        fallback = MagicMock(return_value='small prompt')
        with patch.object(preview_support, 'call_llm_with_retry', side_effect=ValueError('synthetic error')) as call:
            with self.assertRaisesRegex(ValueError, 'synthetic error'):
                preview_support.generate_preview(config, 'skill', 'primary', fallback)
            fallback.assert_not_called()
            self.assertEqual(call.call_count, 1)

    def test_preview_recoverable_failure_uses_shared_request_limits(self):
        import preview_support
        fallback = MagicMock(return_value='small prompt')
        with patch.object(preview_support, 'call_llm_with_retry', side_effect=[TimeoutError('synthetic timeout'), 'done']) as call:
            self.assertEqual(preview_support.generate_preview(object(), 'skill', 'primary', fallback), 'done')
        fallback.assert_called_once_with()
        self.assertEqual([item.kwargs['max_tokens'] for item in call.call_args_list], [4200, 3200])
        self.assertEqual([item.kwargs['max_attempts'] for item in call.call_args_list], [1, 2])


if __name__ == '__main__':
    unittest.main()
