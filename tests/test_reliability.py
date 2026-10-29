"""本轮回归：保护已有结果，拒绝无效输入与错误缓存。"""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'agent'))
import evidence_validation
import artifact_audit
import chunk_pipeline as workflow
import analysis_artifacts
import analysis_service
import llm_client
import run_single_pass_analysis as preview_analysis
import run_single_pass_persona as preview_persona
import web_app
import web_worker

KEYS = ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')
CONFIG = SimpleNamespace(api_key='test-only', model='test', api_base='https://test.invalid')


def chat():
    return {'session': {'displayName': '测试对象'}, 'messages': [
        {'type': '文本消息', 'content': f'消息{i}', 'isSend': i % 2,
         'formattedTime': f'2026-05-01 10:0{i}:00'} for i in range(4)]}


def metadata():
    return {'chunk_index': 1, 'start_message': 1, 'end_message': 2,
            'start_time': '2026-05-01 10:00:00', 'end_time': '2026-05-01 10:01:00'}


class ResultProtectionTests(unittest.TestCase):
    def test_both_preview_commands_preserve_full_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'talks').mkdir()
            (root / 'talks' / 'sample.json').write_text(json.dumps(chat()), encoding='utf-8')
            output = root / 'analysis'
            output.mkdir()
            originals = [output / '分析_测试对象.md', output / '人物侧写_测试对象.md']
            for path in originals:
                path.write_text('FULL REPORT', encoding='utf-8')
            for module, generator in ((preview_analysis, 'generate_report_with_fallback'),
                                      (preview_persona, 'generate_persona_with_fallback')):
                with patch.object(module, '__file__', str(root / 'agent' / 'preview.py')), \
                     patch.object(module, 'load_llm_config', return_value=CONFIG), \
                     patch.object(module, 'load_skill_prompt', return_value='skill'), \
                     patch.object(module, generator, return_value='PREVIEW ONLY'), \
                     patch.object(sys, 'argv', ['preview', 'sample']), redirect_stdout(io.StringIO()):
                    self.assertEqual(module.main(), 0)
            self.assertTrue(all(p.read_text(encoding='utf-8') == 'FULL REPORT' for p in originals))
            previews = list((output / '预览').glob('人物侧写_*.md'))
            self.assertEqual(len(previews), 1)
            self.assertIn('预览', previews[0].read_text(encoding='utf-8'))

    def test_batch_entry_never_deletes_existing_analysis(self):
        batch = (ROOT / '一键运行全部分析.bat').read_text(encoding='utf-8').lower()
        self.assertNotIn('remove-item', batch)
        self.assertNotIn('rmdir', batch)
        self.assertNotIn(' --force', batch)

    def test_empty_workspace_audit_returns_empty_json(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(artifact_audit, '__file__', str(Path(tmp) / 'agent' / 'artifact_audit.py')), \
             patch.object(sys, 'argv', ['audit', '--json']), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(artifact_audit.main(), 0)
        self.assertEqual(json.loads(out.getvalue()), [])


class InputAndEvidenceTests(unittest.TestCase):
    def test_web_worker_surfaces_audit_warnings_in_result_and_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'job'
            directory.mkdir()
            (directory / 'input.json').write_text(json.dumps(chat()), encoding='utf-8')
            warnings = ['events 摘要覆盖缩减，完整条目仍在总表。']
            events = []
            with patch.object(web_worker, 'load_llm_config', return_value=CONFIG), \
                 patch.object(analysis_service, 'audit_artifacts', side_effect=[SimpleNamespace(ok=False), SimpleNamespace(ok=True, warnings=warnings)]), \
                 patch.object(workflow, 'run_single_workflow_with_auto_shrink', return_value={}), \
                 patch.object(analysis_service, 'generate_persona_report'):
                result = web_worker.run_job(root, directory, {'mode': 'full'}, events.append)
            self.assertEqual(result.get('warnings'), warnings)
            self.assertTrue(any('覆盖缩减' in event.get('message', '') for event in events))

    def test_other_type_conventions_are_not_silently_treated_as_weflow(self):
        for value in ('text', 'image', ''):
            data = chat()
            data['messages'][0]['type'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                web_app.validate_chat(data)

    def test_malformed_ledger_returns_audit_errors_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ledger.json'
            for ledger in ([], {'documents': [None]}, {'documents': ['invalid']}):
                path.write_text(json.dumps(ledger), encoding='utf-8')
                result = artifact_audit.AuditResult('test')
                artifact_audit.audit_evidence_ledger(path, result)
                self.assertFalse(result.ok)

    def test_audit_discloses_summary_coverage_reduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ledger.json'
            document = {key: [] for key in KEYS}
            document.update(chunk_index=1, parse_status='ok', events=[
                {'summary': '事实', 'evidence': '消息', 'confidence': 'low'} for _ in range(61)])
            path.write_text(json.dumps({'documents': [document], 'chunk_count': 1}), encoding='utf-8')
            result = artifact_audit.AuditResult('test')
            artifact_audit.audit_evidence_ledger(path, result)
            self.assertTrue(result.ok, result.errors)
            self.assertTrue(any('60/61' in warning for warning in result.warnings))

    def test_summary_spans_time_axis_when_chunks_exceed_budget(self):
        documents = [{'chunk_index': i, 'events': [{'summary': str(i)}]} for i in range(70)]
        items = evidence_validation.balanced_items(documents, 'events', 60)
        self.assertEqual(len(items), 60)
        self.assertEqual(items[0]['chunk_index'], 0)
        self.assertEqual(items[-1]['chunk_index'], 69)

    def test_non_transient_report_failure_is_not_retried(self):
        with patch.object(llm_client, 'call_llm_with_retry_result', side_effect=ValueError('invalid')) as api:
            with self.assertRaises(ValueError):
                llm_client.generate_marked_report(CONFIG, 'skill', 'prompt', 10,
                    '<!-- END_FINAL_REPORT -->', 'report', 4800, retry_sleep_seconds=0)
        self.assertEqual(api.call_count, 1)

    def test_invalid_or_oversized_report_requests_are_not_retried_unchanged(self):
        for code in (400, 401, 413):
            with self.subTest(code=code), patch.object(llm_client, 'call_llm_with_retry_result',
                    side_effect=HTTPError('https://test.invalid', code, 'invalid', {}, None)) as api:
                with self.assertRaises(HTTPError):
                    llm_client.generate_marked_report(CONFIG, 'skill', 'prompt', 10,
                        '<!-- END_FINAL_REPORT -->', 'report', 4800, retry_sleep_seconds=0)
                self.assertEqual(api.call_count, 1)

    def test_cli_rejects_missing_sender_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'input.json', Path(tmp) / 'out'
            data = chat()
            del data['messages'][0]['isSend']
            source.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                analysis_artifacts.build_artifacts(source, out)
            self.assertFalse(out.exists())

    def test_web_rejects_group_session_and_multiple_other_ids(self):
        for mode in ('session', 'senders'):
            data = chat()
            if mode == 'session':
                data['session']['type'] = 'group'
            else:
                data['messages'][0]['senderUsername'] = 'person-a'
                data['messages'][2]['senderUsername'] = 'person-b'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                web_app.validate_chat(data)

    def test_non_integer_sender_is_rejected(self):
        for value in (True, 1.0, '1', None):
            data = chat()
            data['messages'][0]['isSend'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                web_app.validate_chat(data)

    def test_malformed_evidence_never_becomes_complete(self):
        valid = {key: [] for key in KEYS}
        for data in ({}, {**valid, 'events': 'bad'}, {**valid, 'persona_signals': ['bad']},
                     {**valid, 'events': [{}]}):
            with self.subTest(data=data):
                evidence = workflow.normalize_chunk_evidence(json.dumps(data), metadata())
                self.assertFalse(workflow.is_complete_chunk_evidence(evidence))

    def test_valid_empty_evidence_is_explicitly_insufficient(self):
        evidence = workflow.normalize_chunk_evidence(json.dumps({key: [] for key in KEYS}), metadata())
        self.assertTrue(workflow.is_complete_chunk_evidence(evidence))
        self.assertEqual(evidence.get('evidence_status'), 'insufficient')

    def test_summary_is_balanced_across_chunks_and_discloses_omission(self):
        docs = [{**{key: [] for key in KEYS}, 'chunk_index': i, 'parse_status': 'ok',
                 'events': [{'summary': f'item-{i}-{j}'} for j in range(70)]} for i in (1, 2)]
        items = evidence_validation.balanced_items(docs, 'events', 60)
        self.assertEqual({item['chunk_index'] for item in items}, {1, 2})
        summary = workflow.format_evidence_summary(docs)
        self.assertIn('总数 140', summary)
        self.assertIn('省略 80', summary)

    def test_final_report_retries_transient_network_failure(self):
        response = llm_client.LLMRetryResult('done <!-- END_FINAL_REPORT -->', 4800)
        with patch.object(llm_client, 'call_llm_with_retry_result', side_effect=[TimeoutError('timeout'), response]) as api:
            result = llm_client.generate_marked_report(CONFIG, 'skill', 'prompt', 10,
                '<!-- END_FINAL_REPORT -->', 'report', 4800, retry_sleep_seconds=0)
        self.assertIn('END_FINAL_REPORT', result)
        self.assertEqual(api.call_count, 2)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(llm_client, 'call_llm', side_effect=AssertionError('tests must not call a provider'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_web_automatic_resume_skips_legacy_metadata_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'job'
            directory.mkdir()
            (directory / 'input.json').write_text(json.dumps(chat()), encoding='utf-8')
            artifacts = analysis_artifacts.build_artifacts(directory / 'input.json', directory / 'output')
            run = artifacts['runs_dir'] / 'legacy'
            run.mkdir(parents=True)
            (run / 'run_metadata.json').write_text('{}', encoding='utf-8')
            with patch.object(web_worker, 'load_llm_config', return_value=CONFIG), \
                 patch.object(workflow, 'run_single_workflow_with_auto_shrink', side_effect=RuntimeError('new run')) as dispatch:
                with self.assertRaisesRegex(RuntimeError, 'new run'):
                    web_worker.run_job(root, directory, {'mode': 'full', 'resume': True}, lambda event: None)
                self.assertIsNone(dispatch.call_args.kwargs['resume_run_id'])

    def test_failed_run_audit_cannot_replace_an_existing_full_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.prepare(Path(tmp))
            artifacts['analysis_file'].write_text('OLD FULL REPORT', encoding='utf-8')
            text = json.dumps({key: [] for key in KEYS})
            def generated(config, prompt, timeout, meta, **kwargs):
                return text, workflow.normalize_chunk_evidence(text, meta), 1200
            failed = artifact_audit.AuditResult('test')
            failed.add_error('test failure')
            with patch.object(workflow, 'generate_chunk_analysis', side_effect=generated), \
                 patch.object(workflow, 'generate_marked_report', return_value='report <!-- END_FINAL_REPORT -->'), \
                 patch.object(workflow, 'audit_artifacts', return_value=failed):
                with self.assertRaisesRegex(ValueError, '审计'):
                    workflow.run_single_workflow(artifacts, 'skill', CONFIG, 5, chunk_sleep=0)
            self.assertEqual(artifacts['analysis_file'].read_text(encoding='utf-8'), 'OLD FULL REPORT')

    def test_resume_reuses_valid_chunks_and_retains_identity_after_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.prepare(Path(tmp))
            text = json.dumps({key: [] for key in KEYS})
            def generated(config, prompt, timeout, meta, **kwargs):
                return text, workflow.normalize_chunk_evidence(text, meta), 1200
            with patch.object(workflow, 'generate_chunk_analysis', side_effect=generated), \
                 patch.object(workflow, 'generate_marked_report', side_effect=RuntimeError('final stop')):
                with self.assertRaisesRegex(RuntimeError, 'final stop'):
                    workflow.run_single_workflow(artifacts, 'skill', CONFIG, 5, chunk_size=2, chunk_sleep=0)
            run = next(artifacts['runs_dir'].iterdir())
            with patch.object(workflow, 'generate_chunk_analysis', side_effect=AssertionError('cached chunks must not call api')) as chunk_api, \
                 patch.object(workflow, 'generate_marked_report', return_value='report <!-- END_FINAL_REPORT -->'):
                workflow.run_single_workflow(artifacts, 'skill', CONFIG, 5, resume_run_id=run.name, chunk_sleep=0)
            self.assertEqual(chunk_api.call_count, 0)
            saved = json.loads((run / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'success')
            self.assertEqual(saved['chunk_size'], 2)
            self.assertEqual(len(saved['input_sha256']), 64)

    def test_corrupt_cached_message_range_is_rejected_without_rewriting_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.prepare(Path(tmp))
            run = self.interrupted_run(artifacts)
            saved = (run / 'run_metadata.json').read_bytes()
            evidence = workflow.normalize_chunk_evidence(json.dumps({key: [] for key in KEYS}), metadata())
            evidence['message_range']['end'] = 999
            paths = workflow.build_run_paths(artifacts, run.name)
            (paths['chunk_dir'] / 'chunk_001.md').write_text('cached', encoding='utf-8')
            workflow.write_chunk_evidence(paths['evidence_dir'], evidence)
            with self.assertRaisesRegex(ValueError, '范围'):
                workflow.run_single_workflow(artifacts, 'skill', CONFIG, 5, resume_run_id=run.name)
            self.assertEqual((run / 'run_metadata.json').read_bytes(), saved)

    def prepare(self, root):
        source = root / 'input.json'
        source.write_text(json.dumps(chat()), encoding='utf-8')
        return analysis_artifacts.build_artifacts(source, root / 'out')

    def interrupted_run(self, artifacts):
        with patch.object(workflow, 'generate_chunk_analysis', side_effect=RuntimeError('stop')):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                workflow.run_single_workflow(artifacts, 'skill', CONFIG, 5, chunk_size=2,
                    chunk_max_bytes=13000, chunk_max_prompt_bytes=20000, chunk_sleep=0)
        return next(artifacts['runs_dir'].iterdir())

    def test_run_ids_do_not_collide_within_one_second(self):
        self.assertEqual(len({workflow.make_run_id() for _ in range(20)}), 20)

    def test_resume_restores_limits_instead_of_repartitioning(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.prepare(Path(tmp))
            run = self.interrupted_run(artifacts)
            before = (run / f'分块覆盖清单_{artifacts["safe_name"]}.json').read_bytes()
            with patch.object(workflow, 'generate_chunk_analysis', side_effect=RuntimeError('stop')):
                with self.assertRaisesRegex(RuntimeError, 'stop'):
                    workflow.run_single_workflow(artifacts, 'skill', CONFIG, 5, resume_run_id=run.name, chunk_sleep=0)
            self.assertEqual((run / f'分块覆盖清单_{artifacts["safe_name"]}.json').read_bytes(), before)
            state = json.loads((run / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(state['chunk_size'], 2)
            self.assertEqual(state['chunk_max_bytes'], 13000)

    def test_changed_input_is_rejected_before_metadata_is_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.prepare(Path(tmp))
            run = self.interrupted_run(artifacts)
            saved = (run / 'run_metadata.json').read_bytes()
            data = chat()
            data['messages'][0]['content'] = 'changed'
            artifacts['chat_file'].write_text(json.dumps(data), encoding='utf-8')
            changed = analysis_artifacts.build_artifacts(artifacts['chat_file'], Path(tmp) / 'out')
            with patch.object(workflow, 'generate_chunk_analysis', side_effect=AssertionError('no api')):
                with self.assertRaisesRegex(ValueError, '输入|指纹'):
                    workflow.run_single_workflow(changed, 'skill', CONFIG, 5, resume_run_id=run.name)
            self.assertEqual((run / 'run_metadata.json').read_bytes(), saved)

    def test_legacy_resume_and_changed_prompt_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.prepare(Path(tmp))
            run = self.interrupted_run(artifacts)
            with self.assertRaisesRegex(ValueError, '流程|指纹'):
                workflow.run_single_workflow(artifacts, 'changed skill', CONFIG, 5, resume_run_id=run.name)
            (run / 'run_metadata.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '旧|校验|元数据'):
                workflow.run_single_workflow(artifacts, 'skill', CONFIG, 5, resume_run_id=run.name)


if __name__ == '__main__':
    unittest.main()
