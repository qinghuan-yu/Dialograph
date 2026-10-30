"""脱敏测试只使用虚构昵称与示例号码，不访问真实私聊。"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
import chunking
import chunk_pipeline as workflow
import analysis_artifacts
import run_workflow as workflow_cli
import report_generation
import llm_client
from web_app import WebService
from web_worker import run_job
from privacy_redaction import Redactor, load_redaction_terms

SENSITIVE = ['示例甲姓名', '示例乙姓名', 'wxid_synthetic_secret', '13800138000',
             'synthetic.person@example.invalid', '11010519491231002X', '6222021234567890',
             'https://example.invalid/private?token=synthetic', '123456789']


def private_chat():
    return {'session': {'id': 'wxid_synthetic_secret', 'displayName': '示例乙姓名', 'myDisplayName': '示例甲姓名'},
            'messages': [
                {'type': '文本消息', 'content': '示例甲姓名，我的手机13800138000，邮箱synthetic.person@example.invalid，'
                 '身份证11010519491231002X，银行卡6222021234567890，'
                 '微信wxid_synthetic_secret，QQ:123456789，https://example.invalid/private?token=synthetic',
                 'isSend': 0, 'formattedTime': '2026-10-28 10:00:00', 'createTime': 1793152800,
                 'senderUsername': 'wxid_synthetic_secret', 'unused_private_field': '不该保留的额外字段'},
                {'type': '文本消息', 'content': '示例乙姓名，13800138000我记下了，周末再聊。',
                 'isSend': 1, 'formattedTime': '2026-10-28 10:01:00', 'createTime': 1793152860},
            ]}


class PrivacyPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / '示例乙姓名.json'
        self.source.write_text(json.dumps(private_chat(), ensure_ascii=False), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def assert_redacted(self, text):
        for value in SENSITIVE:
            self.assertNotIn(value, text)

    def test_default_preprocessing_redacts_before_writing_and_preserves_original(self):
        original = self.source.read_bytes()
        artifacts = analysis_artifacts.build_artifacts(self.source, self.root / 'out')
        self.assert_redacted(json.dumps({'session': artifacts['session'], 'messages': artifacts['messages']}, ensure_ascii=False))
        self.assertNotEqual(artifacts['safe_name'], '示例乙姓名')
        self.assertEqual(artifacts['stats']['total_messages'], 2)
        self.assertEqual([m['original_index'] for m in artifacts['messages']], [0, 1])
        for path in (self.root / 'out').rglob('*'):
            if path.is_file():
                self.assert_redacted(path.name + path.read_text(encoding='utf-8'))
        self.assertEqual(self.source.read_bytes(), original)
        self.assertTrue(artifacts['parse_diagnostics'].get('redaction', {}).get('enabled'))

    def test_consistent_phone_placeholder_retains_repeated_evidence(self):
        artifacts = analysis_artifacts.build_artifacts(self.source, self.root / 'out')
        first, second = [m['content'] for m in artifacts['messages']]
        import re
        token = re.search(r'\[手机号\d+\]', first)
        self.assertIsNotNone(token)
        self.assertIn(token.group(), second)

    def test_custom_terms_affect_cache_identity_without_leaking_terms_in_metadata(self):
        first = analysis_artifacts.build_artifacts(self.source, self.root / 'out', redaction_terms=['周末'])
        second = analysis_artifacts.build_artifacts(self.source, self.root / 'out', redaction_terms=['其他词'])
        config = SimpleNamespace(model='test', api_base='https://test.invalid')
        from workflow_state import run_identity
        self.assertNotEqual(run_identity(first, 'skill', config), run_identity(second, 'skill', config))
        self.assertNotIn('周末', first['messages'][1]['content'])
        self.assertNotIn('周末', json.dumps(first['redaction'], ensure_ascii=False))

    def test_all_model_prompts_and_outputs_use_redaction_and_grounded_spans(self):
        artifacts = analysis_artifacts.build_artifacts(self.source, self.root / 'out')
        quote = artifacts['messages'][1]['content']
        chunk = {key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')}
        chunk['events'] = [{'summary': '记录约定', 'evidence': quote, 'confidence': 'low',
                            'source_message_ids': [artifacts['messages'][1]['message_id']]}]
        config = SimpleNamespace(model='test', api_base='https://test.invalid', api_key='test-only')
        responses = [json.dumps(chunk), '# 报告\n联系13800138000\n<!-- END_FINAL_REPORT -->',
                     '# 侧写\nsynthetic.person@example.invalid\n<!-- END_PERSONA_REPORT -->']
        def api(*args, **kwargs):
            self.assert_redacted(kwargs['system_prompt'] + kwargs['user_prompt'])
            return responses.pop(0)
        with patch.object(llm_client, 'call_llm', side_effect=api), contextlib.redirect_stdout(io.StringIO()):
            workflow.run_single_workflow(artifacts, 'skill', config, 5, settings=chunking.ChunkSettings(chunk_sleep=0))
            report_generation.generate_persona_report(artifacts, 'skill', config, 5)
        for path in (self.root / 'out').rglob('*'):
            if path.is_file():
                self.assert_redacted(path.name + path.read_text(encoding='utf-8'))
        ledger = json.loads(artifacts['evidence_ledger_file'].read_text(encoding='utf-8'))
        item = ledger['documents'][0]['events'][0]
        self.assertEqual(item['grounding']['status'], 'matched')

    def test_web_saves_supplemental_terms_with_task_and_keeps_original_snapshot(self):
        service = WebService(self.root)
        raw = self.source.read_bytes()
        row = service.import_chat('示例乙姓名.json', raw)
        with patch('web_app.threading.Thread.start'):
            state = service.start({'chat_id': row['id'], 'mode': 'preprocess', 'redaction_terms': ['周末']})
            service.cancel(state['id'])
            service.start({'retry_of': state['id']})
        folder = service.jobs_dir / state['id']
        options = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
        self.assertEqual(options['redaction_terms'], ['周末'])
        self.assertEqual((folder / 'input.json').read_bytes(), raw)
        result = run_job(self.root, folder, options, lambda event: None)
        self.assertTrue(result['diagnostics']['redaction']['enabled'])
        for item in result['files'].values():
            self.assert_redacted((folder / item['path']).read_text(encoding='utf-8'))

    def test_web_rejects_invalid_redaction_terms_before_creating_a_task(self):
        service = WebService(self.root)
        row = service.import_chat('synthetic.json', self.source.read_bytes())
        for terms in ('word', [123], [''], ['x' * 501]):
            with self.subTest(terms=terms), patch('web_app.threading.Thread.start'), self.assertRaises(ValueError):
                service.start({'chat_id': row['id'], 'mode': 'preprocess', 'redaction_terms': terms})
        self.assertEqual(service.snapshot(), [])

    def test_participant_alias_does_not_change_message_types_or_valid_timestamps(self):
        data = private_chat()
        data['session']['displayName'] = '文本'
        self.source.write_text(json.dumps(data), encoding='utf-8')
        artifacts = analysis_artifacts.build_artifacts(self.source, self.root / 'out')
        self.assertEqual(artifacts['messages'][0]['type'], '文本消息')
        self.assertEqual(artifacts['messages'][0]['time_str'], '2026-10-28 10:00:00')

    def test_escaped_model_json_is_redacted_before_any_chunk_file_is_saved(self):
        artifacts = analysis_artifacts.build_artifacts(self.source, self.root / 'out')
        chunk = {key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')}
        chunk['events'] = [{'summary': '电话', 'evidence': '13800138000', 'confidence': 'low',
                            'source_message_ids': [artifacts['messages'][0]['message_id']]}]
        encoded = json.dumps(chunk).replace('13800138000', ''.join('\\u%04x' % ord(c) for c in '13800138000'))
        config = SimpleNamespace(model='test', api_base='https://test.invalid', api_key='test-only')
        with patch.object(llm_client, 'call_llm', side_effect=[encoded, '报告 <!-- END_FINAL_REPORT -->']), \
             contextlib.redirect_stdout(io.StringIO()):
            workflow.run_single_workflow(artifacts, 'skill', config, 5, settings=chunking.ChunkSettings(chunk_sleep=0))
        for path in artifacts['runs_dir'].rglob('*'):
            if path.suffix in ('.json', '.md') and path.is_file():
                text = path.read_text(encoding='utf-8')
                if text.lstrip().startswith('{'):
                    self.assert_redacted(json.dumps(json.loads(text), ensure_ascii=False))

    def test_model_error_details_are_redacted_without_changing_error_type(self):
        artifacts = analysis_artifacts.build_artifacts(self.source, self.root / 'out')
        config = SimpleNamespace(model='test', api_base='https://test.invalid', api_key='test-only')
        with patch.object(llm_client, 'call_llm', side_effect=ValueError('bad 示例乙姓名 13800138000')), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError) as caught:
                workflow.run_single_workflow(artifacts, 'skill', config, 5, settings=chunking.ChunkSettings(chunk_sleep=0))
        self.assert_redacted(str(caught.exception))

    def test_opaque_evidence_ids_are_not_mistaken_for_phones_inside_hashes(self):
        message_id = 'M_' + 'a' * 10 + '13800138000' + 'b' * 11
        redactor = Redactor()
        self.assertEqual(redactor.redact(message_id), message_id)

    def test_supplemental_word_files_and_cli_argument_use_utf8(self):
        terms = self.root / '词表.txt'
        terms.write_text('合成单位\n合成住址', encoding='utf-8-sig')
        self.assertEqual(load_redaction_terms(terms), ['合成住址', '合成单位'])
        json_terms = self.root / '词表.json'
        json_terms.write_text(json.dumps(['合成单位']), encoding='utf-8')
        self.assertEqual(load_redaction_terms(json_terms), ['合成单位'])
        with patch.object(sys, 'argv', ['run_workflow', '--redact-terms-file', str(terms)]):
            self.assertEqual(workflow_cli.parse_args().redact_terms_file, terms)

    def test_phone_variants_share_token_and_placeholders_are_idempotent(self):
        redactor = Redactor()
        masked = redactor.redact('13800138000，+86 138 0013 8000')
        self.assertEqual(masked, '[手机号1]，[手机号1]')
        self.assertEqual(redactor.redact(masked), masked)

    def test_changed_hidden_value_changes_source_ids_but_keeps_conversation_directory(self):
        first = analysis_artifacts.build_artifacts(self.source, self.root / 'out')
        self.source.write_text(self.source.read_text(encoding='utf-8').replace('13800138000', '13900139000'), encoding='utf-8')
        second = analysis_artifacts.build_artifacts(self.source, self.root / 'out')
        self.assertEqual(first['safe_name'], second['safe_name'])
        self.assertEqual(first['messages'][0]['content'], second['messages'][0]['content'])
        self.assertNotEqual(first['messages'][0]['message_id'], second['messages'][0]['message_id'])

    def test_nickname_equal_to_confidence_enum_does_not_corrupt_structured_evidence(self):
        data = private_chat()
        data['session']['displayName'] = 'low'
        self.source.write_text(json.dumps(data), encoding='utf-8')
        artifacts = analysis_artifacts.build_artifacts(self.source, self.root / 'out')
        payload = {key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')}
        payload['events'] = [{'summary': '约定', 'evidence': '周末再聊', 'confidence': 'low'}]
        config = SimpleNamespace(model='test', api_base='https://test.invalid', api_key='test-only')
        with patch.object(llm_client, 'call_llm', return_value=json.dumps(payload)), \
             patch.object(workflow, 'generate_marked_report', return_value='报告 <!-- END_FINAL_REPORT -->'), \
             contextlib.redirect_stdout(io.StringIO()):
            workflow.run_single_workflow(artifacts, 'skill', config, 5, settings=chunking.ChunkSettings(chunk_sleep=0))
        ledger = json.loads(artifacts['evidence_ledger_file'].read_text(encoding='utf-8'))
        self.assertEqual(ledger['documents'][0]['events'][0]['confidence'], 'low')

    def test_malformed_timestamp_metadata_cannot_carry_private_dictionary_keys(self):
        data = private_chat()
        data['messages'][0]['createTime'] = {'synthetic.person@example.invalid': 13800138000}
        self.source.write_text(json.dumps(data), encoding='utf-8')
        artifacts = analysis_artifacts.build_artifacts(self.source, self.root / 'out')
        self.assert_redacted(json.dumps(artifacts['messages'], ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
