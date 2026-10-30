"""实测问题回归：全部使用虚构导出与替代 HTTP 响应。"""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from analyze import parse_chat_messages
from config import load_llm_config
from conversation_formats import parse_alpha_markdown
from evidence_grounding import match_quote
import llm_client
from workflow_state import run_identity


class AlphaOpaqueAttachmentTests(unittest.TestCase):
    def test_known_opaque_types_keep_position_and_sender_without_importing_attachment_text(self):
        text = (Path(__file__).parent / 'fixtures' / 'alpha_smoke.md').read_text(encoding='utf-8')
        labels = ('聊天记录', '转账', '红包', '通话')
        for index, label in enumerate(labels, 4):
            text += f'\n**self\\_test** · 2026-10-28 09:0{index}:00 · {label}\n\n附件中的第三人原话不属于发送者\n'
        normalized = parse_alpha_markdown(text)
        self.assertEqual(len(normalized['messages']), 8)
        for ordinal, (message, label) in enumerate(zip(normalized['messages'][4:], labels), 5):
            self.assertEqual(message['source_message_id'], f'alpha:{ordinal}')
            self.assertEqual(message['isSend'], 1)
            self.assertEqual(message['type'], '文件消息')
            self.assertIn(label, message['content'])
            self.assertIn('附件内容未解析', message['content'])
            self.assertNotIn('第三人原话', message['content'])
        parsed = parse_chat_messages(normalized['messages'])
        self.assertEqual(len(parsed['messages']), 7)
        self.assertEqual(parsed['diagnostics']['skip_reasons'], {'system_message': 1})
        for message in parsed['messages'][-4:]:
            self.assertEqual(match_quote({'evidence': message['content']}, parsed['messages'])['status'], 'unmatched')


class ThinkingConfigurationTests(unittest.TestCase):
    def load(self, **extra):
        environment = {'LLM_API_BASE': 'https://api.deepseek.com', 'LLM_API_KEY': 'test-only',
                       'LLM_MODEL': 'deepseek-v4-flash', **extra}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, environment, clear=True):
            return load_llm_config(Path(tmp))

    def test_official_deepseek_v4_defaults_to_non_thinking_for_bounded_outputs(self):
        for model in ('deepseek-v4-flash', 'deepseek-v4-pro'):
            with self.subTest(model=model):
                self.assertEqual(self.load(LLM_MODEL=model).thinking, 'disabled')

    def test_other_services_and_models_keep_provider_defaults(self):
        for settings in ({'LLM_API_BASE': 'https://compatible.invalid/v1'},
                         {'LLM_MODEL': 'other-model'},
                         {'LLM_API_BASE': 'https://api.deepseek.com.untrusted.invalid'}):
            with self.subTest(settings=settings):
                self.assertIsNone(self.load(**settings).thinking)

    def test_explicit_thinking_mode_overrides_automatic_default(self):
        self.assertEqual(self.load(LLM_THINKING='enabled').thinking, 'enabled')
        self.assertEqual(self.load(LLM_THINKING='disabled').thinking, 'disabled')
        self.assertIsNone(self.load(LLM_THINKING='default').thinking)

    def test_invalid_thinking_setting_is_rejected_before_requests(self):
        with self.assertRaisesRegex(ValueError, 'LLM_THINKING'):
            self.load(LLM_THINKING='typo')

    def test_request_contains_configured_thinking_mode(self):
        config = self.load()
        response = io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop',
                                  'message': {'content': '合成响应'}}]}).encode('utf-8'))
        with patch.object(llm_client.request, 'urlopen', return_value=response) as opened:
            result = llm_client.call_llm_with_retry_result(config, 'system', 'synthetic', 10, 1200)
        self.assertEqual(result.content, '合成响应')
        payload = json.loads(opened.call_args.args[0].data.decode('utf-8'))
        self.assertEqual(payload['thinking'], {'type': 'disabled'})
        self.assertEqual(payload['max_tokens'], 1200)

    def test_provider_default_omits_extra_request_parameter(self):
        config = self.load(LLM_API_BASE='https://compatible.invalid/v1')
        response = io.BytesIO(b'{"choices":[{"message":{"content":"done"}}]}')
        with patch.object(llm_client.request, 'urlopen', return_value=response) as opened:
            llm_client.call_llm_with_retry_result(config, 'system', 'synthetic', 10, 1200)
        self.assertNotIn('thinking', json.loads(opened.call_args.args[0].data.decode('utf-8')))

    def test_thinking_mode_changes_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.json'
            source.write_text('{}', encoding='utf-8')
            artifacts = {'chat_file': source, 'messages': []}
            first = SimpleNamespace(model='test', api_base='https://test.invalid', thinking='disabled')
            second = SimpleNamespace(model='test', api_base='https://test.invalid', thinking='enabled')
            self.assertNotEqual(run_identity(artifacts, 'skill', first), run_identity(artifacts, 'skill', second))


if __name__ == '__main__':
    unittest.main()
