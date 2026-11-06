"""实测问题回归：全部使用虚构导出与替代 HTTP 响应。"""
from claim_fixtures import claim_response, with_reviews
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
    def test_empty_alpha_text_and_quote_keep_occurrence_without_becoming_literal_evidence(self):
        from analysis_artifacts import build_artifacts
        from web_app import WebService

        fixture = (Path(__file__).parent / 'fixtures' / 'alpha_smoke.md').read_text(encoding='utf-8')
        for label in ('', ' · 文本', ' · 引用'):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                raw = (fixture + f'\n**self\\_test** · 2026-10-28 09:04:00{label}\n\n \t\u3000\n'
                       '\n**合成对象** · 2026-10-28 09:05:00\n\n后续正文\n').encode('utf-8')
                service = WebService(Path(tmp))
                imported = service.import_chat('empty.md', raw)
                source = service.chat_path(imported['id'])
                self.assertEqual(source.read_bytes(), raw)
                normalized = parse_alpha_markdown(raw.decode('utf-8'))
                self.assertEqual(normalized['messages'][-2]['source_message_id'], 'alpha:5')
                artifacts = build_artifacts(source, Path(tmp) / 'output', write_supporting_files=False)
                empty, following = artifacts['messages'][-2:]
                self.assertEqual(empty['original_index'], 4)
                self.assertEqual(empty['sender'], 'me')
                self.assertEqual(empty['time_str'], '2026-10-28 09:04:00')
                self.assertEqual(empty['type'], '文件消息')
                self.assertIn('导出正文为空', empty['content'])
                self.assertEqual(match_quote({'evidence': empty['content']}, artifacts['messages'])['status'], 'unmatched')
                self.assertEqual(following['original_index'], 5)
                self.assertEqual(following['content'], '后续正文')

    def test_other_and_future_types_keep_occurrence_without_importing_body(self):
        fixture = (Path(__file__).parent / 'fixtures' / 'alpha_smoke.md').read_text(encoding='utf-8')
        for label in ('其他', '小程序', '视频号', '音乐', '实时位置', '实况照片',
                      '群公告', '收藏笔记', '拍一拍', '未来扩展类型-v2'):
            with self.subTest(label=label):
                text = fixture + (
                    f'\n**self\\_test** · 2026-10-28 09:04:00 · {label}\n\n'
                    '<appmsg>附件内第三人原话</appmsg>\n'
                    '\n**合成对象** · 2026-10-28 09:05:00\n\n后续正文\n')
                normalized = parse_alpha_markdown(text)
                opaque = normalized['messages'][-2]
                self.assertEqual(opaque['source_message_id'], 'alpha:5')
                self.assertEqual(opaque['isSend'], 1)
                self.assertEqual(opaque['formattedTime'], '2026-10-28 09:04:00')
                self.assertEqual(opaque['type'], '文件消息')
                self.assertIn(f'原导出类型：{label}', opaque['content'])
                self.assertNotIn('附件内第三人原话', str(normalized))
                parsed = parse_chat_messages(normalized['messages'])['messages']
                self.assertEqual(parsed[-2]['original_index'], 4)
                self.assertEqual(parsed[-2]['sender'], 'me')
                self.assertEqual(parsed[-1]['content'], '后续正文')
                self.assertEqual(match_quote({'evidence': opaque['content']}, parsed)['status'], 'unmatched')

    def test_weflow_native_export_labels_are_accepted_without_numeric_type_guessing(self):
        from conversation_formats import normalize_conversation

        labels = ('文本消息', '图片消息', '语音消息', '名片消息', '视频消息', '动画表情',
                  '位置消息', '链接消息', '通话消息', '系统消息', '引用消息', '音乐消息',
                  '群公告', '转账消息', '文件消息', '聊天记录', '小程序消息', '其他消息')
        data = {'messages': [
            {'type': label, 'content': '合成导出说明', 'isSend': 1,
             'formattedTime': '2026-10-28 09:00:00'} for label in labels]}
        self.assertEqual(normalize_conversation(data), data)

    def test_location_messages_pass_full_web_workflow_without_importing_coordinates(self):
        from analysis_artifacts import build_artifacts
        from web_app import WebService
        from web_worker import run_job

        text = (Path(__file__).parent / 'fixtures' / 'alpha_smoke.md').read_text(encoding='utf-8')
        text += ('\n**self\\_test** · 2026-10-28 09:04:00 · 位置\n\n'
                 '合成地址甲：纬度 31.123456 经度 121.654321\n'
                 '\n**合成对象** · 2026-10-28 09:05:00 · 位置\n\n'
                 '合成地址乙：纬度 32.123456 经度 122.654321\n'
                 '\n**合成对象** · 2026-10-28 09:06:00\n\n收到位置\n')
        raw = text.encode('utf-8')
        config = SimpleNamespace(api_base='https://test.invalid', api_key='test-only', model='test')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = WebService(root)
            imported = service.import_chat('location.md', raw)
            source = service.chat_path(imported['id'])
            with patch('web_app.load_llm_config', return_value=config), patch('web_app.threading.Thread.start'):
                job = service.start({'chat_id': imported['id'], 'mode': 'full'})
            folder = service.jobs_dir / job['id']
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual((folder / 'input.md').read_bytes(), raw)
            options = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
            artifacts = build_artifacts(source, root / 'check', write_supporting_files=False)
            locations = artifacts['messages'][-3:-1]
            self.assertEqual([m['original_index'] for m in locations], [4, 5])
            self.assertEqual([m['sender'] for m in locations], ['me', 'other'])
            self.assertEqual([m['time_str'] for m in locations], ['2026-10-28 09:04:00', '2026-10-28 09:05:00'])
            for location in locations:
                self.assertEqual(location['type'], '文件消息')
                self.assertIn('原导出类型：位置', location['content'])
                self.assertEqual(match_quote({'evidence': location['content']}, artifacts['messages'])['status'], 'unmatched')
            chunk = json.dumps({key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')})
            responses = [chunk, claim_response('合成报告', report_kind='analysis'), claim_response('合成侧写', report_kind='persona')]
            with patch('web_worker.load_llm_config', return_value=config), patch('llm_client.call_llm', side_effect=with_reviews(responses)):
                result = run_job(root, folder, options, lambda event: None)
            self.assertEqual(result['stats']['total_messages'], 6)
            self.assertEqual(len(result['files']), 11)
            ledger = json.loads((folder / result['files']['evidence']['path']).read_text(encoding='utf-8'))
            saved = json.dumps(ledger, ensure_ascii=False)
            for private in ('合成地址甲', '合成地址乙', '31.123456', '121.654321', '32.123456', '122.654321'):
                self.assertNotIn(private, saved)
            self.assertEqual(ledger['source_messages'][-1]['content'], '收到位置')

    def test_contact_card_import_keeps_source_position_and_omits_contact_details(self):
        from analysis_artifacts import build_artifacts
        from web_app import WebService

        fixture = Path(__file__).parent / 'fixtures' / 'alpha_smoke.md'
        text = fixture.read_text(encoding='utf-8')
        text += ('\n**self\\_test** · 2026-10-28 09:04:00 · 名片\n\n'
                 '姓名：合成联系人\n微信号：wxid_synthetic_contact\n'
                 '\n**合成对象** · 2026-10-28 09:05:00\n\n名片收到了\n')
        raw = text.encode('utf-8')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = WebService(root)
            imported = service.import_chat('contact-card.md', raw)
            source = service.chat_path(imported['id'])
            artifacts = build_artifacts(source, root / 'output', write_supporting_files=False)
            self.assertEqual(source.read_bytes(), raw)
        card, following = artifacts['messages'][-2:]
        self.assertEqual(card['original_index'], 4)
        self.assertEqual(card['sender'], 'me')
        self.assertEqual(card['time_str'], '2026-10-28 09:04:00')
        self.assertEqual(card['type'], '文件消息')
        self.assertIn('原导出类型：名片', card['content'])
        self.assertNotIn('合成联系人', card['content'])
        self.assertNotIn('wxid_synthetic_contact', card['content'])
        self.assertEqual(following['original_index'], 5)
        self.assertEqual(following['content'], '名片收到了')
        self.assertEqual(match_quote({'evidence': card['content']}, artifacts['messages'])['status'], 'unmatched')

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

    def test_output_floor_is_explicit_and_validated(self):
        self.assertEqual(self.load().min_output_tokens, 0)
        self.assertEqual(self.load(LLM_MIN_OUTPUT_TOKENS='24000').min_output_tokens, 24000)
        for value in ('-1', '1.5', 'typo', ''):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'LLM_MIN_OUTPUT_TOKENS'):
                self.load(LLM_MIN_OUTPUT_TOKENS=value)

    def test_output_floor_reaches_requests_and_truncation_retries(self):
        config = self.load(LLM_MIN_OUTPUT_TOKENS='24000')
        for marked in (False, True):
            with self.subTest(marked=marked), patch.object(llm_client, 'call_llm', side_effect=[
                    llm_client.LLMOutputTruncated(), 'done END']) as api:
                if marked:
                    llm_client.generate_marked_report(config, 's', 'u', 10, 'END', 'report', 2600,
                                                     retry_sleep_seconds=0)
                else:
                    result = llm_client.call_llm_with_retry_result(config, 's', 'u', 10, 2600,
                                                                  retry_sleep_seconds=0)
                    self.assertEqual(result.max_tokens_used, 48000)
            self.assertEqual([call.kwargs['max_tokens'] for call in api.call_args_list], [24000, 48000])

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
