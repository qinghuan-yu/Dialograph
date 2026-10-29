"""多格式导入回归；只使用合成私聊和群聊。"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from analyze import load_chat, parse_chat_messages
from run_workflow import build_artifacts, select_chat_files
from web_app import WebService, validate_chat
from web_worker import run_job


def canonical_chat():
    return {
        'schema': 'dialograph.conversation', 'version': 1,
        'session': {'id': 'peer_test', 'type': 'private', 'display_name': '合成对象'},
        'participants': [{'id': 'self_test', 'role': 'me', 'name': '合成用户'},
                         {'id': 'peer_test', 'role': 'other', 'name': '合成对象'}],
        'messages': [
            {'id': 'm1', 'sender_id': 'self_test', 'time': '2026-10-28 09:00:00',
             'type': 'text', 'content': '测试第一行\n测试第二行'},
            {'id': 'm2', 'sender_id': 'peer_test', 'time': '2026-10-28 09:01:00',
             'type': 'quote', 'content': '收到\n引用：合成用户：测试第一行'},
        ],
    }


ALPHA_CHAT = (Path(__file__).parent / 'fixtures' / 'alpha_smoke.md').read_text(encoding='utf-8')


class ConversationFormatTests(unittest.TestCase):
    def load(self, data, suffix='.json'):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / ('chat' + suffix)
            raw = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            source.write_text(raw, encoding='utf-8-sig')
            return load_chat(str(source))

    def test_canonical_maps_roles_types_multiline_and_message_ids(self):
        normalized = self.load(canonical_chat())
        parsed = parse_chat_messages(normalized['messages'])['messages']
        self.assertEqual(normalized['session']['displayName'], '合成对象')
        self.assertEqual([m['sender'] for m in parsed], ['me', 'other'])
        self.assertEqual([m['source_message_id'] for m in parsed], ['m1', 'm2'])
        self.assertEqual(parsed[0]['content'], '测试第一行\n测试第二行')
        self.assertEqual(parsed[1]['type'], '引用消息')
        self.assertTrue(all(m['time_valid'] and m['timestamp_valid'] for m in parsed))

    def test_canonical_rejects_ambiguous_identity_version_and_invalid_messages(self):
        mutations = [
            lambda d: d.update(version=2),
            lambda d: d.update(version=True),
            lambda d: d['session'].update(type='group'),
            lambda d: d['session'].update(id='room@chatroom'),
            lambda d: d['session'].update(isGroup=True),
            lambda d: d.update(isGroupChat=True),
            lambda d: d['participants'].append({'id': 'third', 'role': 'other', 'name': '第三人'}),
            lambda d: d['participants'][1].update(id='self_test'),
            lambda d: d['participants'][1].update(role='me'),
            lambda d: d['messages'][0].update(sender_id='unknown'),
            lambda d: d['messages'][1].update(id='m1'),
            lambda d: d['messages'][0].update(time='2026-02-30 09:00:00'),
            lambda d: d['messages'][0].update(type='unknown'),
            lambda d: d['messages'][0].update(content={'text': 'invalid'}),
            lambda d: d['messages'][0].update(content=''),
        ]
        for mutate in mutations:
            data = canonical_chat()
            mutate(data)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.load(data)

    def test_alpha_markdown_preserves_quote_media_and_skips_system(self):
        normalized = self.load(ALPHA_CHAT, '.md')
        result = parse_chat_messages(normalized['messages'])
        self.assertEqual([m['sender'] for m in result['messages']], ['me', 'other', 'other'])
        self.assertEqual(result['messages'][0]['content'], '测试第一行\n测试第二行')
        self.assertIn('引用：合成用户', result['messages'][1]['content'])
        self.assertEqual(result['messages'][2]['content'], '[图片消息]')
        self.assertEqual(result['diagnostics']['skip_reasons'], {'system_message': 1})
        self.assertEqual(normalized['source_format'], 'alpha_markdown')
        self.assertEqual(normalized['messages'][0]['source_message_id'], 'alpha:1')

    def test_alpha_rejects_groups_missing_owner_multiple_senders_and_bad_headers(self):
        invalid = [
            ALPHA_CHAT.replace('`peer_test`', '`room@chatroom`'),
            ALPHA_CHAT.replace('- 导出账号：self\\_test\n', ''),
            ALPHA_CHAT.replace('**self\\_test**', '**无法确认的昵称**'),
            ALPHA_CHAT.replace('**合成对象** · 2026-10-28 09:02:00', '**第三人** · 2026-10-28 09:02:00'),
            ALPHA_CHAT.replace('2026-10-28 09:01:00', '2026-13-28 09:01:00'),
            ALPHA_CHAT.replace('**合成对象** · 2026-10-28 09:01:00', '**合成对象** · 时间缺失'),
            ALPHA_CHAT.replace('**合成对象** · 2026-10-28 09:01:00', '**合成对象** 2026-10-28 09:01:00'),
            ALPHA_CHAT.replace('· 图片', '· 未知类型'),
        ]
        for text in invalid:
            with self.subTest(text=text[:20]), self.assertRaises(ValueError):
                self.load(text, '.md')

    def test_weflow_remains_unchanged_and_unversioned_foreign_json_is_rejected(self):
        legacy = {'session': {'displayName': '合成对象'}, 'messages': [
            {'isSend': 1, 'formattedTime': 'bad time', 'type': '文本消息', 'content': '原有诊断'}]}
        self.assertEqual(self.load(legacy), legacy)
        wrong = copy.deepcopy(legacy)
        wrong['messages'][0]['type'] = 'text'
        with self.assertRaises(ValueError):
            self.load(wrong)

    def test_weflow_exporter_version_metadata_is_not_a_protocol_declaration(self):
        data = {'version': 'exporter-2026', 'messages': [
            {'isSend': 1, 'formattedTime': '2026-10-28 09:00:00',
             'type': '文本消息', 'content': '兼容已有导出元数据'}]}
        self.assertEqual(self.load(data), data)

    def test_canonical_empty_media_description_keeps_a_placeholder(self):
        data = canonical_chat()
        data['messages'][0].update(type='voice', content='')
        parsed = parse_chat_messages(self.load(data)['messages'])['messages']
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]['content'], '[语音消息]')

    def test_web_validates_canonical_without_mutating_input(self):
        data = canonical_chat()
        original = copy.deepcopy(data)
        normalized = validate_chat(data)
        self.assertEqual(data, original)
        self.assertEqual(normalized['messages'][0]['isSend'], 1)

    def test_web_import_stores_original_bytes_and_worker_reads_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = WebService(root)
            raw = ALPHA_CHAT.replace('\n', '\r\n').encode('utf-8-sig')
            row = service.import_chat('alpha.md', raw)
            source = service.chat_path(row['id'])
            self.assertEqual(source.suffix, '.md')
            self.assertEqual(source.read_bytes(), raw)
            with patch('web_app.threading.Thread.start'):
                job = service.start({'chat_id': row['id'], 'mode': 'preprocess'})
                service.cancel(job['id'])
                resumed = service.start({'retry_of': job['id']})
                self.assertEqual(resumed['id'], job['id'])
            folder = service.jobs_dir / job['id']
            self.assertEqual((folder / 'input.md').read_bytes(), raw)
            options = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
            with patch('llm_client.call_llm', side_effect=AssertionError('本地预处理不能调用模型')):
                result = run_job(root, folder, options, lambda event: None)
            self.assertEqual(result['stats']['total_messages'], 3)
            self.assertEqual(result['diagnostics']['raw_message_count'], 4)

    def test_invalid_alpha_writes_no_import_or_analysis_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = WebService(root)
            bad = ALPHA_CHAT.replace('`peer_test`', '`room@chatroom`')
            with self.assertRaises(ValueError):
                service.import_chat('group.md', bad.encode('utf-8'))
            self.assertEqual(service.conversations(), [])
            source = root / 'group.md'
            source.write_text(bad, encoding='utf-8')
            with self.assertRaises(ValueError):
                build_artifacts(source, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_full_workflow_discovers_json_and_alpha_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('a.json', 'b.md', 'c.txt', 'd.JSON'):
                (root / name).write_text('', encoding='utf-8')
            self.assertEqual([p.name for p in select_chat_files(root, [], None)],
                             ['a.json', 'b.md', 'd.JSON'])
            self.assertEqual([p.name for p in select_chat_files(root, ['b'], 1)], ['b.md'])

    def test_both_formats_complete_report_audit_and_reuse_without_new_calls(self):
        for suffix, raw, expected in [('.md', ALPHA_CHAT, 3),
                                      ('.json', json.dumps(canonical_chat()), 2)]:
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = root / 'job'
                folder.mkdir()
                input_name = 'input' + suffix
                (folder / input_name).write_text(raw, encoding='utf-8')
                config = SimpleNamespace(api_base='https://test.invalid', api_key='test-only', model='test')
                options = {'mode': 'full', 'input_name': input_name}
                chunk = json.dumps({key: [] for key in
                                    ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')})
                responses = [chunk, '# 合成报告\n<!-- END_FINAL_REPORT -->',
                             '# 合成侧写\n<!-- END_PERSONA_REPORT -->']
                with patch('web_worker.load_llm_config', return_value=config):
                    with patch('llm_client.call_llm', side_effect=responses) as llm:
                        result = run_job(root, folder, options, lambda event: None)
                    self.assertEqual(llm.call_count, 3)
                    self.assertEqual(result['stats']['total_messages'], expected)
                    self.assertEqual(len(result['files']), 9)
                    with patch('llm_client.call_llm', side_effect=AssertionError('不应重复请求')):
                        resumed = run_job(root, folder, {**options, 'resume': True}, lambda event: None)
                    self.assertEqual(result['files'], resumed['files'])


if __name__ == '__main__':
    unittest.main()
