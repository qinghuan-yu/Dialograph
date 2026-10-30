"""只用合成消息验证身份、引文锚定、隔离和缓存审计。"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from analyze import parse_chat_messages
import chunk_pipeline as workflow
import chunking
import workflow_state
import analysis_artifacts
import report_generation
import llm_client
from artifact_audit import AuditResult, audit_evidence_ledger, audit_artifacts
from web_worker import run_job

KEYS = ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')
CONFIG = SimpleNamespace(api_base='https://test.invalid', api_key='test-only', model='test')


def chat():
    return {'session': {'id': 'synthetic-peer', 'displayName': '合成对象'}, 'messages': [
        {'type': '文本消息', 'content': '明天再谈吧', 'isSend': 0,
         'formattedTime': '2026-10-28 10:02:00', 'createTime': 3},
        {'type': '系统消息', 'content': '系统提示', 'isSend': 0,
         'formattedTime': '2026-10-28 10:00:00', 'createTime': 1},
        {'type': '文本消息', 'content': '我可以帮忙\n周末有空', 'isSend': 1,
         'formattedTime': '2026-10-28 10:01:00', 'createTime': 2},
        {'type': '图片消息', 'content': None, 'isSend': 0,
         'formattedTime': '2026-10-28 10:03:00', 'createTime': 4},
    ]}


def evidence(**groups):
    return {**{key: [] for key in KEYS}, **groups}


def event(quote, summary='【事实】合成事件', **extra):
    return {'summary': summary, 'evidence': quote, 'confidence': 'low', **extra}


class MessageIdentityTests(unittest.TestCase):
    def test_ids_survive_repeated_parsing_sorting_and_retain_raw_positions(self):
        raw = chat()['messages']
        first = parse_chat_messages(raw)['messages']
        second = parse_chat_messages(copy.deepcopy(raw))['messages']
        self.assertTrue(all(m.get('message_id', '').startswith('M_') for m in first))
        self.assertEqual([m['message_id'] for m in first], [m['message_id'] for m in second])
        self.assertEqual([m['original_index'] for m in first], [2, 0, 3])
        self.assertEqual([m['position'] for m in first], [1, 2, 3])

    def test_identical_messages_remain_distinct_and_input_changes_identity(self):
        raw = [chat()['messages'][0]] * 2
        rows = parse_chat_messages(raw)['messages']
        self.assertEqual(len({m.get('message_id') for m in rows}), 2)
        changed = copy.deepcopy(raw)
        changed[0]['content'] = '改过的原文'
        newer = parse_chat_messages(changed)['messages']
        self.assertNotEqual(rows[0]['message_id'], newer[0]['message_id'])

    def test_session_identity_namespaces_identical_messages(self):
        raw = chat()['messages']
        first = parse_chat_messages(raw, {'id': 'synthetic-a'})['messages']
        second = parse_chat_messages(raw, {'id': 'synthetic-b'})['messages']
        self.assertNotEqual(first[0]['message_id'], second[0]['message_id'])


class GroundingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / 'input.json'
        source.write_text(json.dumps(chat(), ensure_ascii=False), encoding='utf-8')
        self.artifacts = analysis_artifacts.build_artifacts(source, self.root / 'out')

    def tearDown(self):
        self.temp.cleanup()

    def run_pipeline(self, payload, **options):
        with patch.object(llm_client, 'call_llm', return_value=json.dumps(payload, ensure_ascii=False)), \
             patch.object(workflow, 'generate_marked_report', return_value='报告 <!-- END_FINAL_REPORT -->'):
            workflow.run_single_workflow(self.artifacts, 'test skill', CONFIG, 5, settings=chunking.ChunkSettings(chunk_sleep=0, **options))
        return json.loads(self.artifacts['evidence_ledger_file'].read_text(encoding='utf-8'))

    def test_exact_quote_has_id_offsets_sender_and_original_position(self):
        ledger = self.run_pipeline(evidence(events=[event('可以帮忙\n周末', evidence_id='E_fake',
                                                         grounding={'status': 'unmatched'})]))
        item = ledger['documents'][0]['events'][0]
        self.assertTrue(item.get('evidence_id', '').startswith('E_'))
        self.assertNotEqual(item['evidence_id'], 'E_fake')
        self.assertEqual(item['grounding']['status'], 'matched')
        ref = item['grounding']['source_refs'][0]
        self.assertEqual((ref['original_index'], ref['position'], ref['sender']), (2, 1, 'me'))
        source = next(m for m in ledger['source_messages'] if m['message_id'] == ref['message_id'])
        self.assertEqual(source['content'][ref['start']:ref['end']], '可以帮忙\n周末')

    def test_unmatched_and_media_placeholders_are_excluded_from_all_report_material(self):
        payload = evidence(events=[event('明天再谈吧'), event('凭空编造的原话', '不应进入报告的解释'),
                                   event('[图片消息]', '不能把占位符当原话'), event('系统提示')])
        with patch.object(llm_client, 'call_llm', return_value=json.dumps(payload, ensure_ascii=False)), \
             patch.object(workflow, 'generate_marked_report', return_value='报告 <!-- END_FINAL_REPORT -->') as final:
            workflow.run_single_workflow(self.artifacts, 'skill', CONFIG, 5, settings=chunking.ChunkSettings(chunk_sleep=0))
        ledger = json.loads(self.artifacts['evidence_ledger_file'].read_text(encoding='utf-8'))
        statuses = [item.get('grounding', {}).get('status') for item in ledger['documents'][0]['events']]
        self.assertEqual(statuses, ['matched', 'unmatched', 'unmatched', 'unmatched'])
        prompt = final.call_args.args[2]
        self.assertIn('明天再谈吧', prompt)
        self.assertNotIn('凭空编造的原话', prompt)
        self.assertNotIn('不应进入报告的解释', prompt)
        self.assertNotIn('不能把占位符当原话', prompt)
        self.assertEqual(len(ledger['grounding_issues']), 3)

    def test_repeated_quote_requires_explicit_source_message_id(self):
        data = chat()
        data['messages'].append({**data['messages'][0], 'formattedTime': '2026-10-28 10:04:00', 'createTime': 5})
        self.artifacts['chat_file'].write_text(json.dumps(data), encoding='utf-8')
        self.artifacts = analysis_artifacts.build_artifacts(self.artifacts['chat_file'], self.root / 'out')
        ledger = self.run_pipeline(evidence(events=[event('明天再谈吧')]))
        item = ledger['documents'][0]['events'][0]
        self.assertEqual(item.get('grounding', {}).get('status'), 'ambiguous')
        message_id = self.artifacts['messages'][1]['message_id']
        ledger = self.run_pipeline(evidence(events=[event('明天再谈吧', source_message_ids=[message_id])]))
        item = ledger['documents'][0]['events'][0]
        self.assertEqual(item['grounding']['status'], 'matched')
        self.assertEqual(item['grounding']['source_refs'][0]['message_id'], message_id)

    def test_counter_evidence_is_bound_and_uncertainty_is_not_a_quote(self):
        ledger = self.run_pipeline(evidence(
            counter_evidence=[{'against': '没有帮助意愿', 'evidence': '我可以帮忙', 'confidence': 'medium'}],
            uncertainties=[{'question': '是否持续', 'reason': '缺少后续'}]))
        document = ledger['documents'][0]
        self.assertEqual(document['counter_evidence'][0].get('grounding', {}).get('status'), 'matched')
        self.assertEqual(document['uncertainties'][0]['grounding']['status'], 'not_applicable')
        self.assertTrue(document['uncertainties'][0]['evidence_id'].startswith('E_'))

    def test_evidence_ids_do_not_depend_on_run_item_order_or_chunk_size(self):
        items = [event('明天再谈吧'), event('我可以帮忙')]
        first = self.run_pipeline(evidence(events=items))
        second = self.run_pipeline(evidence(events=list(reversed(items))), chunk_size=2)
        def matched_ids(ledger):
            return {i['evidence']: i.get('evidence_id') for d in ledger['documents'] for i in d['events']
                    if i.get('grounding', {}).get('status') == 'matched'}
        self.assertEqual(len(matched_ids(first)), 2)
        self.assertTrue(all(matched_ids(first).values()))
        self.assertEqual(matched_ids(first), matched_ids(second))

    def test_audit_rejects_tampered_source_offsets_and_reports_unmatched_counts(self):
        ledger = self.run_pipeline(evidence(events=[event('明天再谈吧'), event('未发生的引文')]))
        clean = AuditResult('synthetic')
        audit_evidence_ledger(self.artifacts['evidence_ledger_file'], clean)
        self.assertTrue(clean.ok, clean.errors)
        self.assertTrue(any('未匹配' in warning for warning in clean.warnings))
        item = ledger['documents'][0]['events'][0]
        item['grounding']['source_refs'][0]['start'] = 999
        self.artifacts['evidence_ledger_file'].write_text(json.dumps(ledger), encoding='utf-8')
        corrupt = AuditResult('synthetic')
        audit_evidence_ledger(self.artifacts['evidence_ledger_file'], corrupt)
        self.assertFalse(corrupt.ok)

    def test_current_run_audit_rejects_a_ledger_downgraded_to_legacy(self):
        ledger = self.run_pipeline(evidence(events=[event('明天再谈吧')]))
        ledger['schema_version'] = 'chat-evidence-v1'
        del ledger['grounding_version']
        self.artifacts['evidence_ledger_file'].write_text(json.dumps(ledger), encoding='utf-8')
        self.assertFalse(audit_artifacts(self.artifacts).ok)

    def test_quote_hints_cannot_bind_another_chunk_or_override_literal_text(self):
        message_id = self.artifacts['messages'][0]['message_id']
        ledger = self.run_pipeline(evidence(events=[
            event('明天再谈吧', source_message_ids=[message_id]),
            event('我可以帮忙', source_message_ids=['M_missing']),
            event('我可以 帮忙'),
        ]), chunk_size=1)
        for document in ledger['documents']:
            self.assertEqual([i['grounding']['status'] for i in document['events']],
                             ['unmatched', 'unmatched', 'unmatched'])
        self.assertEqual(ledger['documents'][1]['events'][0]['grounding']['reason'], 'source_outside_chunk')

    def test_malformed_grounding_ledger_is_an_audit_error_without_crashing(self):
        ledger = self.run_pipeline(evidence(events=[event('明天再谈吧')]))
        mutations = [
            lambda d: d['documents'][0].update(message_range=[]),
            lambda d: d['documents'][0].update(message_range=None),
            lambda d: d.update(source_messages=[None]),
            lambda d: d.update(chunk_count=True),
        ]
        for mutate in mutations:
            broken = copy.deepcopy(ledger)
            mutate(broken)
            self.artifacts['evidence_ledger_file'].write_text(json.dumps(broken), encoding='utf-8')
            result = AuditResult('synthetic')
            audit_evidence_ledger(self.artifacts['evidence_ledger_file'], result)
            self.assertFalse(result.ok)

    def test_tampered_cached_anchor_is_rejected_before_metadata_or_api_changes(self):
        payload = evidence(events=[event('明天再谈吧')])
        with patch.object(llm_client, 'call_llm', return_value=json.dumps(payload)), \
             patch.object(workflow, 'generate_marked_report', side_effect=RuntimeError('stop final')):
            with self.assertRaisesRegex(RuntimeError, 'stop final'):
                workflow.run_single_workflow(self.artifacts, 'skill', CONFIG, 5, settings=chunking.ChunkSettings(chunk_sleep=0))
        run = next(self.artifacts['runs_dir'].iterdir())
        saved = (run / 'run_metadata.json').read_bytes()
        paths = workflow_state.build_run_paths(self.artifacts, run.name)
        path = paths['evidence_dir'] / 'evidence_001.json'
        document = json.loads(path.read_text(encoding='utf-8'))
        document['events'][0]['grounding']['source_refs'][0]['start'] = 99
        path.write_text(json.dumps(document), encoding='utf-8')
        with patch.object(llm_client, 'call_llm', side_effect=AssertionError('不能请求模型')):
            with self.assertRaisesRegex(ValueError, '绑定'):
                workflow.run_single_workflow(self.artifacts, 'skill', CONFIG, 5, resume_run_id=run.name)
        self.assertEqual((run / 'run_metadata.json').read_bytes(), saved)

    def test_reports_are_instructed_to_cite_ids_and_keep_uncertainty_separate(self):
        final = report_generation.build_final_prompt('合成对象', 'stats', 'coverage', 'evidence', ['phase'])
        persona = report_generation.build_persona_prompt('合成对象', 'stats', 'coverage', 'evidence', ['phase'], 'relation')
        for prompt in (final, persona):
            self.assertIn('引用 evidence_id', prompt)
            self.assertIn('存疑条目不是已证实事实', prompt)

    def test_web_alpha_and_canonical_outputs_expose_grounding_and_reuse_reports(self):
        fixtures = Path(__file__).parent / 'fixtures'
        for filename in ('alpha_smoke.md', 'canonical_smoke.json'):
            with self.subTest(filename=filename):
                folder = self.root / filename
                folder.mkdir()
                input_name = 'input' + Path(filename).suffix
                (folder / input_name).write_bytes((fixtures / filename).read_bytes())
                options = {'mode': 'full', 'input_name': input_name}
                payload = evidence(events=[event('测试第二行'), event('并不存在的合成引文')])
                responses = [json.dumps(payload), '报告 <!-- END_FINAL_REPORT -->',
                             '侧写 <!-- END_PERSONA_REPORT -->']
                with patch('web_worker.load_llm_config', return_value=CONFIG):
                    with patch.object(llm_client, 'call_llm', side_effect=responses):
                        result = run_job(self.root, folder, options, lambda event: None)
                    ledger = json.loads((folder / result['files']['evidence']['path']).read_text(encoding='utf-8'))
                    source = ledger['documents'][0]['events'][0]['grounding']['source_refs'][0]
                    self.assertIn('source_message_id', source)
                    self.assertEqual(len(ledger['grounding_issues']), 1)
                    self.assertTrue(any('未匹配' in warning for warning in result['warnings']))
                    with patch.object(llm_client, 'call_llm', side_effect=AssertionError('已完成报告不应重做')):
                        reused = run_job(self.root, folder, {**options, 'resume': True}, lambda event: None)
                    self.assertEqual(result['files'], reused['files'])


if __name__ == '__main__':
    unittest.main()
