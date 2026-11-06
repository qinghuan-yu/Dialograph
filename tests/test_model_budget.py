"""用合成 API 响应验证用量账本与公共提示词前缀；不调用模型。"""
from claim_fixtures import claim_response
import io
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
import llm_client
from chunking import build_chunk_prompt
from summary_merge import build_merge_prompt


CONFIG = SimpleNamespace(api_base='https://test.invalid', api_key='secret-test-key', model='synthetic')


def response(content='合成响应', usage=None, finish='stop'):
    data = {'choices': [{'finish_reason': finish, 'message': {'content': content}}]}
    if usage is not None:
        data['usage'] = usage
    return io.BytesIO(json.dumps(data, ensure_ascii=False).encode('utf-8'))


class ModelUsageTests(unittest.TestCase):
    def test_incomplete_transport_retries_without_shrinking_input(self):
        failure = http.client.IncompleteRead(b'partial')
        with patch.object(llm_client, 'call_llm', side_effect=[failure, 'complete']) as api:
            result = llm_client.call_llm_with_retry_result(
                CONFIG, 's', 'u', 10, 24000, retry_sleep_seconds=0)
        self.assertEqual(result.content, 'complete')
        self.assertEqual(api.call_count, 2)
        self.assertEqual(api.call_args_list[0], api.call_args_list[1])
        self.assertFalse(llm_client.should_retry_with_smaller_prompt(failure))

    def test_high_output_budget_grows_on_truncation(self):
        with patch.object(llm_client, 'call_llm', side_effect=[
                llm_client.LLMOutputTruncated(), 'complete']) as api:
            result = llm_client.call_llm_with_retry_result(
                CONFIG, 's', 'u', 10, 24000, retry_sleep_seconds=0)
        self.assertEqual(result.max_tokens_used, 48000)
        self.assertEqual([call.kwargs['max_tokens'] for call in api.call_args_list], [24000, 48000])

    def test_high_marked_report_budget_grows_for_truncation_and_missing_marker(self):
        for first in (llm_client.LLMOutputTruncated(), llm_client.LLMRetryResult('partial', 24000)):
            with self.subTest(first=type(first).__name__), patch.object(
                    llm_client, 'call_llm', side_effect=[first, 'complete END']) as api:
                if isinstance(first, llm_client.LLMRetryResult):
                    api.side_effect = [first.content, 'complete END']
                result = llm_client.generate_marked_report(
                    CONFIG, 's', 'u', 10, 'END', 'report', 24000, retry_sleep_seconds=0)
            self.assertEqual(result, 'complete END')
            self.assertEqual([call.kwargs['max_tokens'] for call in api.call_args_list], [24000, 48000])

    def test_records_actual_usage_without_prompt_response_or_credentials(self):
        from llm_usage import usage_session, usage_operation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            usage = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
                     'prompt_cache_hit_tokens': 64, 'prompt_cache_miss_tokens': 36}
            with usage_session(root), usage_operation('extraction', chunk=3):
                with patch.object(llm_client.request, 'urlopen', return_value=response('私密响应', usage)):
                    self.assertEqual(llm_client.call_llm(CONFIG.api_base, CONFIG.api_key, CONFIG.model,
                                                       '私密规则', '私密对话', 10), '私密响应')
            raw = (root / 'model_usage.jsonl').read_text(encoding='utf-8')
            record = json.loads(raw)
            self.assertEqual(record['stage'], 'extraction')
            self.assertEqual(record['details'], {'chunk': 3})
            self.assertEqual(record['attempt'], 1)
            self.assertEqual(record['usage']['prompt_cache_hit_tokens'], 64)
            for value in ('私密规则', '私密对话', '私密响应', CONFIG.api_key, CONFIG.api_base):
                self.assertNotIn(value, raw)
            summary = json.loads((root / 'model_usage_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['totals']['requests'], 1)
            self.assertEqual(summary['totals']['prompt_tokens'], 100)
            self.assertEqual(summary['totals']['cache_hit_rate'], .64)

    def test_truncated_response_and_retry_are_both_counted(self):
        from llm_usage import usage_session, usage_operation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = [response('截断', {'prompt_tokens': 100, 'completion_tokens': 10}, 'length'),
                      response('完成', {'prompt_tokens': 100, 'completion_tokens': 20,
                                       'prompt_tokens_details': {'cached_tokens': 80}})]
            with usage_session(root), usage_operation('analysis'):
                with patch.object(llm_client.request, 'urlopen', side_effect=values):
                    result = llm_client.call_llm_with_retry_result(CONFIG, 's', 'u', 10, 100,
                                                                   max_attempts=2, retry_sleep_seconds=0)
            self.assertEqual(result.content, '完成')
            records = [json.loads(line) for line in (root / 'model_usage.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['attempt'] for r in records], [1, 2])
            self.assertEqual(records[0]['error_type'], 'LLMOutputTruncated')
            summary = json.loads((root / 'model_usage_summary.json').read_text(encoding='utf-8'))['totals']
            self.assertEqual(summary['prompt_tokens'], 200)
            self.assertEqual(summary['completion_tokens'], 30)
            self.assertEqual(summary['prompt_cache_miss_tokens'], 20)
            self.assertEqual(summary['cache_measured_requests'], 1)
            self.assertEqual(summary['cache_unknown_requests'], 1)

    def test_transport_failure_is_unknown_usage_and_preserves_original_error(self):
        from llm_usage import usage_session, usage_operation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with usage_session(root), usage_operation('merge', round=1, batch=2):
                with patch.object(llm_client.request, 'urlopen', side_effect=TimeoutError('私密异常内容')):
                    with self.assertRaisesRegex(TimeoutError, '私密异常内容'):
                        llm_client.call_llm(CONFIG.api_base, CONFIG.api_key, CONFIG.model, 's', 'u', 10)
            totals = json.loads((root / 'model_usage_summary.json').read_text(encoding='utf-8'))['totals']
            self.assertEqual(totals['requests'], 1)
            self.assertIsNone(totals['prompt_tokens'])
            self.assertIsNone(totals['completion_tokens'])
            self.assertIsNone(totals['cache_hit_rate'])
            self.assertEqual(totals['usage_unknown_requests'], 1)
            self.assertNotIn('私密异常内容', (root / 'model_usage.jsonl').read_text(encoding='utf-8'))

    def test_sessions_append_without_recounting_history_and_context_does_not_leak(self):
        from llm_usage import usage_session, usage_operation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stage in ('extraction', 'persona'):
                with usage_session(root), usage_operation(stage):
                    with patch.object(llm_client.request, 'urlopen', return_value=response(usage={'prompt_tokens': 10, 'completion_tokens': 5})):
                        llm_client.call_llm(CONFIG.api_base, CONFIG.api_key, CONFIG.model, 's', 'u', 10)
            summary = json.loads((root / 'model_usage_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['totals']['requests'], 2)
            self.assertEqual(summary['totals']['prompt_tokens'], 20)
            self.assertEqual(summary['latest_session']['totals']['requests'], 1)
            self.assertEqual(summary['stages']['extraction']['requests'], 1)
            self.assertEqual(summary['stages']['persona']['requests'], 1)
            before = (root / 'model_usage.jsonl').read_bytes()
            with patch.object(llm_client.request, 'urlopen', return_value=response()):
                llm_client.call_llm(CONFIG.api_base, CONFIG.api_key, CONFIG.model, 's', 'u', 10)
            self.assertEqual((root / 'model_usage.jsonl').read_bytes(), before)

    def test_invalid_usage_values_are_unknown_not_zero_or_estimated_from_max_tokens(self):
        from llm_usage import usage_session
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with usage_session(root), patch.object(llm_client.request, 'urlopen', return_value=response(usage={
                    'prompt_tokens': True, 'completion_tokens': -1, 'prompt_cache_hit_tokens': '50'})):
                llm_client.call_llm(CONFIG.api_base, CONFIG.api_key, CONFIG.model, 's', 'u', 10, max_tokens=8000)
            totals = json.loads((root / 'model_usage_summary.json').read_text(encoding='utf-8'))['totals']
            self.assertIsNone(totals['prompt_tokens'])
            self.assertIsNone(totals['completion_tokens'])
            self.assertIsNone(totals['prompt_cache_hit_tokens'])

    def test_full_web_workflow_records_stages_and_reuse_adds_no_api_requests(self):
        from test_conversation_formats import ALPHA_CHAT
        from web_worker import run_job
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'job'; folder.mkdir()
            (folder / 'input.md').write_text(ALPHA_CHAT, encoding='utf-8')
            chunk = json.dumps({key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')})
            usage = {'prompt_tokens': 100, 'completion_tokens': 10,
                     'prompt_cache_hit_tokens': 50, 'prompt_cache_miss_tokens': 50}
            review = json.dumps({'reviews':[{'claim_id':'R_1','verdict':'pass','reason':'合成有限判断。'}]})
            values = [response(chunk, usage), response(chunk, usage),
                      response('## 阶段总结\n合成材料\n<!-- END_MERGE_SUMMARY -->', usage),
                      response(claim_response('关系', report_kind='analysis'), usage),
                      response(review, usage),
                      response(claim_response('侧写', report_kind='persona'), usage), response(review, usage)]
            options = {'mode': 'full', 'input_name': 'input.md', 'chunk_size': 2}
            with patch('web_worker.load_llm_config', return_value=CONFIG), patch('summary_merge.SUMMARY_MAX_BYTES', 1), patch.object(llm_client.request, 'urlopen', side_effect=values):
                result = run_job(root, folder, options, lambda event: None)
            support = (folder / result['files']['stats']['path']).parent
            summary = json.loads((support / 'model_usage_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['totals']['requests'], 7)
            self.assertEqual({s: v['requests'] for s, v in summary['stages'].items()}, {'extraction': 2, 'merge': 1, 'analysis': 1, 'persona': 1, 'semantic_review': 2})
            with patch('web_worker.load_llm_config', return_value=CONFIG), patch.object(llm_client.request, 'urlopen', side_effect=AssertionError('完整复用不得请求模型')):
                run_job(root, folder, {**options, 'resume': True}, lambda event: None)
            after = json.loads((support / 'model_usage_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(after['totals']['requests'], 7)
            self.assertEqual(after['latest_session']['totals']['requests'], 0)


class CachePrefixTests(unittest.TestCase):
    def test_chunk_rules_and_schema_precede_all_variable_metadata(self):
        first = {'sender': 'me', 'time_str': '2026-01-01 10:00:00', 'content': '逐字原文\n🙂', 'message_id': 'M_' + '1' * 32}
        second = {**first, 'time_str': '2026-02-02 12:34:56', 'content': '另一条原文', 'message_id': 'M_' + '2' * 32}
        a = build_chunk_prompt({'other_name': '对象甲'}, 1, 8, [first])
        b = build_chunk_prompt({'other_name': '对象乙'}, 7, 9, [second])
        prefix = os.path.commonprefix([a, b])
        self.assertGreater(len(prefix.encode('utf-8')), 1800)
        for field in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties'):
            self.assertIn(field, prefix)
        self.assertIn('逐字原文\\n🙂', a)
        self.assertIn('[m1]', a)
        self.assertNotIn('[' + first['message_id'] + ']', a)
        self.assertIn('分块: 1/8', a)
        self.assertIn('分块: 7/9', b)

    def test_merge_rules_and_structure_precede_name_and_batch(self):
        a = build_merge_prompt('对象甲', 1, 8, ['第一段材料'])
        b = build_merge_prompt('对象乙', 2, 9, ['第二段材料'])
        prefix = os.path.commonprefix([a, b])
        self.assertGreater(len(prefix.encode('utf-8')), 800)
        self.assertIn('END_MERGE_SUMMARY', prefix)
        self.assertIn('阶段批次：1/8', a)
        self.assertIn('阶段批次：2/9', b)
        self.assertIn('第一段材料', a)

    def test_analysis_and_persona_share_unified_system_prompt_and_public_material_prefix(self):
        import report_generation
        from test_report_context_budget import ledger
        data = ledger()
        eid = data['documents'][0]['counter_evidence'][0]['evidence_id']
        summary = '## 结构化证据摘要\n- schema: chat-evidence-ledger-v1\n### 反证与边界\n- 总数 1；选取 1；省略 0。\n- ' + eid + ' / 对方 / 2026-01-01 10:20:00: 反驳 预设: 连续消息20'
        sys_a = report_generation.build_report_system_prompt('analysis skill', 'persona skill')
        sys_b = report_generation.build_report_system_prompt('analysis skill', 'persona skill')
        self.assertEqual(sys_a, sys_b)
        for keyword in ('来源核验', '说话人归属', '人物与关系判断边界', '对方/此人'):
            self.assertIn(keyword, sys_a)
        final_in = report_generation.prepare_final_input('合成对象', '统计块', '覆盖块', summary, ['阶段材料块'], ledger=data)
        persona_in = report_generation.prepare_persona_input('合成对象', '统计块', '覆盖块', summary, ['阶段材料块'], '关系报告线索', ledger=data)
        prefix = os.path.commonprefix([final_in.prompt, persona_in.prompt])
        for section in ('统计块', '覆盖块', '阶段材料块', '## 入选证据的连续原文', '## 本轮可用于解释性结论的完整窗口编号'):
            self.assertIn(section, prefix)
        self.assertEqual(final_in.evidence_ids, persona_in.evidence_ids)
        self.assertEqual(final_in.context_ids, persona_in.context_ids)
        self.assertIn('关系报告线索', persona_in.prompt[len(prefix):])

    def test_local_phase_aggregation_preserves_cross_time_recurrence_and_ids(self):
        from summary_merge import build_local_phase_summaries, compare_phase_summaries
        from test_report_context_budget import ledger
        data = ledger()
        doc1 = data['documents'][0]
        local_batches = build_local_phase_summaries('合成对象', [doc1], batch_size=4)
        self.assertEqual(len(local_batches), 1)
        self.assertIn('<!-- END_MERGE_SUMMARY -->', local_batches[0])
        self.assertIn(doc1['counter_evidence'][0]['evidence_id'], local_batches[0])
        comparison = compare_phase_summaries(local_batches, local_batches, [doc1])
        self.assertGreaterEqual(comparison['local_aggregation']['evidence_id_count'], 2)
        self.assertTrue(comparison['local_aggregation']['has_counter_section'])


if __name__ == '__main__':
    unittest.main()
