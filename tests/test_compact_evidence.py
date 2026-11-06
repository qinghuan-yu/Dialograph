"""短来源编号可逆映射与引文发送者约束；只使用合成消息。"""
from claim_fixtures import claim_response, with_reviews
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from analyze import parse_chat_messages
import analysis_artifacts
import chunking
import chunk_evidence
import chunk_pipeline
import llm_client
from evidence_grounding import ground_document


KEYS = ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')


def messages():
    return parse_chat_messages([
        {'type': '文本消息', 'content': '这周安排很紧\n🙂先休息', 'isSend': 1, 'formattedTime': '2026-10-28 10:00:00'},
        {'type': '文本消息', 'content': '我愿意帮忙', 'isSend': 0, 'formattedTime': '2026-10-28 10:01:00'},
    ])['messages']


def document(**items):
    return {**{key: [] for key in KEYS}, **items, 'chunk_index': 1,
            'message_range': {'start': 1, 'end': 2},
            'time_range': {'start': '2026-10-28 10:00:00', 'end': '2026-10-28 10:01:00'}}


def event(quote, ids):
    return {'summary': '合成事件', 'evidence': quote, 'confidence': 'medium', 'source_message_ids': ids}


class CompactSourceTests(unittest.TestCase):
    def test_prompt_uses_local_short_ids_without_changing_time_sender_or_content(self):
        rows = messages()
        prompt = chunking.build_chunk_prompt({'other_name': '对方'}, 1, 2, rows)
        self.assertIn('=== 2026-10-28 ===', prompt)
        self.assertIn('[m1] 10:00:00 【我】 这周安排很紧\\n🙂先休息', prompt)
        self.assertIn('[m2] 10:01:00 【对方】 我愿意帮忙', prompt)
        expanded = chunking.expand_compact_transcript(prompt)
        self.assertEqual(
            expanded,
            [f'[m{index}] ' + chunking.format_message_line(row, '对方') for index, row in enumerate(rows, 1)],
        )
        for row in rows:
            self.assertNotIn(f"[{row['message_id']}]", prompt)
        self.assertIn('source_message_ids', prompt)
        self.assertIn('["m1"]', prompt)

    def test_shortening_headers_does_not_automatically_increase_messages_per_chunk(self):
        rows = messages() * 8
        compact_size = len(chunking.build_chunk_prompt({'other_name': '对方'}, 1, 1, rows).encode('utf-8'))
        chunks = chunking.split_chunks_by_prompt_budget({'other_name': '对方'}, [rows], compact_size + 1)
        self.assertGreater(len(chunks), 1, '分块规划继续预留完整编号预算，避免短编号导致证据条目配额下降')
        self.assertEqual([row for chunk in chunks for row in chunk], rows)

    def test_resolved_aliases_preserve_canonical_grounding_and_evidence_id(self):
        rows = messages()
        short = document(events=[event('我愿意帮忙', ['m2'])])
        canonical = document(events=[event('我愿意帮忙', [rows[1]['message_id']])])
        resolved = chunk_evidence.resolve_source_aliases(short, rows)
        self.assertEqual(short['events'][0]['source_message_ids'], ['m2'], '映射不修改模型输出对象')
        self.assertEqual(ground_document(resolved, rows), ground_document(canonical, rows))

    def test_unknown_malformed_and_other_chunk_aliases_remain_unmatched(self):
        rows = messages()
        for ids in (['m999'], ['m2', 'm1'], [None], {'id': 'm2'}, ['m02']):
            with self.subTest(ids=ids):
                result = chunk_evidence.resolve_source_aliases(document(events=[event('我愿意帮忙', ids)]), rows)
                self.assertEqual(ground_document(result, rows)['events'][0]['grounding']['status'], 'unmatched')
        first = chunk_evidence.resolve_source_aliases(document(events=[event(rows[0]['content'], ['m1'])]), rows)
        other_chunk = [rows[1]]
        second = chunk_evidence.resolve_source_aliases(document(events=[event(rows[1]['content'], ['m1'])]), other_chunk)
        self.assertEqual(first['events'][0]['source_message_ids'], [rows[0]['message_id']])
        self.assertEqual(second['events'][0]['source_message_ids'], [rows[1]['message_id']])

    def test_duplicate_quotes_can_be_disambiguated_by_short_id(self):
        rows = messages()
        rows[0]['content'] = rows[1]['content']
        payload = chunk_evidence.resolve_source_aliases(document(events=[event('我愿意帮忙', ['m2'])]), rows)
        item = ground_document(payload, rows)['events'][0]
        self.assertEqual(item['grounding']['status'], 'matched')
        self.assertEqual(item['grounding']['source_refs'][0]['sender'], 'other')

    def test_pipeline_persists_full_ids_and_reuses_cache_after_alias_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.json'
            raw = {'session': {'id': 'synthetic', 'displayName': '合成对象'}, 'messages': [
                {'type': '文本消息', 'content': '我愿意帮忙', 'isSend': 0, 'formattedTime': '2026-10-28 10:01:00'}]}
            source.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
            artifacts = analysis_artifacts.build_artifacts(source, root / 'output')
            payload = document(events=[event('我愿意帮忙', ['m1'])])
            config = SimpleNamespace(api_base='https://test.invalid', api_key='test', model='synthetic')
            with patch.object(llm_client, 'call_llm', return_value=json.dumps(payload)), patch.object(chunk_pipeline, 'call_llm_with_retry', side_effect=with_reviews([claim_response('报告', report_kind='analysis')])):
                result = chunk_pipeline.run_single_workflow(artifacts, 'skill', config, 10, settings=chunking.ChunkSettings(chunk_sleep=0))
            ledger = json.loads(artifacts['evidence_ledger_file'].read_text(encoding='utf-8'))
            sent = next(artifacts['runs_dir'].glob('*/report_input_analysis.md')).read_text(encoding='utf-8')
            self.assertIn('入选证据的连续原文', sent)
            self.assertIn('我愿意帮忙', sent)
            item = ledger['documents'][0]['events'][0]
            self.assertEqual(item['source_message_ids'], [artifacts['messages'][0]['message_id']])
            self.assertEqual(item['grounding']['source_refs'][0]['sender'], 'other')
            with patch.object(llm_client, 'call_llm', side_effect=AssertionError('应复用已绑定的分块')), patch.object(chunk_pipeline, 'call_llm_with_retry', side_effect=with_reviews([claim_response('报告', report_kind='analysis')])):
                chunk_pipeline.run_single_workflow(artifacts, 'skill', config, 10, settings=chunking.ChunkSettings(chunk_sleep=0), resume_run_id=result['run_id'])


class SpeakerSummaryTests(unittest.TestCase):
    def test_summary_keeps_verified_sender_and_excludes_user_quotes_from_peer_persona(self):
        rows = messages()
        payload = document(
            events=[event(rows[0]['content'], [rows[0]['message_id']])],
            persona_signals=[
                {'trait': '用户信号', 'signal': '仅用户自述', 'evidence': rows[0]['content'], 'stability': 'local', 'confidence': 'medium'},
                {'trait': '对方信号', 'signal': '对方表达意愿', 'evidence': rows[1]['content'], 'stability': 'local', 'confidence': 'medium'},
            ])
        grounded = ground_document(payload, rows)
        summary = chunk_evidence.format_evidence_summary([grounded])
        user_event = grounded['events'][0]['evidence_id']
        event_line = next(line for line in summary.splitlines() if user_event in line)
        self.assertIn(' / 我 / ', event_line)
        persona = summary.split('### 人物建模证据', 1)[1].split('### ', 1)[0]
        self.assertNotIn('仅用户自述', persona)
        self.assertNotIn(grounded['persona_signals'][0]['evidence_id'], persona)
        self.assertIn('对方表达意愿', persona)
        self.assertIn(' / 对方 / ', persona)
        self.assertEqual(len(grounded['persona_signals']), 2, '上下文条目仍保留于完整账本')


if __name__ == '__main__':
    unittest.main()
