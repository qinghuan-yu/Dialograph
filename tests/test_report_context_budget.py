"""证据预算与连续原文回填；仅使用合成消息。"""
import copy
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
import prompt_budget
import report_generation
from analyze import parse_chat_messages
from evidence_grounding import ground_document


def evidence_rows(count=40):
    return ['- E_' + f'{n:032x}' + f' / 对方 / 2026-01-{n % 28 + 1:02d}: '
            + ('模型 / support: 支持' if n % 2 else '模型 / against: 反证') + f'条目{n}'
            for n in range(1, count + 1)]


def ledger():
    messages = parse_chat_messages([
        {'type': '文本消息', 'isSend': n % 2, 'formattedTime': f'2026-01-01 10:{n:02d}:00', 'createTime': 1767261600 + n * 60,
         'content': f'连续消息{n}'} for n in range(40)])['messages']
    doc = {'chunk_index': 1, 'message_range': {'start': 1, 'end': 40},
           'time_range': {'start': messages[0]['time_str'], 'end': messages[-1]['time_str']},
           **{key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')}}
    doc['counter_evidence'] = [{'against': '待检查的解释', 'evidence': messages[20]['content'],
                                'source_message_ids': [messages[20]['message_id']], 'confidence': 'medium'}]
    doc['events'] = [{'summary': '邻近锚点', 'evidence': messages[22]['content'],
                      'source_message_ids': [messages[22]['message_id']], 'confidence': 'high'}]
    bound = ground_document(doc, messages)
    return {'source_id': messages[0]['source_id'], 'source_messages': messages, 'documents': [bound]}


class SelectionBudgetTests(unittest.TestCase):
    def test_one_slot_selects_middle_instead_of_last(self):
        self.assertEqual(prompt_budget.evenly_spaced(list(range(9)), 1), [4])

    def test_evidence_packs_whole_rows_by_actual_bytes_and_balances_directions(self):
        rows = evidence_rows()
        selected = prompt_budget.budget_evidence_summary('## 结构化证据摘要\n### 关系建模证据\n' + '\n'.join(rows), 2000)
        ids = re.findall(r'E_[0-9a-f]{32}', selected)
        self.assertGreater(len(ids), 12)
        support = sum(row in selected for row in rows if '/ support:' in row)
        against = sum(row in selected for row in rows if '/ against:' in row)
        self.assertLessEqual(abs(support - against), 1)
        self.assertLessEqual(len(selected.encode('utf-8')), 2000)
        self.assertIn('时间轴', selected)
        for line in selected.splitlines():
            if line.startswith('- E_'):
                self.assertIn(line, rows)

    def test_oversized_evidence_is_omitted_instead_of_cutting_quote_or_id(self):
        long_row = evidence_rows(1)[0] + '原话' * 1000
        selected = prompt_budget.budget_evidence_summary('## 结构化证据摘要\n### 事件证据\n' + long_row, 700)
        self.assertNotIn('E_', selected)
        self.assertIn('省略', selected)
        self.assertLessEqual(len(selected.encode('utf-8')), 700)

    def test_merged_batches_keep_tables_paragraphs_and_numbered_items(self):
        table = '| 支持 | 反证 |\n|---|---|\n| 私人分享 | 尚无长期依据 |'
        text = ('## 阶段总结（批次 1/2）\n### 关系\n' + table
                + '\n\n正文说明需要结合连续对话。\n\n1. 编号事项。\n\n- 支持：共同兴趣。\n- 反证：有静默期。\n'
                + '## 阶段总结（批次 2/2）\n### 后续\n' + '过长段落' * 2000)
        selected = prompt_budget.budget_phase_summaries([text], 1800)
        self.assertIn(table, selected)
        self.assertIn('正文说明需要结合连续对话。', selected)
        self.assertIn('1. 编号事项。', selected)
        self.assertIn('批次 1/2', selected)
        self.assertIn('批次 2/2', selected)
        self.assertIn('省略', selected)
        self.assertLessEqual(len(selected.encode('utf-8')), 1800)


class SourceContextTests(unittest.TestCase):
    def build(self, text, data, **options):
        from report_context import build_evidence_context
        return build_evidence_context(text, data, **options)

    def test_selected_anchor_recovers_both_speakers_and_nearby_continuous_messages(self):
        data = ledger()
        evidence_id = data['documents'][0]['counter_evidence'][0]['evidence_id']
        before = copy.deepcopy(data)
        result = self.build(evidence_id, data, max_bytes=9000)
        for n in range(4, 37):
            self.assertIn(f'连续消息{n}\n', result)
        self.assertIn('【我】', result)
        self.assertIn('【对方】', result)
        self.assertIn('待复核', result)
        self.assertEqual(data, before)

    def test_overlapping_windows_are_not_repeated(self):
        data = ledger()
        ids = [data['documents'][0][key][0]['evidence_id'] for key in ('counter_evidence', 'events')]
        result = self.build(' '.join(ids), data, max_bytes=9000)
        self.assertEqual(result.count('连续消息21\n'), 1)
        self.assertTrue(all(eid in result for eid in ids))

    def test_context_is_bounded_and_does_not_cross_a_session_gap(self):
        data = ledger()
        messages = data['source_messages']
        for message in messages[21:]:
            message['timestamp'] += 86400
        evidence_id = data['documents'][0]['counter_evidence'][0]['evidence_id']
        result = self.build(evidence_id, data, max_bytes=5000)
        self.assertIn('连续消息20\n', result)
        self.assertNotIn('连续消息21\n', result)
        self.assertLessEqual(len(result.encode('utf-8')), 5000)

    def test_quarantined_and_unknown_ids_cannot_recover_quote_context(self):
        data = ledger()
        item = data['documents'][0]['counter_evidence'][0]
        item['grounding']['status'] = 'ambiguous'
        result = self.build(item['evidence_id'] + ' E_' + 'f' * 32, data, max_bytes=2000)
        self.assertNotIn('连续消息20', result)
        self.assertIn('未回填', result)

    def test_context_rejects_source_anchor_mismatch(self):
        data = ledger()
        item = data['documents'][0]['counter_evidence'][0]
        item['grounding']['source_refs'][0]['start'] += 1
        with self.assertRaisesRegex(ValueError, '原文绑定'):
            self.build(item['evidence_id'], data, max_bytes=4000)

    def test_context_rejects_other_snapshot_and_reports_whole_window_omissions(self):
        data = ledger()
        eid = data['documents'][0]['counter_evidence'][0]['evidence_id']
        result = self.build(eid, data, max_bytes=700)
        self.assertIn('未回填 1', result)
        self.assertNotIn('连续消息20', result)
        data['source_id'] = 'another_snapshot'
        with self.assertRaisesRegex(ValueError, '原文绑定'):
            self.build(eid, data, max_bytes=9000)

    def test_nearby_selected_anchors_across_session_gap_stay_in_separate_windows(self):
        data = ledger()
        for message in data['source_messages'][21:]:
            message['timestamp'] += 86400
        ids = [data['documents'][0][key][0]['evidence_id'] for key in ('counter_evidence', 'events')]
        result = self.build(' '.join(ids), data, max_bytes=9000)
        self.assertEqual(result.count('### 连续原文'), 2)

    def test_both_report_prompts_include_context_and_treat_extraction_as_candidate(self):
        data = ledger()
        item = data['documents'][0]['counter_evidence'][0]
        summary = '## 结构化证据摘要\n### 反证与边界\n- ' + item['evidence_id'] + ': 初步反证'
        final = report_generation.build_final_prompt('合成', '统计', '覆盖', summary, [], ledger=data)
        persona = report_generation.build_persona_prompt('合成', '统计', '覆盖', summary, [], '关系报告', ledger=data)
        for prompt in (final, persona):
            self.assertIn('连续消息19', prompt)
            self.assertIn('连续消息21', prompt)
            self.assertIn('待复核', prompt)
            self.assertIn('不能把省略', prompt)
        self.assertIn('二手解释', persona)

    def test_candidate_event_unit_deduplicates_same_message_and_separates_physical_metrics_from_candidate_labels(self):
        from report_context import select_evidence_context
        messages = parse_chat_messages([
            {'type': '文本消息', 'isSend': n % 2, 'formattedTime': f'2026-01-01 10:{n:02d}:00',
             'createTime': 1767261600 + n * 60, 'content': f'会话A消息{n}'} for n in range(15)
        ] + [
            {'type': '文本消息', 'isSend': n % 2, 'formattedTime': f'2026-01-02 10:{n:02d}:00',
             'createTime': 1767348000 + n * 60, 'content': f'会话B消息{n}'} for n in range(25)
        ])['messages']
        # 会话A：同一条消息绑定3个高置信度条目（物理锚点应去重为 1）
        # 会话B：3条不同消息组成起因—转折—结果（低置信度，但物理锚点为 3 且含方向对立）
        doc = {
            'chunk_index': 1,
            'parse_status': 'ok',
            'message_range': {'start': 1, 'end': 40},
            'time_range': {'start': messages[0]['time_str'], 'end': messages[-1]['time_str']},
            'events': [
                {'summary': '同源事件', 'evidence': messages[4]['content'], 'source_message_ids': [messages[4]['message_id']], 'confidence': 'high'},
                {'summary': '进展起因', 'evidence': messages[20]['content'], 'source_message_ids': [messages[20]['message_id']], 'confidence': 'low'},
            ],
            'relation_signals': [
                {'model': '普通朋友', 'direction': 'support', 'signal': '同源信号', 'evidence': messages[4]['content'], 'source_message_ids': [messages[4]['message_id']], 'confidence': 'high'},
                {'model': '较亲近朋友', 'direction': 'mixed', 'signal': '进展转折', 'evidence': messages[24]['content'], 'source_message_ids': [messages[24]['message_id']], 'confidence': 'low'},
            ],
            'persona_signals': [
                {'trait': '直接', 'signal': '同源画像', 'evidence': messages[4]['content'], 'stability': 'local', 'source_message_ids': [messages[4]['message_id']], 'confidence': 'high'},
            ],
            'counter_evidence': [
                {'against': '过度推断', 'evidence': messages[28]['content'], 'source_message_ids': [messages[28]['message_id']], 'confidence': 'low'},
            ],
            'uncertainties': [
                {'question': '存疑问题', 'reason': '缺少外部信息'},
            ],
        }
        bound = ground_document(doc, messages)
        data = {'source_id': messages[0]['source_id'], 'source_messages': messages, 'documents': [bound]}
        ctx = select_evidence_context('', data, select_from_ledger=True)
        self.assertEqual(len(ctx.candidate_units), 2)
        unit_a, unit_b = ctx.candidate_units
        self.assertEqual(unit_a.physical_metrics['anchor_count'], 1)
        self.assertEqual(len(unit_a.evidence_ids), 3)
        self.assertEqual(unit_b.physical_metrics['anchor_count'], 3)
        self.assertTrue(unit_b.candidate_labels['has_direction_contrast'])
        self.assertNotIn('confidence', unit_b.physical_metrics)
        self.assertNotIn('confidence', unit_b.candidate_labels)

    def test_unified_selection_recovers_late_chunk_progression_into_both_summary_and_context(self):
        from chunk_evidence import format_evidence_summary
        messages = parse_chat_messages([
            {'type': '文本消息', 'isSend': n % 2, 'formattedTime': f'2026-01-01 10:{n % 60:02d}:00',
             'createTime': 1767261600 + n * 60 + (86400 if n >= 30 else 0),
             'content': f'消息内容_{n}'} for n in range(60)
        ])['messages']
        doc1 = {
            'chunk_index': 1, 'parse_status': 'ok',
            'message_range': {'start': 1, 'end': 30},
            'time_range': {'start': messages[0]['time_str'], 'end': messages[29]['time_str']},
            'events': [{'summary': f'早期事件{i}', 'evidence': messages[i]['content'], 'source_message_ids': [messages[i]['message_id']], 'confidence': 'high'} for i in range(1, 6, 2)],
            'relation_signals': [], 'persona_signals': [], 'counter_evidence': [], 'uncertainties': [],
        }
        doc2 = {
            'chunk_index': 2, 'parse_status': 'ok',
            'message_range': {'start': 31, 'end': 60},
            'time_range': {'start': messages[30]['time_str'], 'end': messages[59]['time_str']},
            'events': [{'summary': '后期起因', 'evidence': messages[35]['content'], 'source_message_ids': [messages[35]['message_id']], 'confidence': 'low'}],
            'relation_signals': [
                {'model': '较亲近朋友', 'direction': 'support', 'signal': '后期初段', 'evidence': messages[37]['content'], 'source_message_ids': [messages[37]['message_id']], 'confidence': 'low'},
                {'model': '较亲近朋友', 'direction': 'support', 'signal': '我愿意去尝试理解你', 'evidence': messages[41]['content'], 'source_message_ids': [messages[41]['message_id']], 'confidence': 'low'},
            ],
            'persona_signals': [{'trait': '回应方式', 'signal': '后期回应', 'evidence': messages[39]['content'], 'stability': 'local', 'source_message_ids': [messages[39]['message_id']], 'confidence': 'low'}],
            'counter_evidence': [{'against': '冷漠预设', 'evidence': messages[43]['content'], 'source_message_ids': [messages[43]['message_id']], 'confidence': 'low'}],
            'uncertainties': [{'question': '后续是否持续', 'reason': '待观察'}],
        }
        docs = [ground_document(doc1, messages[:30]), ground_document(doc2, messages[30:])]
        data = {'source_id': messages[0]['source_id'], 'source_messages': messages, 'documents': docs}
        raw_summary = format_evidence_summary(docs)
        prepared = report_generation.prepare_final_input('合成对象', '统计', '覆盖', raw_summary, [], ledger=data)
        target_eid = docs[1]['relation_signals'][1]['evidence_id']
        self.assertIn(target_eid, prepared.context_ids)
        self.assertIn(target_eid, prepared.evidence_ids)
        self.assertIn('我愿意去尝试理解你', prepared.prompt)
        self.assertTrue(prepared.context_ids <= prepared.evidence_ids)

    def test_over_budget_long_candidate_unit_marks_narrowed_range_and_preserves_short_interactions(self):
        from report_context import select_evidence_context
        messages = parse_chat_messages([
            {'type': '文本消息', 'isSend': n % 2, 'formattedTime': f'2026-01-01 10:{n // 60:02d}:{n % 60:02d}',
             'createTime': 1767261600 + n * 40, 'content': f'长会话消息_{n}_' + '内容' * 15} for n in range(90)
        ] + [
            {'type': '文本消息', 'isSend': n % 2, 'formattedTime': f'2026-01-03 10:{n:02d}:00',
             'createTime': 1767434400 + n * 60, 'content': f'短互动消息_{n}'} for n in range(10)
        ])['messages']
        doc = {
            'chunk_index': 1, 'parse_status': 'ok',
            'message_range': {'start': 1, 'end': 100},
            'time_range': {'start': messages[0]['time_str'], 'end': messages[-1]['time_str']},
            'events': [
                {'summary': '起因节点', 'evidence': messages[5]['content'], 'source_message_ids': [messages[5]['message_id']], 'confidence': 'medium'},
                {'summary': '中间转折', 'evidence': messages[45]['content'], 'source_message_ids': [messages[45]['message_id']], 'confidence': 'medium'},
            ],
            'relation_signals': [
                {'model': '普通朋友', 'direction': 'support', 'signal': '初段支持', 'evidence': messages[7]['content'], 'source_message_ids': [messages[7]['message_id']], 'confidence': 'medium'},
                {'model': '普通朋友', 'direction': 'mixed', 'signal': '中段信号', 'evidence': messages[25]['content'], 'source_message_ids': [messages[25]['message_id']], 'confidence': 'medium'},
                {'model': '普通朋友', 'direction': 'support', 'signal': '短明确互动', 'evidence': messages[95]['content'], 'source_message_ids': [messages[95]['message_id']], 'confidence': 'medium'},
            ],
            'persona_signals': [
                {'trait': '特质', 'signal': '补救结果', 'evidence': messages[74]['content'], 'stability': 'local', 'source_message_ids': [messages[74]['message_id']], 'confidence': 'medium'},
            ],
            'counter_evidence': [
                {'against': '判断', 'evidence': messages[76]['content'], 'source_message_ids': [messages[76]['message_id']], 'confidence': 'medium'},
            ],
            'uncertainties': [
                {'question': '独立存疑条目', 'reason': '无事实锚点'},
            ],
        }
        bound = ground_document(doc, messages)
        data = {'source_id': messages[0]['source_id'], 'source_messages': messages, 'documents': [bound]}
        ctx = select_evidence_context('', data, max_bytes=7500, select_from_ledger=True)
        self.assertTrue(ctx.narrowed_units)
        self.assertIn('范围缩小', ctx.text)
        self.assertIn('不能当作完整事件', ctx.text)
        short_eid = bound['relation_signals'][2]['evidence_id']
        self.assertIn(short_eid, ctx.context_ids)
        uncert_eid = bound['uncertainties'][0]['evidence_id']
        self.assertIn(uncert_eid, ctx.uncertainties)
        self.assertNotIn(uncert_eid, ctx.context_ids)


if __name__ == '__main__':
    unittest.main()
