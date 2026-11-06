"""审查阻断项回归：独立合成对话，不使用真实人名或日期案例。"""
import copy
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from analyze import parse_chat_messages
from evidence_grounding import ground_document
from report_context import select_evidence_context
import report_generation
import summary_merge


def scenario(anchors=(20, 30, 45), count=80, long_turn=False, width=0):
    start = datetime(2023, 2, 8, 9)
    messages = parse_chat_messages([
        {'type': '文本消息', 'isSend': 0 if long_turn else n % 2,
         'formattedTime': (start + timedelta(seconds=n)).strftime('%Y-%m-%d %H:%M:%S'),
         'content': f'节点正文{n:03d}。' + '材料' * width} for n in range(count)])['messages']
    doc = {'chunk_index': 1, 'parse_status': 'ok',
        'message_range': {'start': 1, 'end': count},
        'time_range': {'start': messages[0]['time_str'], 'end': messages[-1]['time_str']},
        **{key: [] for key in ('events', 'relation_signals', 'persona_signals', 'counter_evidence', 'uncertainties')}}
    doc['events'] = [{'summary': f'物理节点{n}', 'evidence': messages[n]['content'],
                     'confidence': 'low', 'source_message_ids': [messages[n]['message_id']]} for n in anchors]
    doc = ground_document(doc, messages)
    return {'source_id': messages[0]['source_id'], 'source_messages': messages, 'documents': [doc]}


class WindowContractTests(unittest.TestCase):
    def test_fast_alternating_dialogue_does_not_shrink_baseline_for_many_anchors(self):
        data = scenario()
        context = select_evidence_context('', data, select_from_ledger=True)
        for n in range(4, 62):
            self.assertIn(f'节点正文{n:03d}。', context.text)

    def test_boundary_turn_is_kept_whole_beyond_sixteen_messages(self):
        data = scenario((35,), long_turn=True)
        context = select_evidence_context('', data, select_from_ledger=True)
        self.assertIn('节点正文000。', context.text)
        self.assertIn('节点正文079。', context.text)

    def test_insufficient_budget_does_not_certify_a_shortened_window(self):
        data = scenario((20, 30, 45), width=30)
        context = select_evidence_context('', data, max_bytes=4500, select_from_ledger=True)
        self.assertFalse(context.context_ids)
        self.assertNotIn('核验引文：', context.text)
        self.assertIn('未回填', context.text)

    def test_partial_candidate_does_not_grant_inference_access_to_shown_fragments(self):
        data = scenario((5, 25, 45, 65), width=10)
        context = select_evidence_context('', data, max_bytes=7500, select_from_ledger=True)
        self.assertTrue(context.narrowed_units)
        self.assertFalse(context.context_ids)
        self.assertIn('局部出处（未列入完整窗口目录）', context.text)
        self.assertIn('整组编号不进入完整窗口目录', context.text)

    def test_late_response_and_result_are_selected_with_earlier_nodes(self):
        data = scenario((5, 25, 45, 65))
        context = select_evidence_context('', data, select_from_ledger=True)
        self.assertEqual(context.context_ids, {x['evidence_id'] for x in data['documents'][0]['events']})
        for n in (5, 25, 45, 65):
            self.assertIn(f'节点正文{n:03d}。', context.text)

    def test_distinct_episode_stages_survive_long_turns_and_pauses(self):
        start = datetime(2021, 7, 3, 8)
        content = ['日常补充。'] * 96
        stages = {3: '请求核对集合地点。', 27: '回应说集合地点临时改变。',
                  51: '追问并澄清两个地点混淆。', 81: '确认已经汇合，问题解决。'}
        for position, text in stages.items():
            content[position] = text
        messages = parse_chat_messages([{'type': '文本消息', 'isSend': (n // 4) % 2,
            'formattedTime': (start + timedelta(seconds=n * 10 + (3600 if n >= 48 else 0))).strftime('%Y-%m-%d %H:%M:%S'),
            'content': text} for n, text in enumerate(content)])['messages']
        doc = scenario()['documents'][0]
        doc['message_range'] = {'start': 1, 'end': len(messages)}
        doc['time_range'] = {'start': messages[0]['time_str'], 'end': messages[-1]['time_str']}
        doc['events'] = [{'summary': '待复核节点', 'evidence': text, 'confidence': 'low',
                         'source_message_ids': [messages[n]['message_id']]} for n, text in stages.items()]
        doc = ground_document(doc, messages)
        data = {'source_id': messages[0]['source_id'], 'source_messages': messages, 'documents': [doc]}
        context = select_evidence_context('', data, select_from_ledger=True)
        self.assertEqual(len(context.context_ids), 4)
        for text in stages.values():
            self.assertIn(text, context.text)

    def test_pruning_allowed_ids_also_removes_verified_labels(self):
        data = scenario()
        context = select_evidence_context('', data, select_from_ledger=True)
        wanted = {data['documents'][0]['events'][0]['evidence_id']}
        pruned = context.with_selected_ids(wanted)
        labels = set(re.findall(r'E_[0-9a-f]{32}', '\n'.join(re.findall(r'核验引文：([^\n]+)', pruned.text))))
        self.assertEqual(pruned.context_ids, wanted)
        self.assertEqual(labels, wanted)
        self.assertTrue(context.context_ids > wanted, '裁剪不能改写原选择对象')
        widened = pruned.with_selected_ids(context.evidence_ids)
        labels = set(re.findall(r'E_[0-9a-f]{32}', '\n'.join(re.findall(r'核验引文：([^\n]+)', widened.text))))
        self.assertEqual(labels, widened.context_ids)

    def test_model_category_and_direction_do_not_raise_unit_priority(self):
        from report_context import CandidateEventUnit, unit_priority
        metrics = {'anchor_count': 3, 'turn_count': 8, 'message_span': 40, 'duration_seconds': 100}
        plain = CandidateEventUnit('U1', (1, 5, 8), frozenset(), metrics, {'categories': ('events',), 'directions': ()})
        labelled = CandidateEventUnit('U1', (1, 5, 8), frozenset(), metrics,
            {'categories': ('events', 'relation_signals', 'persona_signals', 'counter_evidence'),
             'directions': ('support', 'against'), 'has_direction_contrast': True})
        self.assertEqual(unit_priority(plain), unit_priority(labelled))


class FrameworkAndExperimentTests(unittest.TestCase):
    def test_shared_system_includes_both_actual_frameworks(self):
        system = report_generation.build_report_system_prompt('关系框架标记', '人物框架标记')
        self.assertIn('关系框架标记', system)
        self.assertIn('人物框架标记', system)
        default = report_generation.build_report_system_prompt()
        root = Path(report_generation.__file__).parent
        self.assertIn((root / 'SKILL.md').read_text(encoding='utf-8').strip(), default)
        self.assertIn((root / 'PERSONA_SKILL.md').read_text(encoding='utf-8').strip(), default)

    def test_local_aggregation_renders_each_evidence_once(self):
        data = scenario()
        doc = data['documents'][0]
        # 复用合成引文，重新绑定合法关系及反证条目。
        doc['relation_signals'] = [{'model': '朋友', 'direction': 'support', 'signal': '待复核解释',
            'evidence': data['source_messages'][30]['content'], 'confidence': 'low'}]
        doc['counter_evidence'] = [{'against': '待复核命题', 'evidence': data['source_messages'][45]['content'], 'confidence': 'low'}]
        doc = ground_document(doc, data['source_messages'])
        text = '\n'.join(summary_merge.build_local_phase_summaries('合成', [doc]))
        for key in ('relation_signals', 'counter_evidence'):
            self.assertEqual(text.count(doc[key][0]['evidence_id']), 1)

    def test_id_counts_are_not_presented_as_semantic_retention(self):
        data = scenario()
        result = summary_merge.compare_phase_summaries(['早期有请求，随后追问，最后协商解决。'],
            summary_merge.build_local_phase_summaries('合成', data['documents']), data['documents'])
        self.assertEqual(result['comparison_scope'], 'format_observability_only')
        self.assertEqual(result['semantic_preservation'], 'not_evaluated')


if __name__ == '__main__':
    unittest.main()
