"""完整原文模式在明确预算内保留未命中的会话，超预算退回原检索。"""
from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))
from claim_retrieval import retrieve_claim_context
from config import load_llm_config
from test_claim_retrieval import source


QUESTIONS = [{'proposition': '合成命题', 'support_queries': ['愿意了解'],
              'challenge_queries': ['没有命中']}]


class CompleteReportContextTests(unittest.TestCase):
    def test_complete_mode_keeps_unqueried_sessions_when_source_fits(self):
        ledger = source(session_break=80)
        current = retrieve_claim_context(ledger, QUESTIONS)
        self.assertEqual(current.source_positions, frozenset(range(1, 81)))
        complete = retrieve_claim_context(ledger, QUESTIONS, prefer_complete=True)
        self.assertEqual(complete.source_positions, frozenset(range(1, 141)))
        self.assertEqual(complete.retrieval['selection_policy'], 'complete-source-if-fit-v1')
        self.assertEqual(len(complete.retrieval['session_decisions']), 2)
        self.assertLessEqual(len(complete.text.encode('utf-8')), 48000)

    def test_over_budget_complete_mode_keeps_the_existing_query_selection(self):
        ledger = source(session_break=80)
        selected = retrieve_claim_context(ledger, QUESTIONS)
        budget = len(selected.text.encode('utf-8')) + 40
        baseline = retrieve_claim_context(ledger, QUESTIONS, max_bytes=budget)
        fallback = retrieve_claim_context(ledger, QUESTIONS, max_bytes=budget, prefer_complete=True)
        self.assertEqual(fallback.source_positions, baseline.source_positions)
        self.assertEqual(fallback.text, baseline.text)
        self.assertEqual(fallback.retrieval['selection_policy'], 'complete-query-balanced-v3')

    def test_relation_and_persona_use_the_same_explicit_context_options(self):
        from report_generation import prepare_final_input, prepare_persona_input
        ledger = source(session_break=80)
        options = {'max_bytes': 48000, 'prefer_complete': True}
        args = ('合成', '', '', '', [])
        relation = prepare_final_input(*args, ledger=ledger, review_questions=QUESTIONS,
                                       context_options=options)
        persona = prepare_persona_input(*args, '', ledger=ledger, review_questions=QUESTIONS,
                                        context_options=options)
        self.assertEqual(relation.source_positions, frozenset(range(1, 141)))
        self.assertEqual(relation.retrieval, persona.retrieval)

    def test_context_configuration_changes_pipeline_but_not_input_identity(self):
        from config import report_context_options
        from workflow_state import run_identity
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            config = load_llm_config(Path(directory))
        self.assertEqual(report_context_options(config), {'max_bytes': 48000, 'prefer_complete': False})
        artifacts = {'chat_file': Path(__file__).parent / 'fixtures/web_smoke.json', 'messages': []}
        before = run_identity(artifacts, '合成', config)
        for changed in (replace(config, report_context_bytes=800000),
                        replace(config, report_context_mode='complete_if_fits')):
            after = run_identity(artifacts, '合成', changed)
            self.assertEqual(after['input_sha256'], before['input_sha256'])
            self.assertNotEqual(after['pipeline_sha256'], before['pipeline_sha256'])

    def test_environment_loads_explicit_context_settings_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            for values in ({'LLM_REPORT_CONTEXT_BYTES': '-1'}, {'LLM_REPORT_CONTEXT_BYTES': 'wrong'},
                           {'LLM_REPORT_CONTEXT_MODE': 'all'}):
                with self.subTest(values=values), patch.dict(os.environ, values, clear=True), self.assertRaises(ValueError):
                    load_llm_config(Path(directory))
            with patch.dict(os.environ, {'LLM_REPORT_CONTEXT_BYTES': '800000',
                                         'LLM_REPORT_CONTEXT_MODE': 'complete_if_fits'}, clear=True):
                config = load_llm_config(Path(directory))
            self.assertEqual(config.report_context_bytes, 800000)
            self.assertEqual(config.report_context_mode, 'complete_if_fits')
