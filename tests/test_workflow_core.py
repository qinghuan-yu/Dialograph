from __future__ import annotations

import http.client
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))

import analyze  # noqa: E402
import artifact_audit  # noqa: E402
import output_paths  # noqa: E402
import run_workflow  # noqa: E402


def message(index: int, content: str = "hello") -> dict:
    return {
        "sender": "me" if index % 2 else "other",
        "time_str": f"2024-01-01 00:{index:02d}:00",
        "timestamp": index,
        "content": content,
    }


def chunk_meta(index: int = 1) -> dict:
    return {
        "chunk_index": index,
        "start_message": 1,
        "end_message": 2,
        "start_time": "2024-01-01 00:01:00",
        "end_time": "2024-01-01 00:02:00",
    }


def complete_chunk_text() -> str:
    return """```json
{
  "schema_version": "chat-evidence-v1",
  "events": [],
  "relation_signals": [],
  "persona_signals": [],
  "counter_evidence": [],
  "uncertainties": []
}
```

## Analysis
Done.
<!-- END_CHUNK_ANALYSIS -->
"""


def complete_chunk_json_text() -> str:
    return """{
  "schema_version": "chat-evidence-v1",
  "events": [],
  "relation_signals": [],
  "persona_signals": [],
  "counter_evidence": [],
  "uncertainties": []
}"""


class AnalyzeParsingTests(unittest.TestCase):
    def test_parse_chat_messages_sorts_and_diagnoses_bad_time(self) -> None:
        raw = [
            {
                "type": "文本消息",
                "content": "later",
                "isSend": 1,
                "formattedTime": "bad time",
                "createTime": 2000,
            },
            {
                "type": "系统消息",
                "content": "",
                "isSend": 0,
                "formattedTime": "2024-01-01 00:00:00",
                "createTime": 0,
            },
            {
                "type": "文本消息",
                "content": "earlier",
                "isSend": 0,
                "formattedTime": "2024-01-01 00:00:01",
                "createTime": 1000,
            },
        ]

        result = analyze.parse_chat_messages(raw)

        self.assertEqual([item["content"] for item in result["messages"]], ["earlier", "later"])
        self.assertTrue(result["diagnostics"]["sort_changed"])
        self.assertEqual(result["diagnostics"]["skipped_message_count"], 1)
        self.assertEqual(result["diagnostics"]["invalid_time_message_count"], 1)


class ChunkingTests(unittest.TestCase):
    def test_chunk_prompt_requires_compact_json_only_output(self) -> None:
        prompt = run_workflow.build_chunk_prompt(
            {"other_name": "Other"},
            1,
            1,
            [message(1), message(2)],
        )

        self.assertIn("只输出一个合法 JSON 对象", prompt)
        self.assertIn("不要 Markdown", prompt)
        self.assertIn("events 3 条", prompt)
        self.assertNotIn("```", prompt)
        self.assertLessEqual(run_workflow.CHUNK_OUTPUT_TOKENS, 1200)

    def test_chunk_coverage_rejects_missing_tail(self) -> None:
        messages = [message(index) for index in range(1, 5)]

        with self.assertRaises(ValueError):
            run_workflow.validate_chunk_coverage(messages, [messages[:2]])

    def test_prompt_budget_split_preserves_order_and_coverage(self) -> None:
        messages = [message(index, "x" * 400) for index in range(1, 9)]
        initial = [messages]
        chunks = run_workflow.split_chunks_by_prompt_budget(
            {"other_name": "Other"},
            initial,
            max_prompt_bytes=3200,
        )

        run_workflow.validate_chunk_coverage(messages, chunks)
        flattened = [item for chunk in chunks for item in chunk]
        self.assertEqual(flattened, messages)
        self.assertGreater(len(chunks), 1)

    def test_final_prompt_is_budgeted_for_large_summaries(self) -> None:
        prompt = run_workflow.build_final_prompt(
            "Other",
            "s" * 20000,
            "c" * 20000,
            "e" * 30000,
            ["p" * 50000],
        )

        self.assertLess(len(prompt.encode("utf-8")), 30000)
        self.assertIn("内容已按提示词预算截断", prompt)
        self.assertEqual(run_workflow.FINAL_REPORT_MAX_OUTPUT_TOKENS, 4800)


class EvidenceCompletenessTests(unittest.TestCase):
    def test_normalize_evidence_and_completion_marker_accept_good_chunk(self) -> None:
        text = complete_chunk_text()
        evidence = run_workflow.normalize_chunk_evidence(text, chunk_meta())

        self.assertEqual(evidence["parse_status"], "ok")
        self.assertTrue(run_workflow.is_complete_chunk_evidence(evidence))

    def test_raw_json_chunk_is_complete_without_marker(self) -> None:
        text = complete_chunk_json_text()
        evidence = run_workflow.normalize_chunk_evidence(text, chunk_meta())

        self.assertEqual(evidence["parse_status"], "ok")
        self.assertTrue(run_workflow.is_complete_chunk_evidence(evidence))

    def test_missing_json_is_incomplete(self) -> None:
        no_json = "## Analysis\nOnly prose.\n<!-- END_CHUNK_ANALYSIS -->"

        evidence_without_json = run_workflow.normalize_chunk_evidence(no_json, chunk_meta())

        self.assertFalse(run_workflow.is_complete_chunk_evidence(evidence_without_json))

    def test_embedded_balanced_json_can_be_extracted(self) -> None:
        text = 'prefix {"schema_version":"chat-evidence-v1","events":[],"relation_signals":[],"persona_signals":[],"counter_evidence":[],"uncertainties":[]} suffix'
        evidence = run_workflow.normalize_chunk_evidence(text, chunk_meta())

        self.assertEqual(evidence["parse_status"], "ok")
        self.assertTrue(run_workflow.is_complete_chunk_evidence(evidence))

    def test_existing_fallback_chunk_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_paths = {
                "chunk_dir": root / "chunks",
                "evidence_dir": root / "evidence",
            }
            run_paths["chunk_dir"].mkdir()
            run_paths["evidence_dir"].mkdir()
            (run_paths["chunk_dir"] / "chunk_001.md").write_text(
                "incomplete",
                encoding="utf-8",
            )
            (run_paths["evidence_dir"] / "evidence_001.json").write_text(
                json.dumps({"chunk_index": 1, "parse_status": "fallback"}),
                encoding="utf-8",
            )

            self.assertIsNone(run_workflow.load_existing_chunk_result(run_paths, 1))

    def test_timeout_is_retryable(self) -> None:
        self.assertTrue(run_workflow.should_retry_with_smaller_prompt(TimeoutError("timed out")))

    def test_generate_chunk_analysis_returns_successful_token_budget(self) -> None:
        original_call = run_workflow.call_llm_with_retry_result
        try:
            def fake_call(*args, **kwargs):
                return run_workflow.LLMRetryResult(
                    content=complete_chunk_json_text(),
                    max_tokens_used=4800,
                )

            run_workflow.call_llm_with_retry_result = fake_call
            _summary, evidence, used_max_tokens = run_workflow.generate_chunk_analysis(
                object(),
                "prompt",
                10,
                chunk_meta(),
                initial_max_tokens=1200,
            )
        finally:
            run_workflow.call_llm_with_retry_result = original_call

        self.assertEqual(evidence["parse_status"], "ok")
        self.assertEqual(used_max_tokens, 4800)

    def test_report_completion_marker_is_required(self) -> None:
        run_workflow.require_completion_marker("ok <!-- END_FINAL_REPORT -->", "<!-- END_FINAL_REPORT -->", "report")
        with self.assertRaises(run_workflow.IncompleteReportOutput):
            run_workflow.require_completion_marker("not done", "<!-- END_FINAL_REPORT -->", "report")


class MergeCheckpointTests(unittest.TestCase):
    def test_merge_summary_reuses_complete_cached_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            merge_dir = Path(tmp) / "merge"
            cached = "cached summary\n<!-- END_MERGE_SUMMARY -->"
            run_workflow.write_cached_merge_batch(merge_dir, 1, 1, cached)

            original_limit = run_workflow.SUMMARY_MAX_BYTES
            original_call = run_workflow.call_llm_with_retry_result
            calls = []
            try:
                run_workflow.SUMMARY_MAX_BYTES = 10

                def fake_call(*args, **kwargs):
                    calls.append((args, kwargs))
                    return run_workflow.LLMRetryResult(
                        content="new summary\n<!-- END_MERGE_SUMMARY -->",
                        max_tokens_used=run_workflow.MERGE_MAX_OUTPUT_TOKENS,
                    )

                run_workflow.call_llm_with_retry_result = fake_call
                result = run_workflow.merge_summaries_if_needed(
                    "name",
                    ["a" * 100, "b" * 100],
                    "skill",
                    object(),
                    1,
                    merge_dir=merge_dir,
                )
            finally:
                run_workflow.SUMMARY_MAX_BYTES = original_limit
                run_workflow.call_llm_with_retry_result = original_call

            self.assertEqual(result, [cached])
            self.assertEqual(calls, [])

    def test_merge_summary_ignores_incomplete_cached_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            merge_dir = Path(tmp) / "merge"
            run_workflow.write_cached_merge_batch(merge_dir, 1, 1, "incomplete")

            original_limit = run_workflow.SUMMARY_MAX_BYTES
            original_call = run_workflow.call_llm_with_retry_result
            calls = []
            try:
                run_workflow.SUMMARY_MAX_BYTES = 10

                def fake_call(*args, **kwargs):
                    calls.append((args, kwargs))
                    return run_workflow.LLMRetryResult(
                        content="new summary\n<!-- END_MERGE_SUMMARY -->",
                        max_tokens_used=run_workflow.MERGE_MAX_OUTPUT_TOKENS,
                    )

                run_workflow.call_llm_with_retry_result = fake_call
                result = run_workflow.merge_summaries_if_needed(
                    "name",
                    ["a" * 100, "b" * 100],
                    "skill",
                    object(),
                    1,
                    merge_dir=merge_dir,
                )
            finally:
                run_workflow.SUMMARY_MAX_BYTES = original_limit
                run_workflow.call_llm_with_retry_result = original_call

            self.assertEqual(result, ["new summary\n<!-- END_MERGE_SUMMARY -->"])
            self.assertEqual(len(calls), 1)

    def test_merge_summary_reuses_successful_token_budget(self) -> None:
        original_limit = run_workflow.SUMMARY_MAX_BYTES
        original_call = run_workflow.call_llm_with_retry_result
        calls = []
        try:
            run_workflow.SUMMARY_MAX_BYTES = 10

            def fake_call(*args, **kwargs):
                calls.append((args, kwargs))
                return run_workflow.LLMRetryResult(
                    content=f"summary {len(calls)}\n<!-- END_MERGE_SUMMARY -->",
                    max_tokens_used=6400 if len(calls) == 1 else kwargs["max_tokens"],
                )

            run_workflow.call_llm_with_retry_result = fake_call
            result = run_workflow.merge_summaries_if_needed(
                "name",
                ["a" * 25000, "b" * 25000],
                "skill",
                object(),
                1,
            )
        finally:
            run_workflow.SUMMARY_MAX_BYTES = original_limit
            run_workflow.call_llm_with_retry_result = original_call

        self.assertGreaterEqual(len(result), 1)
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["max_tokens"], run_workflow.MERGE_MAX_OUTPUT_TOKENS)
        self.assertEqual(calls[1][1]["max_tokens"], 6400)


class AutoShrinkTests(unittest.TestCase):
    def test_auto_shrink_does_not_restart_non_chunk_retryable_errors(self) -> None:
        original_workflow = run_workflow.run_single_workflow
        calls = []
        try:
            def fake_workflow(**kwargs):
                calls.append(kwargs)
                raise http.client.RemoteDisconnected("final report disconnected")

            run_workflow.run_single_workflow = fake_workflow
            with self.assertRaises(http.client.RemoteDisconnected):
                run_workflow.run_single_workflow_with_auto_shrink(
                    artifacts={},
                    skill_prompt="skill",
                    config=object(),
                    timeout=10,
                )
        finally:
            run_workflow.run_single_workflow = original_workflow

        self.assertEqual(len(calls), 1)

    def test_auto_shrink_restarts_chunk_stage_failures(self) -> None:
        original_workflow = run_workflow.run_single_workflow
        calls = []
        try:
            def fake_workflow(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise run_workflow.RetryableChunkStageFailure("chunk disconnected")
                return {"status": "success"}

            run_workflow.run_single_workflow = fake_workflow
            result = run_workflow.run_single_workflow_with_auto_shrink(
                artifacts={},
                skill_prompt="skill",
                config=object(),
                timeout=10,
            )
        finally:
            run_workflow.run_single_workflow = original_workflow

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(calls), 2)
        self.assertLess(calls[1]["chunk_size"], calls[0]["chunk_size"])


class ArtifactAuditTests(unittest.TestCase):
    def make_artifacts(self, root: Path) -> dict:
        support = root / "临时文件" / "Other"
        chunk_dir = support / "分块分析"
        evidence_dir = support / "结构化证据"
        chunk_dir.mkdir(parents=True)
        evidence_dir.mkdir()
        return {
            "name": "Other",
            "safe_name": "Other",
            "message_count": 2,
            "analysis_file": root / "分析_Other.md",
            "persona_file": root / "人物侧写_Other.md",
            "phase_summary_file": support / "阶段总结_Other.md",
            "coverage_summary_file": support / "全量覆盖说明_Other.md",
            "evidence_summary_file": support / "结构化证据摘要_Other.md",
            "chunk_manifest_file": support / "分块覆盖清单_Other.json",
            "evidence_ledger_file": support / "结构化证据总表_Other.json",
            "chunk_dir": chunk_dir,
            "evidence_dir": evidence_dir,
        }

    def write_complete_artifacts(self, artifacts: dict) -> None:
        artifacts["analysis_file"].write_text("report\n<!-- END_FINAL_REPORT -->\n", encoding="utf-8")
        artifacts["persona_file"].write_text("persona\n<!-- END_PERSONA_REPORT -->\n", encoding="utf-8")
        artifacts["phase_summary_file"].write_text("phase\n", encoding="utf-8")
        artifacts["coverage_summary_file"].write_text("coverage\n", encoding="utf-8")
        artifacts["evidence_summary_file"].write_text("evidence summary\n", encoding="utf-8")
        artifacts["chunk_manifest_file"].write_text(
            json.dumps([
                {"chunk_index": 1, "message_count": 2, "start_message": 1, "end_message": 2}
            ], ensure_ascii=False),
            encoding="utf-8",
        )
        evidence = {
            "schema_version": "chat-evidence-v1",
            "chunk_count": 1,
            "documents": [{
                "chunk_index": 1,
                "parse_status": "ok",
                "events": [],
                "relation_signals": [],
                "persona_signals": [],
                "counter_evidence": [],
                "uncertainties": [],
            }],
        }
        artifacts["evidence_ledger_file"].write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
        (artifacts["chunk_dir"] / "chunk_001.md").write_text("{}\n", encoding="utf-8")
        (artifacts["evidence_dir"] / "evidence_001.json").write_text("{}\n", encoding="utf-8")

    def test_artifact_audit_accepts_complete_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.make_artifacts(Path(tmp))
            self.write_complete_artifacts(artifacts)

            result = artifact_audit.audit_artifacts(artifacts, require_persona=True)

        self.assertTrue(result.ok, result.errors)

    def test_artifact_audit_rejects_missing_report_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.make_artifacts(Path(tmp))
            self.write_complete_artifacts(artifacts)
            artifacts["analysis_file"].write_text("truncated report\n", encoding="utf-8")

            result = artifact_audit.audit_artifacts(artifacts, require_persona=True)

        self.assertFalse(result.ok)
        self.assertTrue(any("完成标记" in error for error in result.errors))

    def test_artifact_audit_rejects_fallback_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.make_artifacts(Path(tmp))
            self.write_complete_artifacts(artifacts)
            ledger = json.loads(artifacts["evidence_ledger_file"].read_text(encoding="utf-8"))
            ledger["documents"][0]["parse_status"] = "fallback"
            artifacts["evidence_ledger_file"].write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")

            result = artifact_audit.audit_artifacts(artifacts, require_persona=True)

        self.assertFalse(result.ok)
        self.assertTrue(any("不可复用分块" in error for error in result.errors))


class PathTests(unittest.TestCase):
    def test_sanitize_filename_removes_unsafe_characters(self) -> None:
        self.assertEqual(output_paths.sanitize_filename('a<b>c:"/\\|?*.'), "a_b_c_______")
        self.assertEqual(output_paths.sanitize_filename("..."), "unknown")


if __name__ == "__main__":
    unittest.main()
