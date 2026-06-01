from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))

import analyze  # noqa: E402
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


class EvidenceCompletenessTests(unittest.TestCase):
    def test_normalize_evidence_and_completion_marker_accept_good_chunk(self) -> None:
        text = complete_chunk_text()
        evidence = run_workflow.normalize_chunk_evidence(text, chunk_meta())

        self.assertEqual(evidence["parse_status"], "ok")
        self.assertTrue(run_workflow.is_complete_chunk_output(text, evidence))

    def test_missing_json_or_marker_is_incomplete(self) -> None:
        no_json = "## Analysis\nOnly prose.\n<!-- END_CHUNK_ANALYSIS -->"
        no_marker = complete_chunk_text().replace("<!-- END_CHUNK_ANALYSIS -->", "")

        evidence_without_json = run_workflow.normalize_chunk_evidence(no_json, chunk_meta())
        evidence_without_marker = run_workflow.normalize_chunk_evidence(no_marker, chunk_meta())

        self.assertFalse(run_workflow.is_complete_chunk_output(no_json, evidence_without_json))
        self.assertFalse(run_workflow.is_complete_chunk_output(no_marker, evidence_without_marker))

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

    def test_report_completion_marker_is_required(self) -> None:
        run_workflow.require_completion_marker("ok <!-- END_FINAL_REPORT -->", "<!-- END_FINAL_REPORT -->", "report")
        with self.assertRaises(run_workflow.IncompleteReportOutput):
            run_workflow.require_completion_marker("not done", "<!-- END_FINAL_REPORT -->", "report")


class PathTests(unittest.TestCase):
    def test_sanitize_filename_removes_unsafe_characters(self) -> None:
        self.assertEqual(output_paths.sanitize_filename('a<b>c:"/\\|?*.'), "a_b_c_______")
        self.assertEqual(output_paths.sanitize_filename("..."), "unknown")


if __name__ == "__main__":
    unittest.main()
