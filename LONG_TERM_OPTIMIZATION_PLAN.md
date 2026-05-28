# Long-Term Optimization Plan

This file is the project memory for ongoing optimization work. Before starting any optimization, read this file first. At the end of each optimization pass, report progress against this plan and update the status table when the work changes the plan.

## Current Goal

Build a reliable local workflow for long-form chat analysis that can:

- read complete conversations through chunked processing;
- produce evidence-grounded relationship analysis;
- produce independent persona modeling;
- preserve evidence traceability;
- support future RAG retrieval over events, relationship signals, persona signals, counter-evidence, and uncertainties.

## Operating Rule For Future Optimization

1. Read this file before making workflow changes.
2. State which plan item is being worked on.
3. Keep generated private data out of Git unless the user explicitly asks otherwise.
4. After implementation, report:
   - what changed;
   - which plan item moved forward;
   - what was verified;
   - what remains next.
5. Update this file if an item is completed, re-scoped, or a new important item appears.

## Progress Snapshot

| Area | Status | Notes |
| --- | --- | --- |
| Git repository setup | Done | Repository initialized. `.gitignore` excludes secrets, virtualenv, raw chats, generated analysis, and local CodeGraph DB files. |
| Full-conversation coverage | Done | Workflow chunks all valid messages and writes a coverage manifest plus coverage summary. |
| Persona modeling | Done | Persona report uses full coverage summary, phase summaries, relationship report, and structured evidence summary. |
| Structured evidence layer | Done | Chunk prompts request JSON evidence; workflow extracts per-chunk evidence and writes a total evidence ledger plus summary. |
| CodeGraph index | Done | `.codegraph/` initialized and indexed for code navigation. Local DB files are ignored. |
| Data validation hardening | Done | Parsed messages are sorted by timestamp, original order is preserved, skipped messages are counted, and malformed time fields are reported without crashing stats. |
| Topic/session modeling | Done | Statistics now include 4-hour gap-based sessions, session initiators/closers, session size/duration, and in-session reply latency. |
| Failure recovery | Planned | Add run-id directories, resume support, and atomic promotion of successful runs. |
| RAG store | Planned | Import structured evidence into SQLite/FTS first; optionally add embeddings later. |
| Claim ledger | Planned | Require every final conclusion to bind to evidence IDs and counter-evidence IDs. |
| Test suite | Planned | Add unit tests for parsing, chunk coverage, evidence extraction, and path safety. |
| CLI cleanup | Planned | Clarify legacy single-pass scripts as preview-only or move them to a legacy folder. |

## Near-Term Queue

1. Add robust run management.
   - Write new analysis output under `analysis/临时文件/{safe_name}/runs/{run_id}/`.
   - Keep `latest` metadata pointing to the successful run.
   - Avoid deleting old chunks before a replacement run succeeds.
   - Support resume from completed chunk evidence.

2. Build local RAG foundation.
   - Create a SQLite database under ignored local data.
   - Ingest `结构化证据总表_*.json`.
   - Add FTS indexes over evidence, signal, model, trait, and uncertainty fields.
   - Add a query helper for relation/persona evidence retrieval.

3. Add conclusion claim ledger.
   - Final reports should include machine-readable claims.
   - Each claim should cite support evidence IDs, counter-evidence IDs, confidence, and uncertainty.
   - Use claim ledger to prevent unsupported final conclusions.

4. Add tests.
   - Unit-test filename sanitization and output path behavior.
   - Unit-test chunk coverage for all messages.
   - Unit-test JSON evidence extraction, fallback behavior, and summary truncation.
   - Unit-test malformed time handling.

## Design Principles

- Evidence first: final prose should never be more confident than its structured evidence.
- Privacy first: raw chats, `.env`, and generated analysis stay local and ignored by default.
- Recoverability: failed long runs should not destroy prior good outputs.
- Traceability: every conclusion should be traceable back to chunk, time range, message range, and evidence item.
- Incremental value: each optimization should leave the workflow more usable even before the whole roadmap is complete.
