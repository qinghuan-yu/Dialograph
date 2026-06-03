from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path


FINAL_MARKER = "<!-- END_FINAL_REPORT -->"
PERSONA_MARKER = "<!-- END_PERSONA_REPORT -->"


@dataclass
class AuditResult:
    name: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(result: AuditResult, key: str, path: Path) -> bool:
    if not path.exists():
        result.add_error(f"缺少{key}: {path.name}")
        return False
    if path.is_file() and path.stat().st_size == 0:
        result.add_error(f"{key}为空: {path.name}")
        return False
    return True


def has_marker(path: Path, marker: str) -> bool:
    return path.exists() and marker in path.read_text(encoding="utf-8", errors="ignore")


def audit_report(path: Path, marker: str, label: str, result: AuditResult) -> None:
    if not require_file(result, label, path):
        return
    if not has_marker(path, marker):
        result.add_error(f"{label}缺少完成标记: {marker}")


def audit_evidence_ledger(path: Path, result: AuditResult) -> int | None:
    if not require_file(result, "结构化证据总表", path):
        return None
    try:
        ledger = read_json(path)
    except json.JSONDecodeError as exc:
        result.add_error(f"结构化证据总表 JSON 无法解析: {exc}")
        return None

    documents = ledger.get("documents")
    if not isinstance(documents, list):
        result.add_error("结构化证据总表缺少 documents 数组")
        return None

    fallback_chunks = [
        doc.get("chunk_index")
        for doc in documents
        if doc.get("parse_status") not in {None, "ok"}
    ]
    if fallback_chunks:
        result.add_error(f"结构化证据包含不可复用分块: {fallback_chunks[:10]}")

    missing_keys = []
    required_keys = {"events", "relation_signals", "persona_signals", "counter_evidence", "uncertainties"}
    for doc in documents:
        absent = sorted(required_keys - set(doc))
        if absent:
            missing_keys.append({"chunk_index": doc.get("chunk_index"), "missing": absent})
    if missing_keys:
        result.add_error(f"结构化证据分块缺少字段: {missing_keys[:5]}")

    declared_count = ledger.get("chunk_count")
    if isinstance(declared_count, int) and declared_count != len(documents):
        result.add_error(f"结构化证据 chunk_count={declared_count} 但 documents={len(documents)}")

    result.stats["evidence_documents"] = len(documents)
    return len(documents)


def audit_manifest(path: Path, result: AuditResult, expected_messages: int | None) -> int | None:
    if not require_file(result, "分块覆盖清单", path):
        return None
    try:
        manifest = read_json(path)
    except json.JSONDecodeError as exc:
        result.add_error(f"分块覆盖清单 JSON 无法解析: {exc}")
        return None
    if not isinstance(manifest, list):
        result.add_error("分块覆盖清单不是数组")
        return None

    total_messages = sum(int(item.get("message_count", 0)) for item in manifest if isinstance(item, dict))
    if expected_messages is not None and total_messages != expected_messages:
        result.add_error(f"分块覆盖消息数 {total_messages} != 有效消息数 {expected_messages}")

    result.stats["manifest_chunks"] = len(manifest)
    result.stats["manifest_messages"] = total_messages
    return len(manifest)


def audit_artifacts(artifacts: dict, require_persona: bool = False) -> AuditResult:
    result = AuditResult(name=str(artifacts.get("name") or artifacts.get("safe_name") or "unknown"))

    audit_report(Path(artifacts["analysis_file"]), FINAL_MARKER, "关系分析报告", result)
    if require_persona:
        audit_report(Path(artifacts["persona_file"]), PERSONA_MARKER, "人物侧写报告", result)

    require_file(result, "阶段总结", Path(artifacts["phase_summary_file"]))
    require_file(result, "全量覆盖说明", Path(artifacts["coverage_summary_file"]))
    require_file(result, "结构化证据摘要", Path(artifacts["evidence_summary_file"]))

    manifest_count = audit_manifest(
        Path(artifacts["chunk_manifest_file"]),
        result,
        artifacts.get("message_count"),
    )
    evidence_count = audit_evidence_ledger(Path(artifacts["evidence_ledger_file"]), result)
    if manifest_count is not None and evidence_count is not None and manifest_count != evidence_count:
        result.add_error(f"分块覆盖清单数量 {manifest_count} != 结构化证据数量 {evidence_count}")

    chunk_dir = Path(artifacts["chunk_dir"])
    evidence_dir = Path(artifacts["evidence_dir"])
    if manifest_count is not None:
        chunk_files = list(chunk_dir.glob("chunk_*.md")) if chunk_dir.exists() else []
        evidence_files = list(evidence_dir.glob("evidence_*.json")) if evidence_dir.exists() else []
        if len(chunk_files) < manifest_count:
            result.add_error(f"分块分析文件数量 {len(chunk_files)} < 覆盖清单数量 {manifest_count}")
        if len(evidence_files) < manifest_count:
            result.add_error(f"分块证据文件数量 {len(evidence_files)} < 覆盖清单数量 {manifest_count}")
        result.stats["chunk_files"] = len(chunk_files)
        result.stats["evidence_files"] = len(evidence_files)

    return result


def audit_support_dir(support_dir: Path, output_dir: Path) -> AuditResult:
    safe_name = support_dir.name
    artifacts = {
        "name": safe_name,
        "safe_name": safe_name,
        "analysis_file": output_dir / f"分析_{safe_name}.md",
        "persona_file": output_dir / f"人物侧写_{safe_name}.md",
        "phase_summary_file": support_dir / f"阶段总结_{safe_name}.md",
        "coverage_summary_file": support_dir / f"全量覆盖说明_{safe_name}.md",
        "evidence_summary_file": support_dir / f"结构化证据摘要_{safe_name}.md",
        "chunk_manifest_file": support_dir / f"分块覆盖清单_{safe_name}.json",
        "evidence_ledger_file": support_dir / f"结构化证据总表_{safe_name}.json",
        "chunk_dir": support_dir / "分块分析",
        "evidence_dir": support_dir / "结构化证据",
        "message_count": None,
    }
    return audit_artifacts(artifacts, require_persona=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="审计已有聊天分析产物完整性")
    parser.add_argument("targets", nargs="*", help="可选：只审计名称包含这些关键词的对象")
    parser.add_argument("--json", action="store_true", help="输出 JSON 结果")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    output_dir = workspace / "analysis"
    support_root = output_dir / "临时文件"
    lowered = [target.lower() for target in args.targets]
    support_dirs = [
        path
        for path in sorted(support_root.iterdir())
        if path.is_dir() and (not lowered or any(target in path.name.lower() for target in lowered))
    ]
    results = [audit_support_dir(path, output_dir) for path in support_dirs]

    if args.json:
        print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            status = "OK" if result.ok else "FAIL"
            print(f"[{status}] {result.name}")
            for error in result.errors:
                print(f"  - ERROR: {error}")
            for warning in result.warnings:
                print(f"  - WARN: {warning}")
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
