from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def build_source_note(source: str, script_name: str, model: str | None = None, note: str | None = None) -> str:
    if source == "api":
        parts = ["由 API 生成"]
        if model:
            parts.append(f"模型 {model}")
    else:
        parts = ["人工整理 fallback"]

    parts.append(f"脚本/来源 {script_name}")
    if note:
        parts.append(note)
    return "> 来源说明：" + "，".join(parts) + "。"


def write_report(
    report_path: Path,
    metadata_path: Path,
    content: str,
    *,
    source: str,
    script_name: str,
    model: str | None = None,
    note: str | None = None,
    extra_metadata: dict | None = None,
) -> None:
    normalized = content.strip() + "\n"
    report_text = build_source_note(source, script_name, model=model, note=note) + "\n\n" + normalized
    report_path.write_text(report_text, encoding="utf-8")

    metadata = {
        "source": source,
        "script": script_name,
        "model": model,
        "note": note,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_file": str(report_path),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
