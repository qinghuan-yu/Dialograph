from __future__ import annotations

import re
from pathlib import Path


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name)
    cleaned = cleaned.strip().rstrip(".")
    return cleaned or "unknown"


def get_support_output_dir(output_dir: Path, safe_name: str) -> Path:
    return output_dir / "临时文件" / safe_name