"""LLM API 配置加载。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LLM_API_BASE = "https://token-plan-sgp.xiaomimimo.com/v1"
DEFAULT_LLM_MODEL = "mimo-v2.5pro"


@dataclass
class LLMConfig:
    api_base: str
    api_key: str
    model: str
    source: str


def load_env_file(env_path: Path) -> bool:
    """从 .env 文件读取环境变量，不覆盖已存在的变量。"""
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(('"', "'")) and value.endswith(('"', "'")) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)

    return True


def load_llm_config(workspace: Path) -> LLMConfig:
    """读取工作区根目录 .env 中的 LLM 配置。"""
    env_path = workspace / ".env"
    loaded = load_env_file(env_path)

    api_base = os.getenv("LLM_API_BASE", DEFAULT_LLM_API_BASE)
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    source = str(env_path) if loaded else "environment/defaults"

    return LLMConfig(
        api_base=api_base,
        api_key=api_key,
        model=model,
        source=source,
    )