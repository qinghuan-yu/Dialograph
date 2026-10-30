"""模型请求、错误分类、退避与报告完成校验。"""
from __future__ import annotations

import http.client
import json
import time
from dataclasses import dataclass
from datetime import datetime
from urllib import error, request

from privacy_redaction import current_redactor

class LLMOutputTruncated(RuntimeError):
    """Raised when the provider reports that output stopped because of token limits."""


class IncompleteReportOutput(RuntimeError):
    """Raised when a final report response does not include its completion marker."""


@dataclass(frozen=True)
class LLMRetryResult:
    content: str
    max_tokens_used: int


def call_llm(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    max_tokens: int = 2600,
) -> str:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"LLM 返回缺少 choices: {data}")

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message") or {}
    content = message.get("content")
    if finish_reason == "length":
        raise LLMOutputTruncated("LLM response was truncated by max_tokens")
    if isinstance(content, str) and content.strip():
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        text = "\n".join(part for part in parts if part).strip()
        if text:
            return text

    raise ValueError("LLM 返回缺少可用正文内容")


def should_retry_with_smaller_prompt(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, http.client.RemoteDisconnected):
        return True
    if isinstance(exc, error.HTTPError):
        return exc.code == 413
    if isinstance(exc, error.URLError):
        reason = str(exc.reason).lower()
        return "closed" in reason or "reset" in reason or "timed out" in reason
    return False


def should_retry_request(exc: Exception) -> bool:
    if isinstance(exc, error.HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    return should_retry_with_smaller_prompt(exc)


def retry_delay(exc: Exception, attempt: int, base_seconds: float) -> float:
    from email.utils import parsedate_to_datetime
    from datetime import timezone
    if isinstance(exc, error.HTTPError):
        value = exc.headers.get('Retry-After') if exc.headers else None
        if value:
            try:
                return max(0, float(value))
            except ValueError:
                try:
                    return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError, OverflowError):
                    # Retry-After 非法时使用下方的指数退避，不能让坏响应头阻断重试。
                    pass
    return base_seconds * 2 ** (attempt - 1)


def call_llm_with_retry_result(
    config,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    max_tokens: int,
    max_attempts: int = 3,
    retry_sleep_seconds: float = 3.0,
) -> LLMRetryResult:
    current_max_tokens = max_tokens
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(f"- 同请求重试: 第 {attempt} 次")
            print(f"- LLM 请求参数: max_tokens={current_max_tokens}, timeout={timeout}s")
            content = call_llm(
                api_base=config.api_base,
                api_key=config.api_key,
                model=config.model,
                system_prompt=current_redactor().redact(system_prompt),
                user_prompt=current_redactor().redact(user_prompt),
                timeout=timeout,
                max_tokens=current_max_tokens,
            )
            return LLMRetryResult(content=current_redactor().response(content), max_tokens_used=current_max_tokens)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, LLMOutputTruncated) and attempt < max_attempts:
                current_max_tokens = min(current_max_tokens * 2, 8000)
                print(f"- 输出被 max_tokens 截断，提升输出上限后重试: {current_max_tokens}")
                time.sleep(retry_sleep_seconds)
                continue
            if should_retry_request(exc) and attempt < max_attempts:
                print(f"- 请求失败，准备重试: {current_redactor().redact(str(exc))}")
                time.sleep(retry_delay(exc, attempt, retry_sleep_seconds))
                continue
            raise
    raise RuntimeError("LLM 调用失败")


def call_llm_with_retry(
    config,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    max_tokens: int,
    max_attempts: int = 3,
    retry_sleep_seconds: float = 3.0,
) -> str:
    return call_llm_with_retry_result(
        config,
        system_prompt,
        user_prompt,
        timeout,
        max_tokens,
        max_attempts=max_attempts,
        retry_sleep_seconds=retry_sleep_seconds,
    ).content


def generate_marked_report(
    config,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    marker: str,
    label: str,
    initial_max_tokens: int,
    max_attempts: int = 3,
    retry_sleep_seconds: float = 3.0,
) -> str:
    max_tokens = initial_max_tokens
    last_report = ""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = call_llm_with_retry_result(
                config,
                system_prompt,
                user_prompt,
                timeout,
                max_tokens=max_tokens,
                max_attempts=1,
                retry_sleep_seconds=retry_sleep_seconds,
            )
        except LLMOutputTruncated as exc:
            last_error = exc
            if attempt < max_attempts:
                max_tokens = min(max_tokens * 2, 8000)
                print(f"- {label}输出被 max_tokens 截断，提升输出上限后重试: {max_tokens}")
                time.sleep(retry_sleep_seconds)
                continue
            raise
        except Exception as exc:
            retryable = should_retry_request(exc)
            if retryable and attempt < max_attempts:
                print(f'- {label}遇到瞬时请求错误，重试同一提示词。')
                time.sleep(retry_delay(exc, attempt, retry_sleep_seconds))
                continue
            raise
        last_report = result.content
        if marker in last_report:
            return last_report
        if attempt < max_attempts:
            max_tokens = min(max_tokens * 2, 8000)
            print(f"- {label}缺少完成标记，提升输出上限后重试: {max_tokens}")
            time.sleep(retry_sleep_seconds)

    if last_error is not None and not last_report:
        raise last_error
    require_completion_marker(last_report, marker, label)
    return last_report


def require_completion_marker(text: str, marker: str, label: str) -> None:
    if marker not in text:
        raise IncompleteReportOutput(f"{label} 缺少完成标记 {marker}，疑似输出被截断")
