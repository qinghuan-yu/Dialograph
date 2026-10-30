"""批量运行聊天记录分析工作流。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import error


from analysis_artifacts import build_artifacts, select_chat_files
from analysis_service import analyze_conversation
from config import load_llm_config
from privacy_redaction import load_redaction_terms
from report_generation import load_skill_prompt
from chunking import (CHUNK_REQUEST_TIMEOUT, CHUNK_REQUEST_SLEEP_SECONDS,
                            CHUNK_MESSAGE_COUNT, CHUNK_MAX_TRANSCRIPT_BYTES,
                            CHUNK_MAX_PROMPT_BYTES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行聊天记录分析工作流")
    parser.add_argument('--redact-terms-file', type=Path, help='额外脱敏词：UTF-8逐行文本或JSON字符串数组')
    parser.add_argument(
        "targets",
        nargs="*",
        help="可选：只处理名称或文件名包含这些关键词的聊天记录",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 analysis/分析_{对象名}.md 已存在，也重新生成",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 个匹配到的聊天记录，便于小范围试跑",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="单个 LLM 请求超时时间，单位秒，默认 600",
    )
    parser.add_argument(
        "--chunk-timeout",
        type=int,
        default=CHUNK_REQUEST_TIMEOUT,
        help=f"单个分块 LLM 请求超时时间，单位秒，默认 {CHUNK_REQUEST_TIMEOUT}",
    )
    parser.add_argument(
        "--chunk-sleep",
        type=float,
        default=CHUNK_REQUEST_SLEEP_SECONDS,
        help=f"新分块请求成功后的冷却秒数，默认 {CHUNK_REQUEST_SLEEP_SECONDS}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_MESSAGE_COUNT,
        help=f"每个分块最多包含的消息数，默认 {CHUNK_MESSAGE_COUNT}",
    )
    parser.add_argument(
        "--chunk-max-bytes",
        type=int,
        default=CHUNK_MAX_TRANSCRIPT_BYTES,
        help=f"每个分块原文的近似最大字节数，默认 {CHUNK_MAX_TRANSCRIPT_BYTES}",
    )
    parser.add_argument(
        "--chunk-max-prompt-bytes",
        type=int,
        default=CHUNK_MAX_PROMPT_BYTES,
        help=f"每个分块实际提示词的近似最大字节数，默认 {CHUNK_MAX_PROMPT_BYTES}",
    )
    parser.add_argument(
        "--no-auto-shrink",
        action="store_true",
        help="关闭 LLM 连接断开/413/超时后的自动缩小分块重跑",
    )
    parser.add_argument(
        "--force-persona",
        action="store_true",
        help="即使人物侧写已存在，也重新生成人物侧写",
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        help="从 analysis/临时文件/{对象名}/runs/{run_id} 继续未完成的完整工作流；通常配合目标名称使用",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    redaction_terms = load_redaction_terms(args.redact_terms_file)
    workspace = Path(__file__).parent.parent
    talks_dir = workspace / "talks"
    output_dir = workspace / "analysis"
    output_dir.mkdir(exist_ok=True)

    config = load_llm_config(workspace)
    if not config.api_key:
        print("LLM_API_KEY 未设置，无法执行工作流分析。请先填写工作区根目录 .env")
        return 1

    skill_prompt = load_skill_prompt(Path(__file__).parent / "SKILL.md")
    persona_skill_prompt = load_skill_prompt(Path(__file__).parent / "PERSONA_SKILL.md")
    chat_files = select_chat_files(talks_dir, args.targets, args.limit)
    if not chat_files:
        print("没有匹配到要处理的聊天记录。")
        return 1

    print("=" * 60)
    print("聊天记录批量工作流分析")
    print("=" * 60)
    print(f"LLM_API_BASE: {config.api_base}")
    print(f"LLM_MODEL: {config.model}")
    print(f"LLM 配置来源: {config.source}")
    print(f"待处理聊天数: {len(chat_files)}")

    results = []
    for index, chat_file in enumerate(chat_files, start=1):
        print(f"\n[{index}/{len(chat_files)}] 处理 {chat_file.name}")
        try:
            initial_artifacts = build_artifacts(
                chat_file,
                output_dir,
                write_supporting_files=True,
                redaction_terms=redaction_terms,
            )
            result = analyze_conversation(
                initial_artifacts, skill_prompt, persona_skill_prompt, config,
                timeout=args.timeout, force=args.force, force_persona=args.force_persona,
                resume_run_id=args.resume_run, chunk_timeout=args.chunk_timeout,
                chunk_sleep=args.chunk_sleep, chunk_size=args.chunk_size,
                chunk_max_bytes=args.chunk_max_bytes,
                chunk_max_prompt_bytes=args.chunk_max_prompt_bytes,
                auto_shrink=not args.no_auto_shrink)
            print(f"- 已生成: {result['analysis_file']}")
            print(f"- 已生成: {result['persona_file']}")
            results.append(result)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"- HTTP 错误: {exc.code} {detail}")
            results.append({
                "name": chat_file.stem,
                "status": "failed",
                "error": f"HTTP {exc.code}: {detail}",
            })
        except Exception as exc:  # noqa: BLE001
            print(f"- 失败: {exc}")
            results.append({
                "name": chat_file.stem,
                "status": "failed",
                "error": str(exc),
            })

    summary_dir = output_dir / "临时文件"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "工作流批处理结果.json"
    summary_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    success_count = sum(1 for item in results if item["status"] == "success")
    skipped_count = sum(1 for item in results if item["status"] == "skipped")
    failed_count = sum(1 for item in results if item["status"] == "failed")

    print("\n" + "=" * 60)
    print("批量分析完成")
    print("=" * 60)
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {failed_count}")
    print(f"结果汇总: {summary_file}")
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
