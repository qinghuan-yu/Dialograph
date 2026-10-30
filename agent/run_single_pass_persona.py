"""单次生成人物侧写报告，不走分块解析。"""

from __future__ import annotations

import sys
from pathlib import Path


from preview_support import preview_materials, generate_preview
from analysis_types import AnalysisArtifacts
from config import load_llm_config
from report_output import write_report
from privacy_redaction import redact_artifact_call
from analysis_artifacts import build_artifacts, select_chat_files
from report_generation import load_skill_prompt

def build_single_pass_persona_prompt(
    name: str,
    stats_summary: str,
    segment_text: str,
    sample_text: str,
    relation_report: str,
) -> str:
    return f"""请基于以下统计摘要、代表性片段、补充文本样本和已经完成的关系分析报告，直接建立一份完整的人物侧写。

对象：{name}

重要要求：
1. 只输出最终人物侧写报告，不要解释你的工作过程。
2. 这不是关系分析的重复版，而是回答“这个人，在与你互动时，呈现出怎样稳定的人物形象”。
3. 严格区分【事实】【推断】【假设】【存疑】。
4. 每个重要判断都尽量绑定具体片段、时间点、语言风格或行为模式证据。
5. 优先写长期稳定模式，其次才写阶段性信号。
6. 不做读心，不做病理化诊断，不把单个事件夸大成整体人格。
7. 不预设对方性别、性取向、关系目标或恋爱/暧昧前提。
8. 除直接引用原文外，全文使用“对方/此人/该对象/对象昵称”，不要使用“他/她/ta/男方/女方”等带性别预设的代称。
9. 风格参考：证据支撑、结构完整、冷静克制。

请严格按以下结构输出 Markdown：

以下是基于你提供的聊天记录片段、统计数据以及关系分析报告，为“{name}”建立的完整人物侧写。

所有结论均标注证据层级，并区分可观察事实与合理推断。

---

# 人物侧写：{name}

## 一、基本画像
- 用表格列出：与我认识时间、互动总量、回复速度、深夜活跃度、表情包使用频率、主动性等。

## 二、性格与表达风格
### 2.1 语言风格
### 2.2 情绪表达
### 2.3 幽默感

## 三、互动模式与边界感
### 3.1 主动性与投入
### 3.2 边界感
### 3.3 面对边界变化、求助、合作与冲突的反应

## 四、动机与需求
- 用表格列出：动机假设 / 支持证据 / 反证 / 可信度

## 五、对方在我面前的倾向性总结
- 用表格列出：主动性、共情能力、边界感、情绪稳定性、关系投入方式、回避倾向、责任感、表达成熟度，以及证据强度。

## 六、不能判断的部分

## 七、对我而言最重要的认知

下面是统计摘要：

{stats_summary}

下面是 15 个代表性对话片段：

{segment_text}

下面是补充文本消息样本：

{sample_text}

下面是已经完成的关系分析报告：

{relation_report}
"""


def compact_relation_report(report: str, max_chars: int = 12000) -> str:
    stripped = report.strip()
    if len(stripped) <= max_chars:
        return stripped

    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return (
        stripped[:head_chars].rstrip()
        + "\n\n[中间部分为控制提示词长度已省略]\n\n"
        + stripped[-tail_chars:].lstrip()
    )


@redact_artifact_call(position=2)
def generate_persona_with_fallback(llm_config, persona_skill_prompt: str, artifacts: AnalysisArtifacts, relation_report: str) -> str:
    def prompt(compact=False):
        segments, samples = preview_materials(artifacts, compact=compact)
        return build_single_pass_persona_prompt(artifacts['name'], artifacts['stats_summary'], segments, samples,
            compact_relation_report(relation_report, max_chars=9000) if compact else relation_report)

    primary_prompt = prompt()
    print(f"单次提示词大小: {len(primary_prompt.encode('utf-8'))} bytes")
    print("正在调用模型生成人物侧写...")
    return generate_preview(llm_config, persona_skill_prompt, primary_prompt, lambda: prompt(compact=True))


def main() -> int:
    workspace = Path(__file__).parent.parent
    output_dir = workspace / "analysis" / "预览"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = sys.argv[1] if len(sys.argv) > 1 else "示例对象"

    llm_config = load_llm_config(workspace)
    if not llm_config.api_key:
        print("LLM_API_KEY 未设置，无法执行单次人物建模。")
        return 1

    talks = select_chat_files(workspace / "talks", [], None)
    target_file = None
    for talk in talks:
        if target in talk.stem:
            target_file = talk
            break

    if target_file is None:
        print(f"未找到匹配的聊天记录: {target}")
        return 1

    artifacts = build_artifacts(target_file, output_dir, write_supporting_files=True)
    if not artifacts["analysis_file"].exists():
        print(f"缺少关系分析文件: {artifacts['analysis_file']}")
        print("请先运行 run_single_pass_analysis.py 生成关系建模报告。")
        return 1

    persona_skill_prompt = load_skill_prompt(Path(__file__).parent / "PERSONA_SKILL.md")
    relation_report = artifacts["analysis_file"].read_text(encoding="utf-8")

    print("=" * 60)
    print("预览：单次人物侧写分析（采样分析，不覆盖正式报告）")
    print("=" * 60)
    print(f"对象: {artifacts['name']}")
    print(f"关系分析文件: {artifacts['analysis_file']}")

    report = generate_persona_with_fallback(
        llm_config,
        persona_skill_prompt,
        artifacts,
        relation_report,
    )
    write_report(
        artifacts["persona_file"],
        artifacts["persona_meta_file"],
        report,
        source="api",
        script_name="agent/run_single_pass_persona.py",
        model=llm_config.model,
        note="预览：仅使用代表性片段，不保证全量证据覆盖，不等同于完整人物侧写",
    )

    print(f"已生成: {artifacts['persona_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
