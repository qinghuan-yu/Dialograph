"""报告上下文构建与经过审计的人物侧写发布。"""
from __future__ import annotations

import tempfile
from pathlib import Path
from artifact_audit import audit_artifacts
from llm_client import generate_marked_report
from report_output import write_report
from run_promotion import publish_files
from workflow_state import report_identity
from privacy_redaction import redact_artifact_call
from prompt_budget import (limit_text_bytes, budget_evidence_summary, budget_phase_summaries)

def load_evidence_summary(artifacts: dict) -> str:
    summary_file = artifacts.get("evidence_summary_file")
    if summary_file and summary_file.exists():
        return summary_file.read_text(encoding="utf-8").strip()
    return "未找到结构化证据摘要；请主要依据阶段总结和覆盖说明，并降低细粒度结论的确定性。"


FINAL_STATS_SUMMARY_MAX_BYTES = 5000
FINAL_EVIDENCE_SUMMARY_MAX_BYTES = 6000
FINAL_COVERAGE_SUMMARY_MAX_BYTES = 2500
FINAL_PHASE_SUMMARY_MAX_BYTES = 12000
FINAL_REPORT_MAX_OUTPUT_TOKENS = 4800
PERSONA_REPORT_MAX_OUTPUT_TOKENS = 4800


def build_final_prompt(
    name: str,
    stats_summary: str,
    coverage_summary: str,
    evidence_summary: str,
    phase_summaries: list[str],
) -> str:
    stats_summary = limit_text_bytes(stats_summary, FINAL_STATS_SUMMARY_MAX_BYTES)
    joined = budget_phase_summaries(phase_summaries, FINAL_PHASE_SUMMARY_MAX_BYTES)
    evidence_summary = budget_evidence_summary(evidence_summary, FINAL_EVIDENCE_SUMMARY_MAX_BYTES)
    coverage_summary = limit_text_bytes(coverage_summary, FINAL_COVERAGE_SUMMARY_MAX_BYTES)
    return f"""请基于以下全局统计、结构化证据摘要和阶段总结，生成一份克制、证据导向的最终分析报告。

对象：{name}

写作标准：
1. 参考法证式分析，不要写空话套话。
2. 每个大结论都要写支持证据、反证、边界条件。
3. 要明确区分长期模式、阶段性模式、一次性事件。
4. 可以细，但不能越界读心。
5. 若证据不足，明确写【存疑】。
6. 输出要信息密度高，不要展开成长篇散文；每节优先写最能支撑结论的 3 到 5 点。
7. 全量覆盖说明记录抽取阶段处理的范围；报告输入经过预算选择，省略内容不能当作已被模型审阅，不得声称逐条核验全部证据。
8. “结构化证据摘要”是从每个分块的 JSON 证据块汇总而来。最终结论应优先绑定这些证据条目，并同时检查反证与存疑。
   事实与推断须引用 evidence_id（E_...）；存疑条目不是已证实事实。不能以隔离条目或媒体占位符作为支持，逐字匹配也不代表解释成立。
9. 不预设对方性别、性取向、关系目标或恋爱/暧昧前提；亲密或暧昧只作为证据充分时的可选模型。
10. 除直接引用原文外，全文必须使用“对方/此人/该对象/对象昵称”称呼分析对象；不要使用“他/她/ta/男方/女方”等带性别预设的代称。

请严格按以下结构输出 Markdown：

# {name} 聊天关系分析报告

**重要声明：** 先写一段 2 到 4 句声明，说明本报告如何区分事实、推断、假设、不能判断的部分，并说明不会进行读心或迎合式结论。

## 一、互动概况
- 3 到 5 条，覆盖频率、主动性、话题控制、关键转折

## 二、对方的表达风格
- 3 到 5 条，覆盖语言风格、情绪表达、严肃/玩笑模式切换

## 三、对方的互动模式
- 3 到 5 条，覆盖关心/追问/记忆/回应脆弱表达/任务协作/边界

## 四、对方在这段关系中的可能定位
- 用表格评估最相关的 5 到 7 个模型，不必覆盖所有模型
- 每行必须包含：支持证据、反证、可信度

## 五、对方可能的动机逻辑
- 给出 3 种解释
- 每种都包含：支持证据、反证、可信度、还需要什么信息验证

## 六、对方在你面前呈现出的性格倾向
- 4 到 6 条，覆盖主动性、共情、边界、情绪稳定性、投入方式、回避倾向等

## 七、你自己的误判风险
- 3 到 5 条，覆盖稀缺性滤镜、投射、特殊关系放大、情境性互动误判、选择性注意等
- 要直接指出哪些判断可能站不住脚

## 八、总结
### 1. 最稳妥的结论
### 2. 中等可信的判断
### 3. 不能下结论的部分
### 4. 低风险验证关系类型或互动边界的建议

格式要求：
1. 全文使用【事实】【推断】【假设】【存疑】四类标记。
2. 每个章节给出概括性证据即可，不要长段复述原文。
3. 写得具体但受控，避免超过必要长度。
4. 最后一行必须是：<!-- END_FINAL_REPORT -->

以下是全局统计：

{stats_summary}

以下是全量覆盖说明：

{coverage_summary}

以下是结构化证据摘要：

{evidence_summary}

以下是按预算选择的阶段材料：

{joined}
"""


def build_persona_prompt(
    name: str,
    stats_summary: str,
    coverage_summary: str,
    evidence_summary: str,
    phase_summaries: list[str],
    relation_report: str,
) -> str:
    stats_summary = limit_text_bytes(stats_summary, FINAL_STATS_SUMMARY_MAX_BYTES)
    coverage_summary = limit_text_bytes(coverage_summary, FINAL_COVERAGE_SUMMARY_MAX_BYTES)
    evidence_summary = budget_evidence_summary(evidence_summary, FINAL_EVIDENCE_SUMMARY_MAX_BYTES)
    joined = budget_phase_summaries(phase_summaries, FINAL_PHASE_SUMMARY_MAX_BYTES)
    relation_report = limit_text_bytes(relation_report, 8000)
    return f"""请基于以下统计摘要、阶段总结和已经完成的关系分析报告，进一步建立对方的人物侧写。

对象：{name}

任务目标：
1. 输出一份独立的人物形象建模报告，而不是重复关系分析报告。
2. 重点回答“这个人，在与你互动时，呈现出怎样稳定的人物形象”。
3. 要有证据支撑，要区分【事实】【推断】【假设】【存疑】。
4. 要写出边界、互动风格、情绪表达、投入方式、回避方式、关系需求等层面。
5. 如果关系分析报告里的某个判断证据不够，要主动降级，不要照搬。
6. 全量覆盖说明描述抽取范围；本次提示词经过预算选择，不能把省略材料当作已审阅的证据，也不能仅据抽取覆盖率宣称长期模式。
7. 必须优先参考“结构化证据摘要”中的人物建模证据、反证与存疑；只有跨分块重复出现的信号才可写成稳定倾向。
   事实与推断须引用 evidence_id（E_...）；存疑条目不是已证实事实。不能以隔离条目或媒体占位符作为支持，逐字匹配也不代表解释成立。
8. 不预设对方性别；除直接引用原文外，全文使用“对方/此人/该对象/对象昵称”，不要使用“他/她/ta/男方/女方”等带性别预设的代称。

请按以下结构输出 Markdown：

# 人物侧写：{name}

## 一、基本画像
## 二、性格与表达风格
## 三、互动模式与边界感
## 四、动机与需求
## 五、对方在你面前的倾向性总结
## 六、不能判断的部分
## 七、对用户最重要的认知

补充要求：
1. 基本画像和倾向性总结尽量用表格。
2. 动机与需求部分用“动机假设 / 支持证据 / 反证 / 可信度”的表格。
3. 语言要像“证据支撑的人物侧写”，不是普通读后感。
4. 最后一行必须是：<!-- END_PERSONA_REPORT -->

以下是统计摘要：

{stats_summary}

以下是全量覆盖说明：

{coverage_summary}

以下是结构化证据摘要：

{evidence_summary}

以下是阶段总结：

{joined}

以下是已经完成的关系分析报告：

{relation_report}
"""


def load_phase_summaries(artifacts: dict) -> list[str]:
    text = artifacts["phase_summary_file"].read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"阶段总结为空: {artifacts['phase_summary_file']}")
    return [text]


@redact_artifact_call()
def generate_persona_report(
    artifacts: dict,
    persona_skill_prompt: str,
    config,
    timeout: int,
) -> str:
    if not artifacts["analysis_file"].exists():
        raise FileNotFoundError(f"缺少关系分析文件: {artifacts['analysis_file']}")
    if not artifacts["phase_summary_file"].exists():
        raise FileNotFoundError(f"缺少阶段总结文件: {artifacts['phase_summary_file']}")

    relation_report = artifacts["analysis_file"].read_text(encoding="utf-8")
    coverage_summary = (
        artifacts["coverage_summary_file"].read_text(encoding="utf-8").strip()
        if artifacts["coverage_summary_file"].exists()
        else "未找到覆盖说明文件；请优先依据阶段总结中的分块编号和统计摘要判断证据范围。"
    )
    persona_prompt = build_persona_prompt(
        artifacts["name"],
        artifacts["stats_summary"],
        coverage_summary,
        load_evidence_summary(artifacts),
        load_phase_summaries(artifacts),
        relation_report,
    )
    print("- 生成对方人物侧写...")
    persona_report = generate_marked_report(
        config,
        persona_skill_prompt,
        persona_prompt,
        timeout,
        marker="<!-- END_PERSONA_REPORT -->",
        label="人物侧写",
        initial_max_tokens=PERSONA_REPORT_MAX_OUTPUT_TOKENS,
    )
    with tempfile.TemporaryDirectory(prefix=".persona-", dir=artifacts["support_dir"]) as temporary:
        staged = {key: Path(temporary) / artifacts[key].name for key in ("persona_file", "persona_meta_file")}
        write_report(
            staged["persona_file"],
            staged["persona_meta_file"],
            persona_report,
            source="api",
            script_name="agent/run_workflow.py",
            model=config.model,
            note=f"基于全量分块阶段总结与关系建模报告生成，覆盖 {artifacts['message_count']} 条有效消息",
            extra_metadata={
                "reuse_identity": report_identity(artifacts, persona_skill_prompt, config, persona=True),
                "report_file": str(artifacts["persona_file"]),
                "message_count": artifacts["message_count"],
                "analysis_file": str(artifacts["analysis_file"]),
                "phase_summary_file": str(artifacts["phase_summary_file"]),
                "coverage_summary_file": str(artifacts["coverage_summary_file"]),
                "evidence_ledger_file": str(artifacts["evidence_ledger_file"]),
                "evidence_summary_file": str(artifacts["evidence_summary_file"]),
                "full_transcript_file": str(artifacts["full_transcript_file"]),
                "diagnostics_file": str(artifacts["diagnostics_file"]),
            },
        )
        audit = audit_artifacts({**artifacts, **staged}, require_persona=True)
        if not audit.ok:
            raise ValueError("人物侧写审计失败，未替换旧报告：" + str(audit.errors))
        publish_files({artifacts[key]: value for key, value in staged.items()})
    return str(artifacts["persona_file"])


def load_skill_prompt(skill_path: Path) -> str:
    return skill_path.read_text(encoding="utf-8")
