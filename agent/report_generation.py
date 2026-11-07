"""报告上下文构建、统一公共材料前缀与经过审计的人物侧写发布。"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from artifact_audit import audit_artifacts
from report_framework import build_report_system_prompt
from chunk_evidence import format_evidence_summary
from evidence_grounding import GROUNDING_VERSION
from llm_client import call_llm_with_retry
from llm_usage import usage_operation
from report_output import write_report
from run_promotion import publish_files
from workflow_state import report_identity
from privacy_redaction import redact_artifact_call
from prompt_budget import (limit_text_bytes, budget_evidence_summary, budget_phase_summaries)
from report_context import select_evidence_context, source_repetition_notes, EvidenceContext
from claim_ledger import ReportInput, claim_contract, context_catalog, generate_claims, render_claim_report, VERSION as CLAIM_VERSION
from claim_retrieval import retrieve_claim_context, retrieval_notes
from evidence_grounding import digest
from claim_field_review import FIELD_REVIEW_INSTRUCTIONS, review_json_mode
from config import report_context_options


def load_evidence_summary(artifacts: dict) -> str:
    summary_file = artifacts.get("evidence_summary_file")
    if summary_file and summary_file.exists():
        return summary_file.read_text(encoding="utf-8").strip()
    return "未找到结构化证据摘要；请主要依据阶段总结和覆盖说明，并降低细粒度结论的确定性。"


FINAL_STATS_SUMMARY_MAX_BYTES = 5000
FINAL_EVIDENCE_SUMMARY_MAX_BYTES = 24000
FINAL_COVERAGE_SUMMARY_MAX_BYTES = 2500
FINAL_PHASE_SUMMARY_MAX_BYTES = 12000
FINAL_RETRIEVAL_NOTES_MAX_BYTES = 4000
FINAL_REPORT_MAX_OUTPUT_TOKENS = 4800
PERSONA_REPORT_MAX_OUTPUT_TOKENS = 4800

def _prepare_shared_materials(
    name: str,
    stats_summary: str,
    coverage_summary: str,
    evidence_summary: str,
    phase_summaries: list[str],
    *,
    ledger: dict | None = None,
    review_questions: list[dict] | None = None,
    context_options: dict | None = None,
) -> tuple[str, EvidenceContext]:
    stats_summary = limit_text_bytes(stats_summary, FINAL_STATS_SUMMARY_MAX_BYTES)
    coverage_summary = limit_text_bytes(coverage_summary, FINAL_COVERAGE_SUMMARY_MAX_BYTES)
    joined = budget_phase_summaries(phase_summaries, FINAL_PHASE_SUMMARY_MAX_BYTES)

    use_full_ledger = (
        ledger is not None
        and bool(ledger.get('documents'))
        and all(doc.get('grounding_version') == GROUNDING_VERSION for doc in ledger['documents'])
        and (review_questions is not None or '- schema: ' in evidence_summary or '- 总数 ' in evidence_summary)
    )
    if use_full_ledger and review_questions is not None:
        context = retrieve_claim_context(ledger, review_questions, **(context_options or {}))
        context = context.with_selected_ids(set(context.retrieval['source_message_ids']))
    elif use_full_ledger:
        context = select_evidence_context(evidence_summary + '\n' + joined, ledger, select_from_ledger=True)
        full_summary = format_evidence_summary(ledger['documents'], max_items_per_type=None)
        omitted_ids = {
            eid for info in context.narrowed_units.values() for eid in info['omitted_evidence_ids']
        }
        priority_ids = context.context_ids | omitted_ids
        evidence_summary = budget_evidence_summary(
            full_summary,
            FINAL_EVIDENCE_SUMMARY_MAX_BYTES,
            priority_ids=priority_ids,
        )
        summary_ids = set(re.findall(r'\bE_[0-9a-f]{32}\b', evidence_summary + '\n' + joined))
        summary_ids.update(eid for eid in context.context_ids if eid.startswith('M_'))
        context = context.with_selected_ids(summary_ids)
    else:
        evidence_summary = budget_evidence_summary(evidence_summary, FINAL_EVIDENCE_SUMMARY_MAX_BYTES)
        context = (
            select_evidence_context(evidence_summary + '\n' + joined, ledger)
            if ledger is not None
            else EvidenceContext('本次未提供可回填的原文账本。', set(), set())
        )

    source_context = context.text
    retrieval_text = (limit_text_bytes(retrieval_notes(context.retrieval), FINAL_RETRIEVAL_NOTES_MAX_BYTES)
                      if context.retrieval is not None else '')
    source_only = context.retrieval is not None
    repetition_text = source_repetition_notes(ledger, context.source_positions) if source_only else ''
    catalog = '' if source_only else context_catalog(context.context_ids, ledger)
    derived_materials = '' if source_only else f"""以下是结构化证据摘要：

{evidence_summary}

以下是按预算选择的阶段材料：

{joined}

"""
    source_rules = ('本轮按完整会话原文直接形成结论，引用使用原文行的P_序号。先选与判断逐句对应的原文，再写判断及解释。'
                    '支持、反证与限制均按当前判断核验。解释中的新增动作也须引用其自身原文。'
                    '涉及失误或冲突时，先写实际发生的行为、主体与影响，再分别说明道歉、补救动作和已知结果。'
                    '前后回应有变化时按时间说明过程。样本数量、情境描述及总结须与各条具体结论一致。'
                    '检索覆盖计数仅用于说明选材范围。\n') if source_only else ''
    reference_rules = ('原文行的P_序号是本轮引用入口。每个具体动作都引用其自身原话，程序核对来源并转换为稳定编号。'
                       if source_only else
                       '事实与推断须引用 evidence_id，逐字复制完整的 E_ 加 32 位十六进制编号，或完整窗口目录列出的完整M编号。存疑条目不是已证实事实，须单列。')
    shared_prefix = f"""请基于以下公共分析材料，按文末的具体任务指令和结论输出协议生成报告。

{source_rules}

对象：{name}

公共写作与核验标准：
1. 参考法证式分析，不要写空话套话；可以细，但不能越界读心，不做人格审判或医学化诊断。
2. 每个大结论写支持依据、相关反证的复核结果及情境限制，允许未找到相关反证；反证须明确削弱当前判断的哪一部分，时间、场景等限制单列。区分【事实】【推断】【假设】【存疑】以及长期、阶段和局部范围；证据不足写【存疑】。
3. 输出要信息密度高，优先保留最能支撑结论的判断，避免重复。
4. 全量覆盖说明记录抽取阶段处理的范围；本次提示词经过预算选择，省略内容不能当作已被模型审阅的证据，不得声称逐条核验全部证据，也不能仅据抽取覆盖率宣称长期模式。
5. {reference_rules} 原文编号仅证明出处，发送者立场及语义角色须逐命题核查。
6. 不预设对方性别、性取向、关系目标或恋爱/暧昧前提；亲密或暧昧只作为证据充分时的可选模型。
7. 除直接引用原文外，全文使用“对方/此人/该对象/对象昵称”称呼分析对象；不要使用“他/她/ta/男方/女方”等带性别预设的代称。
8. 按摘要标注的引文发送方归属言语；“我”的自述不能写成对方说过的话，不能直接据此评估对方的表达风格或性格。关系报告中的转述也须按此核对；互动判断须有对方回应或行为依据。
9. 抽取条目的分类、signal/against、阶段总结和关系报告均为待复核的初步或二手解释，不是事实；逐字绑定只核对出处。优先结合连续原文检查这些解释是否成立，反证必须与所反驳的判断相关。
10. 不能把省略或未回填当作不存在；未回填且需要上下文的解释保留存疑。短消息不等于浅交流，应按连续会话判断，不能用短消息长度判断交流深度。
11. 对方的单次或单场景表现不能直接推成稳定人格；没有足够跨场景或跨分块重复依据时保留局部范围或未知，不得把单次表现写成稳定人格或评价为能力偏弱。

以下是全局统计：

{stats_summary}

以下是全量覆盖说明：

{coverage_summary}

{derived_materials}{source_context}{retrieval_text}{repetition_text}{catalog}"""
    return shared_prefix, context


def prepare_final_input(
    name: str,
    stats_summary: str,
    coverage_summary: str,
    evidence_summary: str,
    phase_summaries: list[str],
    *, ledger: dict | None = None, review_questions: list[dict] | None = None,
    context_options: dict | None = None,
) -> ReportInput:
    shared_prefix, context = _prepare_shared_materials(
        name,
        stats_summary,
        coverage_summary,
        evidence_summary,
        phase_summaries,
        ledger=ledger,
        review_questions=review_questions,
        context_options=context_options,
    )
    task_tail = """
## 本次具体任务：关系与行为模式分析报告
按下方结论协议选择有证据的章节，每条只保留一个独立判断。
优先描述实际互动、冲突与支持的过程，再评估原文支持的关系解释。
总结可合并已有结论；只有原文提出了待解决的问题时才给对应验证建议。
"""
    return ReportInput(
        shared_prefix + task_tail + claim_contract('analysis', raw_sources=context.retrieval is not None),
        context.evidence_ids,
        context.context_ids,
        source_positions=context.source_positions, retrieval=context.retrieval,
    )


def build_final_prompt(*args, **kwargs) -> str:
    return prepare_final_input(*args, **kwargs).prompt


def prepare_persona_input(
    name: str,
    stats_summary: str,
    coverage_summary: str,
    evidence_summary: str,
    phase_summaries: list[str],
    relation_report: str,
    *, ledger: dict | None = None, review_questions: list[dict] | None = None,
    context_options: dict | None = None,
) -> ReportInput:
    shared_prefix, context = _prepare_shared_materials(
        name,
        stats_summary,
        coverage_summary,
        evidence_summary,
        phase_summaries,
        ledger=ledger,
        review_questions=review_questions,
        context_options=context_options,
    )
    relation_report = limit_text_bytes(relation_report, 2000)
    task_tail = f"""
## 本次具体任务：对方人物侧写建模报告
1. 输出一份独立的人物形象建模报告，而不是重复关系分析报告；重点回答“这个人，在与你互动时，呈现出怎样稳定的人物形象”。
2. 写出边界、互动风格、情绪表达、投入方式、回避方式、关系需求等层面；必须优先参考人物建模证据、反证与存疑，只有跨分块重复出现的信号才可写成稳定倾向。
3. 如果关系分析报告里的某个判断证据不够，要主动降级，不要照搬；动机与需求均作为待验证假设，人物倾向描述在这段互动中的表现，未获得跨场景证据时保留局部范围。

以下是关系报告的二手解释，仅作待复核线索：

{relation_report}
"""
    if context.retrieval is not None:
        task_tail = """
## 本次具体任务：对方人物侧写建模报告
根据完整入选会话独立归纳对方可核验的表达方式、互动习惯与边界。
优先说明具体场景中的表现及其依据，重复倾向必须有对应的跨场景支持。
动机与需求须有方向正确的依据，证据不足的维度可省略或明确存疑。
"""
    return ReportInput(
        shared_prefix + task_tail + claim_contract('persona', raw_sources=context.retrieval is not None),
        context.evidence_ids,
        context.context_ids,
        source_positions=context.source_positions, retrieval=context.retrieval,
    )


def build_persona_prompt(*args, **kwargs) -> str:
    return prepare_persona_input(*args, **kwargs).prompt


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
    *, report_system_prompt: str | None = None,
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
    ledger = json.loads(artifacts['evidence_ledger_file'].read_text(encoding='utf-8'))
    retrieval = None
    if artifacts['analysis_claims_file'].exists():
        retrieval = json.loads(artifacts['analysis_claims_file'].read_text(encoding='utf-8')).get('retrieval')
        if retrieval is not None and (retrieval['source_id'] != ledger['source_id'] or retrieval['ledger_sha256'] != digest(ledger)):
            raise ValueError('关系报告的检索计划与人物侧写来源不一致。')
    prepared = prepare_persona_input(
        artifacts["name"],
        artifacts["stats_summary"],
        coverage_summary,
        load_evidence_summary(artifacts),
        load_phase_summaries(artifacts),
        relation_report,
        ledger=ledger,
        review_questions=retrieval['questions'] if retrieval is not None else None,
        context_options=report_context_options(config),
    )
    system_prompt = (report_system_prompt if report_system_prompt is not None else
                     build_report_system_prompt(persona_skill_prompt=persona_skill_prompt))
    print("- 生成对方人物侧写...")
    with usage_operation('persona'):
        claims = generate_claims(prepared, ledger, 'persona', lambda prompt: call_llm_with_retry(
            config,
            system_prompt,
            prompt,
            timeout,
            max_tokens=PERSONA_REPORT_MAX_OUTPUT_TOKENS,
        ), attempt_dir=artifacts['support_dir'] / 'claim_attempts_persona',
            review_request=lambda prompt: call_llm_with_retry(config, FIELD_REVIEW_INSTRUCTIONS, prompt, timeout,
                max_tokens=4800, json_mode=review_json_mode(config)), field_review=True,
            review_mode=getattr(config, 'report_review_mode', 'model'))
    persona_report = render_claim_report(claims, artifacts['name'])
    with tempfile.TemporaryDirectory(prefix=".persona-", dir=artifacts["support_dir"]) as temporary:
        staged = {key: Path(temporary) / artifacts[key].name for key in ("persona_file", "persona_meta_file", "persona_claims_file", "persona_input_file")}
        staged['persona_claims_file'].write_text(json.dumps(claims, ensure_ascii=False, indent=2), encoding='utf-8')
        staged['persona_input_file'].write_text(prepared.prompt, encoding='utf-8')
        write_report(
            staged["persona_file"],
            staged["persona_meta_file"],
            persona_report,
            source="api",
            script_name="agent/run_workflow.py",
            model=config.model,
            note=f"根据本次入选材料生成人物侧写。全量抽取覆盖 {artifacts['message_count']} 条有效消息",
            extra_metadata={
                "claim_schema_version": CLAIM_VERSION,
                "reuse_identity": report_identity(artifacts, persona_skill_prompt, config, persona=True, report_system_prompt=system_prompt),
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
        audit = audit_artifacts({**artifacts, **staged}, require_persona=True, require_claims=True)
        if not audit.ok:
            raise ValueError("人物侧写审计失败，未替换旧报告：" + str(audit.errors))
        publish_files({artifacts[key]: value for key, value in staged.items()})
    return str(artifacts["persona_file"])


def load_skill_prompt(skill_path: Path) -> str:
    return skill_path.read_text(encoding="utf-8")
