# Dialograph Agent Internals

这里记录 `agent/` 目录中的内部脚本和提示词文件。项目定位、完整运行方式、工作流链路和产物说明见根目录 [README.md](../README.md)。

## 主要入口

- `run_workflow.py`：完整工作流命令行适配，参数和批处理展示委托给共享服务。
- `run_analysis.py`：轻量预处理入口，只生成统计、样本和分析 prompt，不保证 LLM 全量阅读原文。
- `run_single_pass_analysis.py`：单次关系分析预览入口，适合快速试跑，不建议作为最终长期建模结果。
- `run_single_pass_persona.py`：单次人物侧写预览入口，适合快速试跑，不建议作为最终长期建模结果。

## 核心模块

- `analyze.py`：聊天导出读取、消息解析、时间排序、解析诊断、统计、会话建模、全量文本导出。
- `analysis_artifacts.py`：共享脱敏预处理、统计摘要与产物路径；采样 prompt 按需生成。
- `analysis_service.py`：CLI/Web 共用的复用、自动续跑选择、关系/侧写生成和最终审计决策。
- `chunk_pipeline.py`：连续分块、请求检查点、原文绑定、证据汇总及关系报告运行。
- `llm_client.py`：HTTP 请求、错误分类、退避、输出截断和完成标记重试。
- `prompt_budget.py`：按证据类别和时间轴选择报告材料，严格限制材料字节数并披露省略。
- `report_generation.py`：关系/侧写提示词以及侧写暂存、审计和发布。
- `run_promotion.py`：文件组发布及异常回滚；关系快照指针最后更新。
- `preprocess_cli.py` / `preview_prompts.py`：共享预处理交互与采样 prompt 模板。
- `config.py`：读取 `.env` 中的 LLM API 配置。
- `output_paths.py`：输出目录与文件名清洗。
- `report_output.py`：报告写入和元数据写入。
- `conversation_input.py`：网页与 CLI 共用的 WeFlow 私聊输入契约。
- `conversation_formats.py`：统一 JSON v1、Alpha Markdown 与 WeFlow 的格式适配、文件发现和读取边界，详见 [导入协议](../docs/CONVERSATION_INPUT.md)。
- `workflow_state.py`：续跑输入/流程指纹和保存参数的校验。
- `evidence_validation.py`：证据字段/枚举校验及跨块摘要取样。
- `evidence_grounding.py`：输入快照和消息编号、逐字引文绑定、歧义隔离、证据编号及独立重算校验，详见 [原文绑定说明](../docs/EVIDENCE_GROUNDING.md)。
- `evidence_store.py`：校验 v2 脱敏账本，按快照隔离的 SQLite/FTS5 中文查询和来源回查，详见 [证据检索](../docs/EVIDENCE_RETRIEVAL.md)。
- `web_evidence.py`：为网页已完成任务按需建立派生索引，核对实际输入快照及脱敏规则，并以版本约束查询和原文回查。
- `privacy_redaction.py`：默认身份/格式化敏感信息替换、补充词、模型请求与返回值脱敏，以及任务上下文隔离，详见 [脱敏说明](../docs/PRIVACY_REDACTION.md)。

两份 single-pass 入口只写 `analysis/预览/`，人物预览读取关系预览；文件头和控制台均标明预览。正式报告仍由完整工作流生成。

## Prompt / Skill

- `SKILL.md`：关系动态、互动模式和边界分析框架。
- `PERSONA_SKILL.md`：人物印象、表达风格和稳定互动倾向建模框架。

两份 skill 都要求：

- 不预设对方性别、性取向、关系目标或恋爱/暧昧前提。
- 严格区分【事实】【推断】【假设】【存疑】。
- 优先寻找反证。
- 长期稳定模式优先于单点片段。

## 预设

- 项目根目录 `.env.example`：LLM API 配置模板；复制为本地 `.env` 后填写自己的配置。

本轮可靠性范围、验证记录和后续边界见 [可靠性改进记录](../docs/RELIABILITY_PASS.md)。
