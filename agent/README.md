# Dialograph Agent Internals

这里记录 `agent/` 目录中的内部脚本和提示词文件。项目定位、完整运行方式、工作流链路和产物说明见根目录 [README.md](../README.md)。

当前交付与接手步骤见[队友接手指南](../docs/HANDOFF.md)。完整 CLI 和网页新任务默认 `human`；内部配置/生成函数以及缺少模式字段的旧网页任务仍兼容 `model`，内部接入应显式指定。

## 主要入口

- `run_workflow.py`：完整工作流命令行适配，参数和批处理展示委托给共享服务。
- `run_analysis.py`：轻量预处理入口，只生成统计、样本和分析 prompt，不保证 LLM 全量阅读原文。
- `run_single_pass_analysis.py`：单次关系分析预览入口，适合快速试跑，不建议作为最终长期建模结果。
- `run_single_pass_persona.py`：单次人物侧写预览入口，适合快速试跑，不建议作为最终长期建模结果。

## 核心模块

- `analyze.py`：聊天导出读取、消息解析、时间排序、解析诊断、统计、会话建模、全量文本导出。
- `analysis_artifacts.py`：共享脱敏预处理、统计摘要与产物路径；采样 prompt 按需生成。
- `analysis_service.py`：CLI/Web 共用的复用、自动续跑选择、关系/侧写生成和最终审计决策。
- `chunk_pipeline.py`：分块证据、阶段总结与关系报告的编排，以及限定于分块失败的缩块重跑。
- `chunking.py`：消息/字节/提示词预算分块、覆盖说明、提取公共日期行且保留精确秒数的无损分块提示词与还原，以及不可变的 `ChunkSettings` 配置。
- `chunk_evidence.py`：模型 JSON 提取、块内短编号还原、证据规范化、带发送方的摘要、账本写入与分块缓存核验。
- `summary_merge.py`：阶段总结的压缩合并、完成标记、批次缓存，以及零模型调用的跨时间确定性阶段聚合与对比工具。
- `analysis_types.py`：产物、运行路径和分块清单的 `TypedDict` 约定，保留现有 JSON 字典形状。
- `llm_client.py`：HTTP 请求、错误分类、退避、输出截断和完成标记重试。
- `llm_usage.py`：完整分析的请求用量账本、阶段/重试归属及已知/未知缓存计数，详见[模型用量与缓存](../docs/MODEL_BUDGET.md)。
- `prompt_budget.py`：按候选事件单元优先级、证据类别和时间轴选择报告材料，严格限制材料字节数并披露省略。
- `report_context.py`：从脱敏账本按物理消息来源去重构建候选事件单元（分离记录物理指标与待复核标签）、统一驱动摘要与连续窗口选择、对超预算长单元标明范围缩小、保留单锚点短互动并单独保留存疑。
- `report_generation.py`：关系/侧写统一系统提示词、共享前置材料前缀组装，以及侧写暂存、审计和发布。
- `report_framework.py`：按固定次序合成两份实际技能全文及公共约束，供两份报告和复用指纹共同使用。
- `evaluate_claim_semantics.py`：离线比对固定命题的人工标注、证据角色和所给原文覆盖，不调用模型。
- `claim_ledger.py` 编排结论生成、有界纠错、确认模式、报告渲染与审计，保留旧导入入口。人工模式初稿最多一次结构纠错；模型模式另有一次语义修订和一次修订结构纠错。详见[结论账本](../docs/CLAIM_LEDGER.md)。
- `claim_compiler.py` 负责结论协议、引用范围、稳定编号和编译；`claim_repair.py` 负责纠错材料与完整条目替换。
- `claim_human_review.py` 生成绑定结论/输入哈希的待确认记录并独立审计；当前只允许 `pending`，不提供已确认状态。
- `claim_facts.py` 复制已核验引文的原话、发送方和时间，计算消息间隔，校验叙述引语与动作的逐字出处及结果确认结构。动作含义在人工模式交由用户核对，在模型模式另做逐项语义复核；程序出处合格不证明动作解释正确。
- `claim_retrieval.py`：完整脱敏原文规划、命题双向字面检索、按命题均衡的完整会话选择、计划缓存与省略原因，初始报告邻文覆盖不足时自动使用。
- `claim_field_review.py` / `claim_semantic_review.py` 仅在模型模式执行逐字段或兼容语义复核，核对全局统计、连续原文、叙述、反证和动作，保存绑定记录。响应协议错误可纠正一次，未解决的问题阻止该模式发布；人工模式不调用。真实质量验收范围见[离线语义验收](../docs/SEMANTIC_EVALUATION.md)。
- `run_promotion.py`：文件组发布及异常回滚；关系快照指针最后更新。
- `preprocess_cli.py` / `preview_prompts.py`：共享预处理交互与采样 prompt 模板。
- `preview_support.py`：两个采样预览入口共用的材料压缩和请求回退；预览入口不互相导入。
- `config.py`：读取 `.env` 中的 LLM API 配置。
- `output_paths.py`：输出目录与文件名清洗。
- `report_output.py`：报告写入和元数据写入。
- `conversation_input.py`：网页与 CLI 共用的 WeFlow 私聊输入契约。
- `conversation_formats.py`：统一 JSON v1、Alpha Markdown 与 WeFlow 的格式适配、文件发现和读取边界，详见 [导入协议](../docs/CONVERSATION_INPUT.md)。
- `workflow_state.py`：运行路径、元数据、续跑输入/流程指纹和保存参数校验；用 `PreparedRun` 汇集预检结果，全部预检通过后才写入运行材料。
- `evidence_validation.py`：证据字段/枚举校验及跨块摘要取样。
- `evidence_grounding.py`：输入快照和消息编号、逐字引文绑定、歧义隔离、证据编号及独立重算校验，详见 [原文绑定说明](../docs/EVIDENCE_GROUNDING.md)。
- `evidence_store.py`：校验 v2 脱敏账本，按快照隔离的 SQLite/FTS5 中文查询和来源回查，详见 [证据检索](../docs/EVIDENCE_RETRIEVAL.md)。
- `web_evidence.py`：为网页已完成任务按需建立派生索引，核对实际输入快照及脱敏规则，并以版本约束查询和原文回查。
- `report_references.py`：在已校验任务账本中解析报告引用，区分证据、单条消息和对话片段；唯一匹配的缩写可回查原文。
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

内部调用使用 `ChunkSettings` 汇集分块配置，`PreparedRun` 汇集恢复预检结果。辅助函数从实际所属模块导入；运行缓存按输入、源码、提示词和模型指纹判断是否可复用。
