# Dialograph Agent Internals

这里记录 `agent/` 目录中的内部脚本和提示词文件。项目定位、完整运行方式、工作流链路和产物说明见根目录 [README.md](../README.md)。

## 主要入口

- `run_workflow.py`：完整工作流入口，负责预处理、全量分块、LLM 分块分析、结构化证据汇总、最终关系报告和人物侧写。
- `run_analysis.py`：轻量预处理入口，只生成统计、样本和分析 prompt，不保证 LLM 全量阅读原文。
- `run_single_pass_analysis.py`：单次关系分析预览入口，适合快速试跑，不建议作为最终长期建模结果。
- `run_single_pass_persona.py`：单次人物侧写预览入口，适合快速试跑，不建议作为最终长期建模结果。

## 核心模块

- `analyze.py`：聊天 JSON 读取、消息解析、时间排序、解析诊断、统计、会话建模、全量文本导出。
- `config.py`：读取 `.env` 中的 LLM API 配置。
- `output_paths.py`：输出目录与文件名清洗。
- `report_output.py`：报告写入和元数据写入。

## Prompt / Skill

- `SKILL.md`：关系动态、互动模式和边界分析框架。
- `PERSONA_SKILL.md`：人物印象、表达风格和稳定互动倾向建模框架。

两份 skill 都要求：

- 不预设对方性别、性取向、关系目标或恋爱/暧昧前提。
- 严格区分【事实】【推断】【假设】【存疑】。
- 优先寻找反证。
- 长期稳定模式优先于单点片段。

## 预设

- `presets/mimo-v2.5pro-sgp.env`：LLM API 配置模板。

