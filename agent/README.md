# 聊天记录关系分析

对 talks/ 中的聊天记录进行深度关系与行为模式分析。

## API 配置

在工作区根目录新建 .env，填写：

```
LLM_API_BASE=https://token-plan-sgp.xiaomimimo.com/v1
LLM_API_KEY=你的密钥
LLM_MODEL=mimo-v2.5pro
```

现成预设模板在 agent/presets/mimo-v2.5pro-sgp.env。

## 步骤

1. 先运行预处理脚本获取统计数据：
   ```
   python agent/run_analysis.py <编号或名称>
   ```

   如果要直接对 talks/ 下全部聊天记录跑完整工作流（预处理 + 全量分块阅读 + 关系建模 + 人物侧写），运行：
   ```
   python agent/run_workflow.py
   ```

   常用参数：
   ```
   python agent/run_workflow.py 黄芷馨
   python agent/run_workflow.py --limit 2
   python agent/run_workflow.py --force
   python agent/run_workflow.py 黄芷馨 --force-persona
   python agent/run_workflow.py --chunk-size 900 --chunk-max-bytes 80000
   ```

   Windows 下如果不想手动敲命令，可以直接双击工作区根目录的：
   ```
   一键运行全部分析.bat
   ```
   它会自动清空旧的 analysis 产物，并处理 talks/ 下全部 json 文件。

2. 阅读 `agent/SKILL.md` 获取完整分析框架

3. 阅读 `analysis/` 下对应的数据文件

   文件含义：
   - `预处理_{对象名}.txt`：统计摘要 + 代表性片段，不是全量消息
   - `全量解析_{对象名}.txt`：完整解析后的全部有效消息文本
   - `统计_{对象名}.json`：结构化统计结果
   - `分块覆盖清单_{对象名}.json`：完整工作流每个分块覆盖的消息序号、时间范围、消息数
   - `全量覆盖说明_{对象名}.md`：供最终报告和人物侧写使用的覆盖范围说明
   - `分块分析/chunk_*.md`：每个连续原文分块的证据分析
   - `结构化证据/evidence_*.json`：每个分块抽取出的事件、关系信号、人物信号、反证和存疑
   - `结构化证据总表_{对象名}.json`：全部分块结构化证据汇总，可作为后续 RAG 数据源
   - `结构化证据摘要_{对象名}.md`：供最终关系建模和人物侧写使用的证据摘要
   - `阶段总结_{对象名}.md`：由全部分块分析合并出的中间证据总结

4. 按照 SKILL.md 中的八个维度生成关系分析报告

5. 按照 PERSONA_SKILL.md 生成独立的人物侧写报告

6. 最终会生成两个结果文件：
- `analysis/分析_{对象名}.md`
- `analysis/人物侧写_{对象名}.md`

## 全量读取说明

聊天记录较长时，不会把 4 万条消息一次性塞进最终 prompt，而是按连续时间顺序分块：

1. 每个分块把原文完整交给 LLM 阅读并产出证据分析。
2. 每个分块会同时产出 Markdown 分析和结构化 JSON 证据。
3. 分块分析会合并为阶段总结，结构化证据会合并为证据总表和摘要。
4. 最终关系建模和人物侧写基于统计摘要、覆盖说明、结构化证据摘要、全部阶段总结和已生成报告。
5. `分块覆盖清单_{对象名}.json` 可用于核对是否覆盖了全部有效消息。

因此完整工作流保证的是“LLM 逐块读完整段对话”，不是“最后一步单 prompt 直接读完整原文”。这样更稳，也更适合超长聊天记录。

## 结构化证据与 RAG

`结构化证据总表_{对象名}.json` 是后续做 RAG 的主要入口。它按分块保存：

- `events`：事件层证据
- `relation_signals`：关系模型支持/反证信号
- `persona_signals`：人物形象建模信号
- `counter_evidence`：反证与边界条件
- `uncertainties`：仍不能判断的问题

后续可以把这些条目导入 SQLite/FTS 或向量库，按“时间、关系模型、人物特征、反证类型”检索。

## 关键原则

- 所有判断必须引用聊天记录具体证据
- 严格区分事实/推断/假设/存疑
- 优先寻找反证
- 不迎合、不放大、不诊断
