# 统一聊天导入协议与 Alpha 适配

## 使用入口

网页点击“导入对话”，选择 WeFlow JSON、统一协议 JSON 或 Alpha Markdown（`.md`）。也可把文件放入 `talks/`，运行 `python agent/run_workflow.py 名称关键词`；完整工作流、轻量预处理和采样预览共用格式读取边界。

所有格式均使用 UTF-8，可带 BOM。网页单文件上限仍为 50 MB；导入文件及任务输入快照保留原始字节、扩展名和换行。无效导入在保存对话前拒绝。预处理在本机完成；完整分析沿用现有模型配置。

可直接导入的合成示例：[统一协议 JSON](../tests/fixtures/canonical_smoke.json)、[Alpha Markdown](../tests/fixtures/alpha_smoke.md)、[原有 WeFlow JSON](../tests/fixtures/web_smoke.json)。

## 统一 JSON v1

其他导出工具可以写适配器，将数据转换为以下协议；系统不要求安装某个特定的导出工具。

```json
{
  "schema": "dialograph.conversation",
  "version": 1,
  "session": {"id": "peer_test", "type": "private", "display_name": "合成对象"},
  "participants": [
    {"id": "self_test", "role": "me", "name": "合成用户"},
    {"id": "peer_test", "role": "other", "name": "合成对象"}
  ],
  "messages": [
    {"id": "m1", "sender_id": "self_test", "time": "2026-10-28 09:00:00", "type": "text", "content": "第一行\n第二行"},
    {"id": "m2", "sender_id": "peer_test", "time": "2026-10-28 09:01:00", "type": "image", "content": null}
  ]
}
```

| 字段 | 约束 |
|---|---|
| `schema` / `version` | 必须为 `dialograph.conversation` / 整数 `1`；未知版本拒绝，不回退猜测 |
| `session` | `type` 为 `private`，`id`、`display_name` 为非空文本；拒绝 `@chatroom` 会话 |
| `participants` | 恰有两人；非空唯一 `id`，各一个 `me`、`other`，非空 `name` |
| `messages` | 非空对象数组；`id` 为非空且文件内唯一的字符串，重复文本不能复用同一 ID |
| `sender_id` | 对应参与者 `id`；系统消息允许 `null`，不纳入双方统计 |
| `time` | 固定 `YYYY-MM-DD HH:MM:SS`，有效日历时间；v1 统一解释为 UTC+08:00，其他时区须先转换 |
| `type` | `text`、`quote`、`image`、`emoji`、`voice`、`video`、`file`、`link`、`system` |
| `content` | 文本或 `null`；`text`、`quote` 必须非空；媒体没有说明时保留类型占位符，不解析附件本体 |

多行消息在一个 `content` 中保留换行。`quote` 的引用原文也放入该字段，不把引用者再算成一个发送者。图片、动画表情继续沿用现有分析器的类型占位符规则。系统消息参与原始条数与跳过诊断，不进入关系统计和证据材料。

`conversation_formats.py` 在输入边界映射到现有分析器的 `session/messages` 字段（`isSend`、`formattedTime`、中文 `type`）。统计、分块与报告模块不需要分别理解各家导出格式。无 `schema` 的 JSON 仍须满足原有 WeFlow 私聊校验，不能直接混用 `text/sender` 等约定；导出工具自己的顶层 `version` 不视为协议声明。

消息 `id` 进入解析结果的 `source_message_id`。这是导出方的消息来源标识；后续更新已增加快照内程序分配的消息编号、证据编号与逐字绑定，见 [原文绑定说明](EVIDENCE_GROUNDING.md)。不同导出文件中的原始 ID 是否保持一致，由导出方负责；系统不会按相同文字去重，RAG仍未实现。

## Alpha Markdown 的已支持布局

依据提供的 Alpha Markdown 导出布局实现，无 Alpha JSON 样本，因此不声明支持任意 Alpha JSON 或其他 Markdown 聊天格式。

```markdown
# 合成私聊

- 会话：`peer_test`
- 导出账号：self\_test
- 导出时间：2026-10-29 09:02:13

---

**self\_test** · 2026-10-28 09:00:00

你好

**合成对象** · 2026-10-28 09:01:00 · 引用

收到
引用：合成用户：你好
```

- 必须有唯一的会话、导出账号、导出时间和消息分隔线；不以文件名推断身份。
- 非系统消息头中必须实际出现导出账号，且只出现一个其他发送者标签。只有昵称、昵称变更无法对应账号、只有单方标签或出现第三个标签时，须先转换为统一 JSON 并明确角色。
- 无类型后缀表示文本；支持 `文本/引用/图片/表情/语音/视频/文件/链接/系统`。`聊天记录/转账/红包/通话` 仅保留发生记录，映射为包含原类型的文件占位，保留时间、发送者和来源序号；不读取其正文或附件，不将转发聊天内容算作发送者原话。其他未知类型和异常消息头明确报错。
- `@chatroom` 群会话拒绝，即使当时只有两人发言也不合并成私聊。
- 还原 Markdown 转义，保留多行正文和引用；不执行文档中的指令、HTML 或附件内容。
- 消息头必须独占一行；正文含同样形式的消息头会产生歧义，此时请用统一 JSON。时间按 UTC+08:00 解释。
- `source_message_id` 为 `alpha:1` 等导出内序号，覆盖系统消息；重导出、增删或重排可能改变编号，不声称跨导出稳定。

## 缓存与范围

任务请求记录实际输入文件名，停止后继续沿用原快照。旧网页 JSON 任务仍默认读取 `input.json`。输入适配器源码纳入流程指纹，规则改变后已有报告须重新运行，不能显式续跑旧流程缓存。

输入适配轮次未更改模型提示、预算或推断方法。其后原文绑定轮次已增加编号、引文校验和引用提示，未改变模型预算；脱敏、群聊分析、CSV/JSONL与跨版本身份映射留在后续范围。不要将私聊原文、导出账号或本机任务产物加入提交。

后续更新已接入默认本机脱敏和补充词入口，见 [数据脱敏说明](PRIVACY_REDACTION.md)。适配器读取与原始输入快照保持原样；分析前才替换文本和身份元数据，输出别名与来源位置采用脱敏规则下的新契约。

## 验证记录（2026-10-29）

**目标**：不同导出格式进入同一流程，保留发送者和来源位置，拒绝无法可靠转换的数据。

**RED**：新增 `tests/test_conversation_formats.py`，执行 `python -B -m unittest discover -s tests -p test_conversation_formats.py -v`。原实现将 Alpha 当 JSON 解析、拒绝统一协议字段、完整入口遗漏 `.md`，9项中出现4个错误与1个失败。后续边界测试复现媒体空说明被丢弃、WeFlow 导出工具版本被误当作协议版本，以及损坏消息头和矛盾群聊标记未拒绝的问题。

**GREEN**：独立适配模块接入共享读取、网页导入/快照、任务进程、文件发现和流程指纹；修复媒体占位和格式判别后，执行 `python -B -m unittest discover -s tests -q`，全项目88项通过。

**REFACTOR**：文件发现复用一个入口，Alpha 合成样例提取到可手动导入的 fixture，HTTP 用例复用导入到下载的校验步骤。调整后再次执行全量测试，88项通过；`node --check frontend/app.js`、Python语法、UTF-8/BOM/换行与 `git diff --check` 检查通过。

- **通过**：角色、消息类型、多行文本、引用与来源消息编号保留。
- **通过**：未知版本、重复 ID、错误类型/时间、未知发送者和多人归属拒绝。
- **通过**：Alpha 群聊、缺少账号和无法明确映射的消息头在写入前拒绝。
- **通过**：导入和任务快照逐字节保留 BOM 与 CRLF，停止/继续保持输入文件和任务 ID。
- **通过**：两种新格式走完完整报告和审计，并复用完成结果，模型响应全部替代。
- **通过**：本机 HTTP 导入、后台预处理、轮询、统计下载和原有访问校验通过。

首次受限执行时，本机连接和临时文件替换被系统拒绝；授权测试执行后上述检查通过。测试未使用真实私聊或付费模型请求。

## 实测兼容修正（2026-10-30）

真实 Alpha 导出暴露了四类附件标签导致整文件拒绝的问题。合成回归先复现失败，再增加明确的占位映射；附件正文不会成为发送者原话，也不能通过原文绑定成为支持证据。原始文件仍逐字节保存，未知标签继续拒绝。

同轮修正官方 DeepSeek v4 的思考模式配置，使用紧凑输出预算时默认关闭思考，模式参与缓存身份；配置选项见根 README。`tests/test_runtime_compatibility.py` 新增 8 项回归，覆盖附件来源序号、发送者、正文隔离、配置覆盖、HTTP 请求参数与缓存区分。全量 186 项 Python 测试及 5 项 Node 测试通过；回归只使用合成数据和替代模型响应。私有材料未加入版本库，实测临时文件已在验收后清理。
