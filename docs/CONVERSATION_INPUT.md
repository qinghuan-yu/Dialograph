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

消息 `id` 进入解析结果的 `source_message_id`。这是导出方的消息来源标识；后续更新已增加快照内程序分配的消息编号、证据编号与逐字绑定，见 [原文绑定说明](EVIDENCE_GROUNDING.md)。不同导出文件中的原始 ID 是否保持一致，由导出方负责；系统不会按相同文字去重。本地关键词检索见 [证据检索](EVIDENCE_RETRIEVAL.md)，报告生成尚未接入检索增强。

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
- 无类型后缀表示文本；`文本/引用/图片/表情/语音/视频/文件/链接/系统` 按对应类型解析。
- Alpha 文本或引用正文为空、仅含空白时，保留“导出正文为空”的发生记录及原始发送者、时间和来源序号，映射为文件占位，不作为逐字引文。统一 JSON v1 的文本非空要求保持不变。
- 其余非空类型标签统一保留发生记录，包括 `聊天记录/转账/红包/通话/名片/位置/其他` 和后续新增标签。映射为包含原标签的文件占位，保留时间、发送者和来源序号；附件正文不进入分析文本，占位符不作为逐字引文依据。此规则无需随每个新标签更新名单。异常消息头、无效时间和身份归属仍按上述约定校验。
- `@chatroom` 群会话拒绝，即使当时只有两人发言也不合并成私聊。
- 还原 Markdown 转义，保留多行正文和引用；不执行文档中的指令、HTML 或附件内容。
- 消息头必须独占一行；正文含同样形式的消息头会产生歧义，此时请用统一 JSON。时间按 UTC+08:00 解释。
- `source_message_id` 为 `alpha:1` 等导出内序号，覆盖系统消息；重导出、增删或重排可能改变编号，不声称跨导出稳定。

## 微信与导出工具的类型对应

核对日期：2026-11-05。微信原始类型、应用消息子类型与导出标签属于不同层次，不能用一张固定名单覆盖所有工具和版本。

| 类别 | 微信原始类型或应用子类型（WeFlow 源码） | 本项目处理 |
|---|---|---|
| 文本、图片、语音、视频、表情 | `1 / 3 / 34 / 43 / 47` | Alpha 对应标签按已有类型解析 |
| 名片、位置、通话 | `42 / 48 / 50` | Alpha 保留类型、发送者、时间与来源序号 |
| 系统 | `10000`；部分系统内容由导出工具额外识别 | Alpha `系统` 跳过分析，保留原始序号 |
| 应用消息 | `49` 及文件、引用等大整数变体 | 依据导出标签处理，不直接猜测原始数字含义 |
| 音乐、链接、文件、转发记录 | 应用子类型 `3 / 5 / 6 / 19` | Alpha 链接、文件按已有类型解析；音乐、转发记录保留发生记录 |
| 小程序、引用、群公告、转账、红包、视频号 | 应用子类型 `33/36 / 57 / 87 / 2000 / 2001 / 51` | Alpha 引用按已有类型解析；其余标签保留发生记录 |
| 其他及扩展类型 | WeFlow 默认导出为 `其他消息`；Alpha 实际样本含 `其他` | Alpha 使用统一占位规则，不中断整份导入 |

WeFlow JSON 本来就接受中文类型标签，没有 Alpha 的固定名单限制；本次保持其原有字段与内容契约。统一 JSON v1 的 `type` 枚举也保持不变。

核对来源：[WeFlow 原始类型映射](https://github.com/hicccc77/WeFlow/blob/main/electron/services/export/constants.ts)、[WeFlow 导出标签与应用消息解析](https://github.com/hicccc77/WeFlow/blob/main/electron/services/export/core/ExportContext.ts)、[Alpha 官方功能说明](https://github.com/xunchahaha/alpha-releases)、[Alpha 导出接口](https://github.com/xunchahaha/alpha-releases/blob/main/docs/api/http-api.md)。Alpha 官方资料列有小程序、实况照片等能力，但未公开完整的 Markdown 类型枚举；因此适配采用可扩展的发生记录规则。

## 缓存与范围

任务请求记录实际输入文件名，停止后继续沿用原快照。旧网页 JSON 任务仍默认读取 `input.json`。输入适配器源码纳入流程指纹，规则改变后已有报告须重新运行，不能显式续跑旧流程缓存。

适配器读取与原始输入快照保持原样；分析前按 [数据脱敏规则](PRIVACY_REDACTION.md) 替换文本和身份元数据，再分配消息与证据编号。群聊分析、CSV/JSONL 与跨导出身份映射尚未提供。

## 验证

`tests/test_conversation_formats.py` 覆盖格式适配、角色与时间校验、消息来源、多行正文、媒体占位和网页导入；`tests/test_runtime_compatibility.py` 覆盖 Alpha 附件标签与正文隔离。

```bash
python -m unittest discover -s tests -p test_conversation_formats.py
python -m unittest discover -s tests -p test_runtime_compatibility.py
```

测试使用合成数据和模型替身，核对原始输入字节、来源位置和完整流程。
