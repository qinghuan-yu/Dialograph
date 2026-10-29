"""采样预览提示词模板。"""

def generate_analysis_prompt(stats: dict, samples: list, segments: list, session: dict) -> str:
    """生成完整的分析 prompt，用于发送给 LLM"""
    name = session.get("displayName", session.get("nickname", "对方"))

    prompt = f"""请基于以下聊天记录统计数据和样本，对「{name}」进行关系与行为模式分析。

## 分析要求

请严格遵守以下原则：
1. 严格区分：可观察事实 / 基于事实的合理推断 / 可能但证据不足的假设 / 不能判断的部分
2. 所有判断必须引用聊天记录中的具体片段或概括性证据
3. 不预设对方性别、性取向、关系目标或恋爱/暧昧前提
4. 除直接引用原文外，全文使用“对方/此人/该对象/对象昵称”，不要使用“他/她/ta/男方/女方”等带性别预设的代称
5. 不要因为单个亲密、玩笑或特殊互动片段就放大结论
6. 不要迎合我的期待
7. 请优先寻找反证
8. 输出要冷静、克制、证据导向

请从以下八个维度分析：

### 一、互动概况
分析聊天频率变化、主动开启话题比例、谁更常延展/结束话题、关系升温/降温节点、重要事件与互动转折。

### 二、对方的表达风格
分析语言风格、口癖、情绪表达方式、是否直接/委婉、是否常用玩笑/吐槽/反问/表情包、轻松话题与严肃话题中的差异。

### 三、对方的互动模式
分析是否主动关心、是否记得你说过的事情、是否会追问你的状态、面对你的情绪表达时如何回应、面对边界/冲突/求助/合作/日常话题时的不同反应、是否存在忽冷忽热/回避/低成本维持/工具性联系等模式。

### 四、对方在这段关系中的可能定位
开放式评估以下每个模型的支持度，不把恋爱或暧昧作为默认前提。每个模型都要给出支持证据、反证和可信度：
- 普通熟人
- 普通朋友
- 较亲近朋友
- 工具性/事务性关系
- 社群/同学/同事式关系
- 情绪支持或情绪依赖
- 礼貌维持
- 低成本社交
- 亲密或暧昧可能（仅在证据明确支持时评估）

### 五、对方可能的动机逻辑
提出3到5种可能解释，每种都要包含：支持证据、反证、可信度、还需要什么信息才能验证。

### 六、对方在你面前呈现出的性格倾向
分析主动性、共情能力、边界感、情绪稳定性、关系投入方式、回避倾向、责任感、表达成熟度等维度的"倾向"（不做医学化、标签化诊断）。

### 七、你自己的误判风险
分析你可能存在的投射、过度解读、自我安慰、选择性注意、稀缺性滤镜、特殊关系放大等问题。直接指出哪些判断可能站不住脚。

### 八、总结
输出：
1. 最稳妥的结论
2. 中等可信的判断
3. 不能下结论的部分
4. 如果你要验证关系定位或互动边界，接下来最合适的低风险行动是什么

## 格式要求
- 使用以下标记区分证据层次：【事实】【推断】【假设】【存疑】
- 引用聊天记录时注明时间或上下文
- 输出为 Markdown 格式

---

## 聊天统计数据

"""
    # 加入统计数据
    prompt += f"### 基本信息\n"
    prompt += f"- 对方: {stats['other_name']}\n"
    prompt += f"- 聊天跨度: {stats['first_date']} ~ {stats['last_date']}\n"
    prompt += f"- 活跃天数: {stats['active_days']} 天\n"
    prompt += f"- 总消息数: {stats['total_messages']}\n"
    prompt += f"- 我发送: {stats['my_message_count']} 条 ({stats['my_message_count']/stats['total_messages']*100:.1f}%)\n"
    prompt += f"- 对方发送: {stats['other_message_count']} 条 ({stats['other_message_count']/stats['total_messages']*100:.1f}%)\n"

    prompt += f"\n### 主动性指标\n"
    prompt += f"- 每日首条消息: 我={stats['daily_first_msg'].get('me',0)}天, 对方={stats['daily_first_msg'].get('other',0)}天\n"
    prompt += f"- 每日末条消息: 我={stats['daily_last_msg'].get('me',0)}天, 对方={stats['daily_last_msg'].get('other',0)}天\n"
    prompt += f"- 会话切分间隔: {stats.get('session_gap_hours', 4)}小时\n"
    prompt += f"- 总会话数: {stats.get('session_count', 0)}\n"
    prompt += f"- 会话发起者: 我={stats.get('session_initiators', {}).get('me',0)}次, 对方={stats.get('session_initiators', {}).get('other',0)}次\n"
    prompt += f"- 会话收尾者: 我={stats.get('session_closers', {}).get('me',0)}次, 对方={stats.get('session_closers', {}).get('other',0)}次\n"
    prompt += f"- 平均每会话消息数: {stats.get('avg_session_message_count')} 条, 中位 {stats.get('median_session_message_count')} 条\n"

    prompt += f"\n### 消息特征\n"
    prompt += f"- 我的平均文本消息长度: {stats['avg_my_msg_length']} 字\n"
    prompt += f"- 对方平均文本消息长度: {stats['avg_other_msg_length']} 字\n"
    prompt += f"- 我的表情/图片: {stats['my_emoji_count']}次\n"
    prompt += f"- 对方表情/图片: {stats['other_emoji_count']}次\n"
    prompt += f"- 深夜消息(23-5点): {stats['late_night_message_count']}条\n"

    prompt += f"\n### 回复速度\n"
    prompt += f"- 我平均回复: {stats['reply_time_my_avg_seconds']}秒 (中位{stats['reply_time_my_median_seconds']}秒)\n"
    prompt += f"- 对方平均回复: {stats['reply_time_other_avg_seconds']}秒 (中位{stats['reply_time_other_median_seconds']}秒)\n"
    prompt += f"- 会话内我平均回复: {stats.get('session_reply_time_my_avg_seconds')}秒\n"
    prompt += f"- 会话内对方平均回复: {stats.get('session_reply_time_other_avg_seconds')}秒\n"

    prompt += f"\n### 月度趋势\n"
    for mt in stats['monthly_trend']:
        prompt += f"- {mt['month']}: 总{mt['total']}条 (我{mt['my_count']}, 对方{mt['other_count']})\n"

    prompt += f"\n### 对话片段样本\n"
    for i, seg in enumerate(segments):
        prompt += f"\n**片段 {i+1}** ({seg[0]['time_str'][:10]})\n"
        for m in seg:
            prefix = "【我】" if m["sender"] == "me" else f"【{stats['other_name']}】"
            content = m["content"][:100] + "..." if len(m["content"]) > 100 else m["content"]
            prompt += f"  {m['time_str'][11:16]} {prefix} {content}\n"

    prompt += f"\n### 文本消息采样\n"
    for m in samples:
        prefix = "【我】" if m["sender"] == "me" else f"【{stats['other_name']}】"
        content = m["content"][:120] + "..." if len(m["content"]) > 120 else m["content"]
        prompt += f"  {m['time_str']} {prefix} {content}\n"

    return prompt
