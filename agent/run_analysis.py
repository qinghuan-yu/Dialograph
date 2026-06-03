"""
聊天记录分析 Agent - 主入口
运行此脚本将自动完成预处理和分析报告生成。
"""

import json
import sys
import os
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from analyze import (
    load_chat,
    parse_chat_messages,
    compute_statistics,
    extract_text_samples,
    extract_conversation_segments,
    format_statistics_report,
)
from config import load_llm_config
from output_paths import get_support_output_dir, sanitize_filename


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


def main():
    workspace = Path(__file__).parent.parent
    talks_dir = workspace / "talks"
    output_dir = workspace / "analysis"
    output_dir.mkdir(exist_ok=True)
    llm_config = load_llm_config(workspace)

    # 列出所有聊天记录
    json_files = list(talks_dir.glob("*.json"))
    if not json_files:
        print("未找到聊天记录文件。")
        sys.exit(1)

    print("=" * 50)
    print("聊天记录分析 Agent")
    print("=" * 50)
    print(f"LLM_API_BASE: {llm_config.api_base}")
    print(f"LLM_MODEL: {llm_config.model}")
    print(f"LLM 配置来源: {llm_config.source}")
    if not llm_config.api_key:
        print("LLM_API_KEY: 未设置，请在工作区根目录 .env 中填写")
    else:
        print("LLM_API_KEY: 已设置")
    print("\n可用的聊天记录：")
    for i, f in enumerate(json_files):
        data = load_chat(str(f))
        session = data.get("session", {})
        name = session.get("displayName", session.get("nickname", f.stem))
        count = session.get("messageCount", len(data.get("messages", [])))
        print(f"  [{i+1}] {name} ({count}条消息)")

    # 选择
    if len(sys.argv) > 1:
        try:
            choice = int(sys.argv[1]) - 1
        except ValueError:
            target = sys.argv[1]
            choice = None
            for i, f in enumerate(json_files):
                if target in f.stem:
                    choice = i
                    break
            if choice is None:
                print(f"无法匹配: {target}")
                sys.exit(1)
    else:
        try:
            choice = int(input("\n请输入要分析的编号: ")) - 1
        except (ValueError, EOFError):
            choice = 0

    if choice < 0 or choice >= len(json_files):
        print("无效选择")
        sys.exit(1)

    selected_file = json_files[choice]
    print(f"\n正在处理: {selected_file.name}")

    # 加载与解析
    data = load_chat(str(selected_file))
    session = data.get("session", {})
    raw_messages = data.get("messages", [])

    parse_result = parse_chat_messages(raw_messages)
    messages = parse_result["messages"]
    parse_diagnostics = parse_result["diagnostics"]

    print(f"有效消息数: {len(messages)}")
    if parse_diagnostics["sort_changed"]:
        print("提示：原始消息顺序已按时间戳重新排序。")
    if parse_diagnostics["skipped_message_count"] or parse_diagnostics["invalid_time_message_count"]:
        print(f"解析诊断: 跳过{parse_diagnostics['skipped_message_count']}条, 时间异常{parse_diagnostics['invalid_time_message_count']}条")
    if not messages:
        print(f"没有可分析的有效消息，解析诊断: {parse_diagnostics}")
        return 1

    # 统计
    stats = compute_statistics(messages, session)
    samples = extract_text_samples(messages, n_samples=150)
    segments = extract_conversation_segments(messages, n_segments=15)

    # 保存预处理报告
    name = session.get("displayName", session.get("nickname", "unknown"))
    report_text = format_statistics_report(stats, samples, segments)

    support_dir = get_support_output_dir(output_dir, sanitize_filename(name))
    support_dir.mkdir(parents=True, exist_ok=True)

    preproc_file = support_dir / f"预处理_{name}.txt"
    with open(preproc_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"预处理报告: {preproc_file}")

    stats_file = support_dir / f"统计_{name}.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计数据: {stats_file}")

    diagnostics_file = support_dir / f"解析诊断_{sanitize_filename(name)}.json"
    with open(diagnostics_file, "w", encoding="utf-8") as f:
        json.dump(parse_diagnostics, f, ensure_ascii=False, indent=2)
    print(f"解析诊断: {diagnostics_file}")

    # 生成分析 prompt
    prompt = generate_analysis_prompt(stats, samples, segments, session)
    prompt_file = support_dir / f"分析prompt_{name}.md"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"分析 prompt: {prompt_file}")

    print("\n" + "=" * 50)
    print("✅ 预处理完成！")
    print("=" * 50)
    print(f"\n接下来请使用 agent/SKILL.md 中的分析框架，")
    print(f"结合 {prompt_file} 中的数据，")
    print(f"生成最终分析报告。")
    print(f"\n在 VS Code Copilot Chat 中输入：")
    print(f'  @agent 请按照 agent/SKILL.md 的分析框架，')
    print(f'  分析 {prompt_file} 中的数据，')
    print(f'  生成分析报告保存到 analysis/分析_{name}.md')


if __name__ == "__main__":
    main()
