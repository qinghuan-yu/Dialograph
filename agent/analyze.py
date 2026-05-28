"""
聊天记录预处理与统计分析脚本
解析 WeFlow 导出的 JSON 聊天记录，提取结构化统计数据供 agent 深度分析。
"""

import json
import os
import sys
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from output_paths import get_support_output_dir, sanitize_filename


def load_chat(filepath: str) -> dict:
    """加载聊天记录 JSON 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_message(msg: dict) -> dict | None:
    """解析单条消息，提取关键字段"""
    msg_type = msg.get("type", "")
    content = msg.get("content", "")
    is_send = msg.get("isSend", 0)
    time_str = msg.get("formattedTime", "")
    timestamp = msg.get("createTime", 0)

    # 跳过无内容的系统消息和图片消息（内容为null）
    if msg_type == "系统消息":
        return None

    # 对于图片/动画表情，标记类型但不提取内容
    if msg_type in ("图片消息", "动画表情"):
        return {
            "type": msg_type,
            "content": f"[{msg_type}]",
            "is_send": is_send,
            "time_str": time_str,
            "timestamp": timestamp,
            "sender": "me" if is_send == 1 else "other",
        }

    if msg_type == "文本消息" and content:
        return {
            "type": msg_type,
            "content": content,
            "is_send": is_send,
            "time_str": time_str,
            "timestamp": timestamp,
            "sender": "me" if is_send == 1 else "other",
        }

    # 其他消息类型（语音、视频、链接等）
    if content:
        return {
            "type": msg_type,
            "content": content,
            "is_send": is_send,
            "time_str": time_str,
            "timestamp": timestamp,
            "sender": "me" if is_send == 1 else "other",
        }

    return None


def compute_statistics(messages: list[dict], session: dict) -> dict:
    """计算聊天统计数据"""
    if not messages:
        return {}

    other_name = session.get("displayName", session.get("nickname", "对方"))
    my_name = session.get("myDisplayName", "我")

    # === 基础统计 ===
    total = len(messages)
    my_msgs = [m for m in messages if m["sender"] == "me"]
    other_msgs = [m for m in messages if m["sender"] == "other"]

    # === 时间维度分析 ===
    # 按日期分组
    daily_counts = defaultdict(lambda: {"me": 0, "other": 0, "total": 0})
    hourly_counts = defaultdict(int)
    monthly_counts = defaultdict(lambda: {"me": 0, "other": 0, "total": 0})
    weekday_counts = defaultdict(int)

    for m in messages:
        date_str = m["time_str"][:10]  # "2024-10-10"
        month_str = m["time_str"][:7]  # "2024-10"
        hour = int(m["time_str"][11:13]) if len(m["time_str"]) > 13 else 0

        daily_counts[date_str][m["sender"]] += 1
        daily_counts[date_str]["total"] += 1
        monthly_counts[month_str][m["sender"]] += 1
        monthly_counts[month_str]["total"] += 1
        hourly_counts[hour] += 1

        try:
            dt = datetime.strptime(m["time_str"], "%Y-%m-%d %H:%M:%S")
            weekday_counts[dt.weekday()] += 1
        except (ValueError, IndexError):
            pass

    # === 对话轮次分析 ===
    # 定义"对话轮次"：同一人连续发消息算一轮，换人则开启新轮
    conversation_turns = []
    current_turn = {"sender": messages[0]["sender"], "msgs": [messages[0]["content"]], "start_time": messages[0]["time_str"]}

    for m in messages[1:]:
        if m["sender"] != current_turn["sender"]:
            conversation_turns.append(current_turn)
            current_turn = {"sender": m["sender"], "msgs": [m["content"]], "start_time": m["time_str"]}
        else:
            current_turn["msgs"].append(m["content"])
    conversation_turns.append(current_turn)

    # 话题发起者统计（每轮对话的第一个发言者）
    topic_starters = Counter(t["sender"] for t in conversation_turns)
    # 排除对话的第一轮（可能是任意一方）
    # 计算非连续对话间隔后的发起者
    gap_initiated = {"me": 0, "other": 0}
    for i, turn in enumerate(conversation_turns):
        if i == 0:
            continue
        # 如果两轮之间有较长间隔（>2小时），算作新话题发起
        try:
            prev_time = datetime.strptime(conversation_turns[i-1]["msgs"][-1] if conversation_turns[i-1]["msgs"] else turn["start_time"], "%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            pass

    # === 主动性分析 ===
    # 谁更常开启新的一天的对话
    daily_first_msg = {}
    for m in messages:
        date_str = m["time_str"][:10]
        if date_str not in daily_first_msg:
            daily_first_msg[date_str] = m["sender"]

    first_msg_counter = Counter(daily_first_msg.values())

    # === 回复速度分析 ===
    # 计算连续消息间的时间差
    reply_times = {"me": [], "other": []}
    for i in range(1, len(messages)):
        try:
            t1 = datetime.strptime(messages[i-1]["time_str"], "%Y-%m-%d %H:%M:%S")
            t2 = datetime.strptime(messages[i]["time_str"], "%Y-%m-%d %H:%M:%S")
            diff_seconds = (t2 - t1).total_seconds()
            # 只统计合理的回复间隔（5秒到2小时）
            if 5 < diff_seconds < 7200 and messages[i]["sender"] != messages[i-1]["sender"]:
                reply_times[messages[i]["sender"]].append(diff_seconds)
        except (ValueError, IndexError):
            pass

    # === 消息长度统计 ===
    my_text_msgs = [m for m in my_msgs if m["type"] == "文本消息" and m["content"]]
    other_text_msgs = [m for m in other_msgs if m["type"] == "文本消息" and m["content"]]

    my_avg_len = sum(len(m["content"]) for m in my_text_msgs) / max(len(my_text_msgs), 1)
    other_avg_len = sum(len(m["content"]) for m in other_text_msgs) / max(len(other_text_msgs), 1)

    # === 表情包/表情使用 ===
    my_emoji = len([m for m in my_msgs if m["type"] in ("动画表情", "图片消息")])
    other_emoji = len([m for m in other_msgs if m["type"] in ("动画表情", "图片消息")])

    # === 话题结束分析 ===
    # 谁更常发出每天最后一条消息
    daily_last_msg = {}
    for m in messages:
        date_str = m["time_str"][:10]
        daily_last_msg[date_str] = m["sender"]
    last_msg_counter = Counter(daily_last_msg.values())

    # === 月度趋势 ===
    sorted_months = sorted(monthly_counts.keys())
    monthly_trend = []
    for month in sorted_months:
        mc = monthly_counts[month]
        monthly_trend.append({
            "month": month,
            "my_count": mc["me"],
            "other_count": mc["other"],
            "total": mc["total"],
        })

    # === 最活跃日期 TOP 10 ===
    sorted_days = sorted(daily_counts.items(), key=lambda x: x[1]["total"], reverse=True)[:10]
    top_days = [{"date": d, "my": c["me"], "other": c["other"], "total": c["total"]} for d, c in sorted_days]

    # === 最近一个月的对话轮次样本 ===
    recent_turns = conversation_turns[-50:] if len(conversation_turns) > 50 else conversation_turns

    # === 深夜对话（23:00-05:00）===
    late_night_msgs = [m for m in messages if len(m["time_str"]) > 13 and int(m["time_str"][11:13]) in (23, 0, 1, 2, 3, 4)]

    def avg_or_none(lst):
        return round(sum(lst) / len(lst), 1) if lst else None

    def median_or_none(lst):
        if not lst:
            return None
        s = sorted(lst)
        n = len(s)
        return round(s[n // 2], 1) if n % 2 else round((s[n // 2 - 1] + s[n // 2]) / 2, 1)

    return {
        "other_name": other_name,
        "my_name": my_name,
        "total_messages": total,
        "my_message_count": len(my_msgs),
        "other_message_count": len(other_msgs),
        "first_date": messages[0]["time_str"][:10],
        "last_date": messages[-1]["time_str"][:10],
        "active_days": len(daily_counts),
        "daily_first_msg": dict(first_msg_counter),
        "daily_last_msg": dict(last_msg_counter),
        "avg_my_msg_length": round(my_avg_len, 1),
        "avg_other_msg_length": round(other_avg_len, 1),
        "my_emoji_count": my_emoji,
        "other_emoji_count": other_emoji,
        "my_text_msg_count": len(my_text_msgs),
        "other_text_msg_count": len(other_text_msgs),
        "reply_time_my_avg_seconds": avg_or_none(reply_times["me"]),
        "reply_time_other_avg_seconds": avg_or_none(reply_times["other"]),
        "reply_time_my_median_seconds": median_or_none(reply_times["me"]),
        "reply_time_other_median_seconds": median_or_none(reply_times["other"]),
        "reply_time_my_count": len(reply_times["me"]),
        "reply_time_other_count": len(reply_times["other"]),
        "conversation_turn_count": len(conversation_turns),
        "topic_starters": dict(topic_starters),
        "monthly_trend": monthly_trend,
        "top_active_days": top_days,
        "late_night_message_count": len(late_night_msgs),
        "hourly_distribution": {str(h): hourly_counts.get(h, 0) for h in range(24)},
        "weekday_distribution": {str(w): weekday_counts.get(w, 0) for w in range(7)},
    }


def extract_text_samples(messages: list[dict], n_samples: int = 100) -> list[dict]:
    """提取文本消息样本，用于深度语义分析"""
    text_msgs = [m for m in messages if m["type"] == "文本消息" and m["content"]]
    if len(text_msgs) <= n_samples:
        return text_msgs

    # 均匀采样：保留首尾和中间均匀分布的样本
    step = len(text_msgs) / n_samples
    indices = [int(i * step) for i in range(n_samples)]
    return [text_msgs[i] for i in indices]


def extract_conversation_segments(messages: list[dict], n_segments: int = 20) -> list[list[dict]]:
    """提取对话片段，每个片段包含连续的对话轮次"""
    segments = []
    current_segment = []
    last_time = None

    for m in messages:
        try:
            current_time = datetime.strptime(m["time_str"], "%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            continue

        if last_time and (current_time - last_time).total_seconds() > 3600 * 4:
            # 超过4小时间隔，开启新片段
            if current_segment:
                segments.append(current_segment)
            current_segment = []
        current_segment.append(m)
        last_time = current_time

    if current_segment:
        segments.append(current_segment)

    # 选择代表性片段：首尾 + 中间均匀分布 + 最长片段
    if len(segments) <= n_segments:
        return segments

    # 按长度排序，取最长的几个
    sorted_by_length = sorted(segments, key=len, reverse=True)[:5]
    # 均匀采样其余
    remaining = [s for s in segments if s not in sorted_by_length]
    step = max(1, len(remaining) // (n_segments - 5))
    sampled = remaining[::step][:n_segments - 5]

    # 合并并按时间排序
    selected = sorted_by_length + sampled
    selected.sort(key=lambda s: s[0]["timestamp"] if s else 0)
    return selected[:n_segments]


def format_statistics_report(stats: dict, samples: list[dict], segments: list[list[dict]]) -> str:
    """将统计数据格式化为结构化文本报告"""
    report = []
    report.append("=" * 60)
    report.append("聊天记录预处理统计报告")
    report.append("=" * 60)

    report.append(f"\n## 基本信息")
    report.append(f"- 对方昵称/备注: {stats['other_name']}")
    report.append(f"- 我的昵称: {stats['my_name']}")
    report.append(f"- 聊天时间跨度: {stats['first_date']} ~ {stats['last_date']}")
    report.append(f"- 活跃天数: {stats['active_days']} 天")
    report.append(f"- 总消息数: {stats['total_messages']}")
    report.append(f"- 我发送: {stats['my_message_count']} 条 ({stats['my_message_count']/stats['total_messages']*100:.1f}%)")
    report.append(f"- 对方发送: {stats['other_message_count']} 条 ({stats['other_message_count']/stats['total_messages']*100:.1f}%)")
    report.append(f"- 每日平均消息数: {stats['total_messages']/max(stats['active_days'],1):.1f} 条")

    report.append(f"\n## 主动性指标")
    report.append(f"- 每日首条消息发起者: 我={stats['daily_first_msg'].get('me',0)}天, 对方={stats['daily_first_msg'].get('other',0)}天")
    report.append(f"- 每日末条消息发送者: 我={stats['daily_last_msg'].get('me',0)}天, 对方={stats['daily_last_msg'].get('other',0)}天")

    report.append(f"\n## 消息特征")
    report.append(f"- 我的平均文本消息长度: {stats['avg_my_msg_length']} 字")
    report.append(f"- 对方平均文本消息长度: {stats['avg_other_msg_length']} 字")
    report.append(f"- 我的表情/图片消息数: {stats['my_emoji_count']}")
    report.append(f"- 对方的表情/图片消息数: {stats['other_emoji_count']}")
    report.append(f"- 深夜消息数(23-5点): {stats['late_night_message_count']}")

    report.append(f"\n## 回复速度")
    report.append(f"- 我的平均回复间隔: {stats['reply_time_my_avg_seconds']}秒 (样本{stats['reply_time_my_count']}次)")
    report.append(f"- 对方平均回复间隔: {stats['reply_time_other_avg_seconds']}秒 (样本{stats['reply_time_other_count']}次)")
    report.append(f"- 我的中位回复间隔: {stats['reply_time_my_median_seconds']}秒")
    report.append(f"- 对方中位回复间隔: {stats['reply_time_other_median_seconds']}秒")

    report.append(f"\n## 对话轮次")
    report.append(f"- 总对话轮次: {stats['conversation_turn_count']}")
    report.append(f"- 话题发起统计: 我={stats['topic_starters'].get('me',0)}次, 对方={stats['topic_starters'].get('other',0)}次")

    report.append(f"\n## 月度趋势")
    for mt in stats['monthly_trend']:
        report.append(f"  {mt['month']}: 总{mt['total']}条 (我{mt['my_count']}, 对方{mt['other_count']})")

    report.append(f"\n## 最活跃日期 TOP 10")
    for td in stats['top_active_days']:
        report.append(f"  {td['date']}: 总{td['total']}条 (我{td['my']}, 对方{td['other']})")

    report.append(f"\n## 每小时分布")
    for h in range(24):
        count = stats['hourly_distribution'].get(str(h), 0)
        if count > 0:
            bar = "█" * min(count // 10, 50)
            report.append(f"  {h:02d}:00  {count:>5} {bar}")

    report.append(f"\n## 对话片段样本（{len(segments)}个代表性片段）")
    for i, seg in enumerate(segments):
        report.append(f"\n--- 片段 {i+1} ({seg[0]['time_str'][:10]}) ---")
        for m in seg:
            prefix = "【我】" if m["sender"] == "me" else f"【{stats['other_name']}】"
            content = m["content"] if len(m["content"]) < 100 else m["content"][:100] + "..."
            report.append(f"  {m['time_str'][11:16]} {prefix} {content}")

    report.append(f"\n## 文本消息样本（{len(samples)}条均匀采样）")
    for m in samples:
        prefix = "【我】" if m["sender"] == "me" else f"【{stats['other_name']}】"
        content = m["content"] if len(m["content"]) < 150 else m["content"][:150] + "..."
        report.append(f"  {m['time_str']} {prefix} {content}")

    return "\n".join(report)


def format_full_transcript(messages: list[dict], session: dict) -> str:
    """导出完整解析后的聊天文本，按时间顺序保留全部有效消息。"""
    other_name = session.get("displayName", session.get("nickname", "对方"))
    my_name = session.get("myDisplayName", "我")

    lines = []
    lines.append("=" * 60)
    lines.append("聊天记录全量解析文本")
    lines.append("=" * 60)
    lines.append(f"- 对方昵称/备注: {other_name}")
    lines.append(f"- 我的昵称: {my_name}")
    lines.append(f"- 有效消息数: {len(messages)}")
    lines.append("")
    lines.append("## 全量消息")

    for message in messages:
        speaker = my_name if message["sender"] == "me" else other_name
        lines.append(f"{message['time_str']} 【{speaker}】 {message['content']}")

    return "\n".join(lines)


def main():
    """主函数"""
    workspace = Path(__file__).parent.parent
    talks_dir = workspace / "talks"
    output_dir = workspace / "analysis"
    output_dir.mkdir(exist_ok=True)

    # 列出所有聊天记录文件
    json_files = list(talks_dir.glob("*.json"))
    if not json_files:
        print("未找到聊天记录文件，请检查 talks/ 目录。")
        sys.exit(1)

    print("=" * 50)
    print("可用的聊天记录：")
    print("=" * 50)
    for i, f in enumerate(json_files):
        data = load_chat(str(f))
        session = data.get("session", {})
        name = session.get("displayName", session.get("nickname", f.stem))
        count = session.get("messageCount", len(data.get("messages", [])))
        print(f"  [{i+1}] {name} ({count}条消息) - {f.name}")

    # 选择要分析的聊天记录
    if len(sys.argv) > 1:
        try:
            choice = int(sys.argv[1]) - 1
        except ValueError:
            # 尝试按文件名匹配
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
    print(f"\n正在分析: {selected_file.name}")

    # 加载数据
    data = load_chat(str(selected_file))
    session = data.get("session", {})
    raw_messages = data.get("messages", [])

    # 解析消息
    messages = []
    for msg in raw_messages:
        parsed = parse_message(msg)
        if parsed:
            messages.append(parsed)

    print(f"有效消息数: {len(messages)}")

    # 计算统计
    stats = compute_statistics(messages, session)

    # 提取样本
    samples = extract_text_samples(messages, n_samples=150)
    segments = extract_conversation_segments(messages, n_segments=15)

    # 生成报告
    report = format_statistics_report(stats, samples, segments)

    # 保存预处理报告
    name = session.get("displayName", session.get("nickname", "unknown"))
    support_dir = get_support_output_dir(output_dir, sanitize_filename(name))
    support_dir.mkdir(parents=True, exist_ok=True)

    preproc_file = support_dir / f"预处理_{name}.txt"
    with open(preproc_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n预处理统计报告已保存: {preproc_file}")

    # 保存原始统计数据为 JSON（供 agent 使用）
    stats_file = support_dir / f"统计_{name}.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"统计数据已保存: {stats_file}")

    # 输出关键指标摘要
    print("\n" + "=" * 50)
    print("关键指标摘要")
    print("=" * 50)
    print(f"聊天跨度: {stats['first_date']} ~ {stats['last_date']}")
    print(f"活跃天数: {stats['active_days']}天")
    print(f"消息比例: 我{stats['my_message_count']} : 对方{stats['other_message_count']}")
    print(f"主动发起: 我{stats['daily_first_msg'].get('me',0)}天 : 对方{stats['daily_first_msg'].get('other',0)}天")
    print(f"平均回复: 我{stats['reply_time_my_avg_seconds']}秒 : 对方{stats['reply_time_other_avg_seconds']}秒")
    print(f"表情使用: 我{stats['my_emoji_count']}次 : 对方{stats['other_emoji_count']}次")

    return stats, report


if __name__ == "__main__":
    main()
