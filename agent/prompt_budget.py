"""按证据类别和材料首尾分配字节预算，明确披露省略范围。"""
from __future__ import annotations

import re

TRUNCATION_NOTE = '\n[内容已按提示词预算截断；首尾抽取，存在省略。]'


def text_bytes(text: str) -> int:
    return len(text.encode('utf-8'))


def limit_text_bytes(text: str, max_bytes: int) -> str:
    if text_bytes(text) <= max_bytes:
        return text
    note = TRUNCATION_NOTE
    if text_bytes(note) >= max_bytes:
        return note.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')
    available = max_bytes - text_bytes(note)
    encoded = text.encode('utf-8')
    head = encoded[:available // 2].decode('utf-8', errors='ignore')
    tail = encoded[-(available - available // 2):].decode('utf-8', errors='ignore')
    return head + note + tail


def evenly_spaced(items, count):
    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [items[-1]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def budget_evidence_summary(text: str, max_bytes: int) -> str:
    """五类章节各自保底；条目均匀选取，避免前类挤掉反证。"""
    if text_bytes(text) <= max_bytes:
        return text
    parts = re.split(r'(?m)^(### .+)$', text)
    if len(parts) == 1:
        return limit_text_bytes(text, max_bytes)
    sections = list(zip(parts[1::2], parts[2::2]))
    prefix = '## 结构化证据摘要\n[内容已按提示词预算截断；各类及时间轴均匀选取，存在省略。]\n'
    quota = (max_bytes - text_bytes(prefix)) // len(sections)
    rendered = []
    for title, body in sections:
        # 原摘要已完成原文绑定筛选；此处只选择，不恢复隔离条目。
        rows = [line for line in body.splitlines() if line.startswith('- ')
                and not line.startswith(('- 总数 ', '- 未匹配'))]
        header = title + '\n'
        note = f'- 预算抽取；本节候选 {len(rows)} 条，未展示部分省略。\n'
        available = max(0, quota - text_bytes(header + note) - 1)
        count = min(len(rows), max(1, available // 420))
        selected = evenly_spaced(rows, count)
        row_quota = max(0, available // max(1, len(selected)) - 1)
        rendered.append(header + note + '\n'.join(limit_text_bytes(row, row_quota) for row in selected))
    return '\n'.join([prefix, *rendered])


def budget_phase_summaries(summaries: list[str], max_bytes: int) -> str:
    summaries = [part.strip() for text in summaries
                 for part in re.split(r'(?m)(?=^## 结构化证据摘要$)', text) if part.strip()]
    joined = '\n\n'.join(summaries)
    if text_bytes(joined) <= max_bytes:
        return joined
    # 选择完整材料的首尾及中间位置，每份材料内部仍保留各类别。
    selected = evenly_spaced(summaries, min(len(summaries), 6))
    note = f'[阶段材料共 {len(summaries)} 份，预算选取 {len(selected)} 份；存在省略，不代表模型逐条看到全量证据。]\n'
    quota = max(0, (max_bytes - text_bytes(note) - 2 * len(selected)) // max(1, len(selected)))
    return note + '\n\n'.join(budget_evidence_summary(text, quota) for text in selected)
