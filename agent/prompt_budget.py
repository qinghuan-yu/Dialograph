"""按完整条目、证据方向和时间轴分配预算；阶段 Markdown 独立处理。"""
from __future__ import annotations

import re
from collections import deque

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
        return [items[(len(items) - 1) // 2]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def spread_order(indices):
    """先选中间，再交替覆盖两侧；增加预算时保留已有选择。"""
    values = list(indices)
    if not values:
        return
    middle = (len(values) - 1) // 2
    priority = list(dict.fromkeys((values[middle], values[0], values[-1])))
    yield from priority
    pending = deque([values[:middle], values[middle + 1:]])
    while pending:
        values = pending.popleft()
        if values:
            middle = (len(values) - 1) // 2
            if values[middle] not in priority:
                yield values[middle]
            pending.extend((values[:middle], values[middle + 1:]))


def _balanced_order(rows):
    buckets = {}
    for index, row in enumerate(rows):
        match = re.search(r'/ (support|against|mixed):|^- (支持|反证)[：:]', row)
        if match:
            direction = next(part for part in match.groups() if part)
        elif row.startswith('|'):
            direction = 'table'
        elif re.match(r'^\d+[.)] ', row):
            direction = 'numbered'
        elif not row.startswith('- '):
            direction = 'paragraph'
        else:
            direction = 'neutral'
        buckets.setdefault(direction, []).append(index)
    queues = [deque(spread_order(indices)) for indices in buckets.values()]
    while any(queues):
        for queue in queues:
            if queue:
                yield queue.popleft()


def _pack_sections(sections, max_bytes, render, priority_ids=None):
    selected = [set() for _ in sections]
    if text_bytes(render(selected)) > max_bytes:
        return limit_text_bytes('[预算不足以展示材料结构；本部分全部省略。]', max(0, max_bytes))
    if priority_ids:
        priority_queues = []
        for _, rows in sections:
            matched = [
                index for index, row in enumerate(rows)
                if any(eid in priority_ids for eid in re.findall(r'\bE_[0-9a-f]{32}\b', row))
            ]
            priority_queues.append(deque(spread_order(matched)))
        while any(priority_queues):
            for number, queue in enumerate(priority_queues):
                if not queue:
                    continue
                index = queue.popleft()
                selected[number].add(index)
                if text_bytes(render(selected)) > max_bytes:
                    selected[number].remove(index)
    queues = [
        deque(index for index in _balanced_order(rows) if index not in selected[number])
        for number, (_, rows) in enumerate(sections)
    ]
    while any(queues):
        for number, queue in enumerate(queues):
            if not queue:
                continue
            index = queue.popleft()
            selected[number].add(index)
            if text_bytes(render(selected)) > max_bytes:
                selected[number].remove(index)
    return render(selected)


def _time_coverage(rows, chosen):
    parts = []
    for number in range(min(4, len(rows))):
        start, end = number * len(rows) // 4, (number + 1) * len(rows) // 4
        # 少于四条时每个条目自成一段。
        if len(rows) < 4:
            start, end = number, number + 1
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', '\n'.join(rows[start:end]))
        label = f'{dates[0]}~{dates[-1]}' if dates else f'段{number + 1}'
        parts.append(f'{label} {sum(i in chosen for i in range(start, end))}/{end - start}')
    return '；'.join(parts) or '无候选'


def budget_evidence_summary(text: str, max_bytes: int, *, priority_ids: set[str] | None = None) -> str:
    """五类轮流使用预算，类内支持/反证轮流选取；不截断条目。"""
    parts = re.split(r'(?m)^(### .+)$', text)
    if len(parts) == 1:
        return text if text_bytes(text) <= max_bytes else '[证据摘要超出预算，完整条目未展示。]'[:max(0, max_bytes // 3)]
    sections = [(title, [line for line in body.splitlines() if line.startswith('- ')
                         and not line.startswith(('- 总数 ', '- 未匹配', '- 说明', '- 预算', '- 选择 ', '- 时间轴'))
                         and not re.match(r'- \d+ 条非对方', line)])
                for title, body in zip(parts[1::2], parts[2::2])]
    metadata = [[line for line in body.splitlines() if line.startswith(('- 总数 ', '- 未匹配'))
                 or re.match(r'- \d+ 条非对方', line)] for body in parts[2::2]]

    def render(selected):
        lines = ['## 结构化证据摘要', '[按完整条目选择；存在省略，不能把省略当作不存在。]']
        for (title, rows), chosen, notes in zip(sections, selected, metadata):
            lines.extend((title, *notes, f'- 选择 {len(chosen)}/{len(rows)} 条；时间轴：{_time_coverage(rows, chosen)}。'))
            lines.extend(row for index, row in enumerate(rows) if index in chosen)
        return '\n'.join(lines)

    return _pack_sections(sections, max_bytes, render, priority_ids=priority_ids)


def _markdown_blocks(text):
    """表格、段落、编号项和代码围栏作为完整单元保留。"""
    blocks, current = [], []
    fence = None
    for line in text.splitlines():
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            if fence is None:
                if current:
                    blocks.append('\n'.join(current))
                current, fence = [line], marker[1][0]
            else:
                current.append(line)
                if marker[1][0] == fence:
                    blocks.append('\n'.join(current))
                    current, fence = [], None
        elif fence:
            current.append(line)
        elif not line.strip():
            if current:
                blocks.append('\n'.join(current))
                current = []
        else:
            if re.match(r'^(?:[-*+] |\d+[.)] )', line) and current:
                blocks.append('\n'.join(current))
                current = []
            current.append(line)
    if current:
        blocks.append('\n'.join(current))
    return blocks


def budget_phase_summaries(summaries: list[str], max_bytes: int) -> str:
    summaries = [part.strip() for text in summaries
                 for part in re.split(r'(?m)(?=^## (?:结构化证据摘要|阶段总结[^\n]*)$)', text) if part.strip()]
    joined = '\n\n'.join(summaries)
    if text_bytes(joined) <= max_bytes:
        return joined
    sections = []
    chosen_summaries = evenly_spaced(summaries, min(len(summaries), 6))
    for summary in chosen_summaries:
        parts = re.split(r'(?m)^(### .+)$', summary)
        batch = parts[0].split('\n', 1)[0]
        introduction = parts[0].partition('\n')[2].strip()
        if introduction:
            sections.append((batch, _markdown_blocks(introduction)))
        for title, body in zip(parts[1::2], parts[2::2]):
            sections.append((batch + '\n' + title, _markdown_blocks(body)))

    def render(selected):
        lines = [f'[阶段材料共 {len(summaries)} 份，预算选取 {len(chosen_summaries)} 份；按完整 Markdown 单元选择，存在省略。阶段解释均待原文复核。]']
        for (title, blocks), chosen in zip(sections, selected):
            lines.extend((title, f'[选择 {len(chosen)}/{len(blocks)} 个单元]'))
            lines.extend(block for index, block in enumerate(blocks) if index in chosen)
        return '\n\n'.join(lines)

    return _pack_sections(sections, max_bytes, render)
