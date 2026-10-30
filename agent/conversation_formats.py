"""导出格式适配边界：统一协议、Alpha Markdown、既有 WeFlow JSON。"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from conversation_input import validate_weflow

SCHEMA = 'dialograph.conversation'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
EXPORT_TIMEZONE = timezone(timedelta(hours=8))
SUPPORTED_SUFFIXES = {'.json', '.md'}
MESSAGE_TYPES = {
    'text': '文本消息', 'quote': '引用消息', 'image': '图片消息',
    'emoji': '动画表情', 'voice': '语音消息', 'video': '视频消息',
    'file': '文件消息', 'link': '链接消息', 'system': '系统消息',
}
ALPHA_TYPES = {
    None: 'text', '文本': 'text', '引用': 'quote', '图片': 'image',
    '表情': 'emoji', '语音': 'voice', '视频': 'video', '文件': 'file',
    '链接': 'link', '系统': 'system',
}
ALPHA_OPAQUE_TYPES = {'聊天记录', '转账', '红包', '通话'}


def conversation_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def required_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} 必须是非空文本。')
    return value


def export_time(value):
    required_text(value, 'time')
    try:
        parsed = datetime.strptime(value, TIME_FORMAT)
    except ValueError as exc:
        raise ValueError('time 必须为有效的 YYYY-MM-DD HH:MM:SS 时间（UTC+08:00）。') from exc
    if parsed.strftime(TIME_FORMAT) != value:
        raise ValueError('time 必须使用固定的 YYYY-MM-DD HH:MM:SS 格式。')
    return int(parsed.replace(tzinfo=EXPORT_TIMEZONE).timestamp())


def normalize_conversation(data):
    """将明确声明版本的交换协议映射到现有分析器字段，不修改输入。"""
    if not isinstance(data, dict):
        raise ValueError('对话必须是 JSON 对象。')
    if 'schema' not in data:
        return validate_weflow(data)
    if data.get('schema') != SCHEMA or type(data.get('version')) is not int or data['version'] != 1:
        raise ValueError('不支持的对话协议；需要 dialograph.conversation version 1。')
    session = data.get('session')
    if not isinstance(session, dict) or session.get('type') != 'private':
        raise ValueError('统一协议当前仅支持两人私聊，session.type 必须为 private。')
    if any(source.get(flag) is True for source in (data, session) for flag in ('isGroup', 'isGroupChat')):
        raise ValueError('输入含有群聊标记，不能按两人私聊导入。')
    session_id = required_text(session.get('id'), 'session.id')
    if session_id.endswith('@chatroom'):
        raise ValueError('当前仅支持两人私聊，不支持群聊。')
    name = required_text(session.get('display_name'), 'session.display_name')
    participants = data.get('participants')
    if not isinstance(participants, list) or len(participants) != 2:
        raise ValueError('participants 必须恰有两人，并明确 me/other 归属。')
    roles, names = {}, {}
    for participant in participants:
        if not isinstance(participant, dict):
            raise ValueError('participants 条目必须是对象。')
        participant_id = required_text(participant.get('id'), 'participant.id')
        role = participant.get('role')
        if participant_id in roles or role not in ('me', 'other') or role in roles.values():
            raise ValueError('参与者 ID 必须唯一，并各有一个 me 和 other。')
        roles[participant_id] = role
        names[role] = required_text(participant.get('name'), 'participant.name')
    messages = data.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages 必须是非空数组。')
    normalized, ids = [], set()
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError('messages 条目必须是对象。')
        message_id = required_text(message.get('id'), 'message.id')
        if message_id in ids:
            raise ValueError('message.id 不能重复；重复文本仍须有不同的消息 ID。')
        ids.add(message_id)
        kind = message.get('type')
        if not isinstance(kind, str) or kind not in MESSAGE_TYPES:
            raise ValueError('未知消息类型；请使用统一协议规定的 type。')
        sender_id = message.get('sender_id')
        if kind == 'system' and sender_id is None:
            role = 'other'
        elif not isinstance(sender_id, str) or sender_id not in roles:
            raise ValueError('sender_id 必须对应已声明的参与者；不能猜测发送者。')
        else:
            role = roles[sender_id]
        content = message.get('content')
        if content is not None and not isinstance(content, str):
            raise ValueError('content 必须为文本或 null。')
        if kind in ('text', 'quote'):
            required_text(content, 'content')
        # 媒体只保留导出说明或占位符，不虚构图片、语音或文件内容。
        content = content if content and content.strip() else f'[{MESSAGE_TYPES[kind]}]'
        normalized.append({
            'type': MESSAGE_TYPES[kind], 'content': content, 'isSend': int(role == 'me'),
            'formattedTime': message.get('time'), 'createTime': export_time(message.get('time')),
            'source_message_id': message_id, 'senderUsername': sender_id,
        })
    return validate_weflow({
        'session': {'id': session_id, 'type': 'private', 'displayName': name,
                    'myDisplayName': names['me']},
        'messages': normalized, 'source_format': 'dialograph_v1',
    })


def markdown_unescape(value):
    # 仅还原 Alpha 导出中的 Markdown 转义，不执行任何 Markdown/HTML 内容。
    return re.sub(r'\\([\\`*_{}\[\]()#+.!>\-])', r'\1', value)


def parse_alpha_markdown(text):
    """解析已知的 Alpha 消息头布局；不将普通 Markdown 当聊天记录。"""
    text = text.replace('\r\n', '\n')
    envelope, separator, body = text.partition('\n---\n')
    if not separator or not envelope.startswith('# '):
        raise ValueError('需要 Alpha Markdown 导出，含会话、导出账号和消息分隔线。')
    metadata = {}
    for key in ('会话', '导出账号', '导出时间'):
        values = re.findall(r'^- ' + key + r'：(.+)$', envelope, re.MULTILINE)
        if len(values) != 1:
            raise ValueError(f'Alpha 导出缺少或重复 {key}；不能猜测发送者归属。')
        metadata[key] = markdown_unescape(values[0].strip().strip('`'))
        required_text(metadata[key], key)
    session_id, owner = metadata['会话'], metadata['导出账号']
    if session_id.endswith('@chatroom'):
        raise ValueError('Alpha 文件为群聊；当前仅支持两人私聊。')
    export_time(metadata['导出时间'])
    headers = list(re.finditer(r'^\*\*.+?\*\*(?: ·| \d{4}-)[^\n]*$', body, re.MULTILINE))
    if not headers or body[:headers[0].start()].strip():
        raise ValueError('Alpha 消息头缺失或前有无法识别的内容。')
    messages, senders = [], set()
    for index, header in enumerate(headers):
        match = re.fullmatch(r'\*\*(.+?)\*\* · (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: · (.+))?', header.group())
        if not match:
            raise ValueError(f'Alpha 第 {index + 1} 条消息头无效。')
        sender, time_str, label = match.groups()
        sender = markdown_unescape(sender)
        if label not in ALPHA_TYPES and label not in ALPHA_OPAQUE_TYPES:
            raise ValueError(f'Alpha 第 {index + 1} 条消息类型不支持：{label}。')
        content_end = headers[index + 1].start() if index + 1 < len(headers) else len(body)
        if label in ALPHA_OPAQUE_TYPES:
            # 转发内容与交易附件不作为发送者原话；只保留发生记录。
            kind = 'file'
            content = f'[原导出类型：{label}；仅保留发生记录，附件内容未解析]'
        else:
            kind = ALPHA_TYPES[label]
            content = markdown_unescape(body[header.end():content_end].strip('\n'))
        if kind != 'system':
            senders.add(sender)
        messages.append({'id': f'alpha:{index + 1}', 'sender_id': None if kind == 'system' else sender,
                         'time': time_str, 'type': kind, 'content': content})
    # Markdown 的消息头可能只有昵称；必须实际出现导出账号，且只能有一个其他标签。
    if owner not in senders or len(senders) != 2:
        raise ValueError('Alpha 消息头须明确出现导出账号及唯一对方；昵称无法对应账号时请转为统一 JSON 协议。')
    peer = next(sender for sender in senders if sender != owner)
    normalized = normalize_conversation({
        'schema': SCHEMA, 'version': 1,
        'session': {'id': session_id, 'type': 'private', 'display_name': peer},
        'participants': [{'id': owner, 'role': 'me', 'name': owner},
                         {'id': peer, 'role': 'other', 'name': peer}],
        'messages': messages,
    })
    normalized['source_format'] = 'alpha_markdown'
    return normalized


def read_conversation(raw: bytes, filename: str):
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError('请选择 WeFlow/统一协议 JSON，或 Alpha Markdown（.md）。')
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeError as exc:
        raise ValueError('导入文件必须使用 UTF-8 编码。') from exc
    if suffix == '.md':
        return parse_alpha_markdown(text)
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError('JSON 文件无法解析，请检查导出格式。') from exc
    return normalize_conversation(data)
