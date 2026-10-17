"""共享的 WeFlow 私聊契约，避免网页与 CLI 静默使用不可靠归属。"""
from __future__ import annotations
import re

IDENTITY_FIELDS = ('senderUsername', 'senderUserName', 'senderId', 'sender_id', 'senderUin')
GROUP_TYPES = {'group', 'groupchat', 'group_chat', 'chatroom', '群聊'}


def validate_weflow(data: dict) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get('session', {}), dict):
        raise ValueError('需要包含 session 对象和 messages 数组的 WeFlow JSON。')
    messages = data.get('messages')
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
        raise ValueError('messages 必须是非空的消息对象数组。')
    session = data.get('session', {})
    for flag in ('isGroup', 'isGroupChat'):
        if session.get(flag) is True or data.get(flag) is True:
            raise ValueError('当前仅支持两人私聊，不能把群聊合并为一个对方。')
    session_type = str(session.get('type', '')).lower()
    if session_type in GROUP_TYPES or any(str(session.get(k, '')).endswith('@chatroom') for k in ('username', 'userName', 'id')):
        raise ValueError('当前仅支持两人私聊，不支持群聊。')
    participants = session.get('participants')
    if isinstance(participants, list) and len(participants) > 2:
        raise ValueError('检测到多个参与者；当前仅支持两人私聊。')
    for field in ('displayName', 'nickname', 'myDisplayName'):
        if field in session and (not isinstance(session[field], str) or not session[field].strip()):
            raise ValueError(f'session.{field} 必须是非空文本。')
    # 身份只用稳定账号字段判断，不把昵称变更误识别为群聊。
    identities = {field: {0: set(), 1: set()} for field in IDENTITY_FIELDS}
    for message in messages:
        if 'isSend' not in message or 'formattedTime' not in message:
            raise ValueError('需要 WeFlow 原始字段 isSend、formattedTime；不支持缺少发送者归属的数据。')
        if type(message['isSend']) is not int or message['isSend'] not in (0, 1):
            raise ValueError('isSend 必须为整数 0 或 1。')
        if not isinstance(message['formattedTime'], str) or not isinstance(message.get('type', ''), str):
            raise ValueError('formattedTime 和 type 必须是文本。')
        if not re.search(r'[\u4e00-\u9fff]', message.get('type', '')):
            raise ValueError('type 需要 WeFlow 中文消息类型（如文本消息），不能把 text/image 等其他约定直接作为 WeFlow 导入。')
        if message.get('content') is not None and not isinstance(message['content'], str):
            raise ValueError('content 必须是文本或 null。')
        if message.get('type') == '系统消息':
            continue
        for field in IDENTITY_FIELDS:
            value = message.get(field)
            if value is not None and value != '':
                if type(value) not in (int, str):
                    raise ValueError(f'{field} 必须为账号字符串或整数。')
                identities[field][message['isSend']].add(str(value))
    if any(len(values) > 1 for roles in identities.values() for values in roles.values()):
        raise ValueError('检测到同一方有多个发送者账号，当前仅支持两人私聊。')
    return data
