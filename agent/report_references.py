"""将报告中的证据缩写、消息编号和片段编号绑定到已校验的任务账本。"""
from collections import defaultdict
import re

from evidence_validation import FIELDS

REFERENCE_PATTERN = r'(?<!\w)(?:[EM]_[0-9a-f]{8,32}(?:…|\.{3})?|chunk_\d{1,6})(?!\w)'


class ReferenceCatalog:
    """输入须先通过任务原文及账本校验；片段和消息只作为来源展示。"""

    def __init__(self, ledger):
        self.messages = {message['message_id']: message for message in ledger['source_messages']}
        self.chunks = {f"chunk_{doc['chunk_index']:03d}": doc for doc in ledger['documents']}
        self.evidence = defaultdict(list)
        for doc in ledger['documents']:
            for category in FIELDS:
                for item in doc[category]:
                    self.evidence[item['evidence_id']].append((category, item, doc))

    def describe(self, token):
        if not isinstance(token, str) or not re.fullmatch(REFERENCE_PATTERN, token):
            raise ValueError('引用编号无效。')
        if token.startswith('chunk_'):
            canonical = f"chunk_{int(token[6:]):03d}"
            return {'kind': 'chunk', 'canonical': canonical,
                    'status': 'resolved' if canonical in self.chunks else 'missing'}
        kind, candidates = ('evidence', self.evidence) if token.startswith('E_') else ('message', self.messages)
        prefix = token.rstrip('.…')
        matches = [value for value in candidates if value.startswith(prefix)]
        result = {'kind': kind, 'status': 'missing' if not matches else 'ambiguous'}
        if len(matches) == 1:
            result.update(canonical=matches[0], status='resolved')
            if kind == 'evidence' and self.evidence[matches[0]][0][1]['grounding']['status'] in ('unmatched', 'ambiguous'):
                result['status'] = 'quarantined'
        return result

    def describe_report(self, text):
        text = re.sub(r'chunk_\d{1,6}(?:/\d{1,6})+', lambda match: match[0].replace('/', '/chunk_'), text)
        return {token: self.describe(token) for token in dict.fromkeys(re.findall(REFERENCE_PATTERN, text))}

    @staticmethod
    def message_ref(message):
        return {**{key: message[key] for key in ('message_id', 'original_index', 'position', 'sender', 'content')},
                'time': message['time_str'], 'start': 0, 'end': 0}

    def read(self, token):
        description = self.describe(token)
        if description['status'] != 'resolved':
            raise ValueError('这条引用尚不能唯一定位到可用来源。')
        canonical, kind = description['canonical'], description['kind']
        if kind == 'evidence':
            occurrences = []
            for category, item, doc in self.evidence[canonical]:
                refs = [{**ref, 'content': self.messages[ref['message_id']]['content']}
                        for ref in item['grounding']['source_refs']]
                occurrences.append({'chunk_index': doc['chunk_index'], 'source_refs': refs})
            return {**description, 'evidence_id': canonical, 'category': category, 'payload': item,
                    'grounding_status': item['grounding']['status'], 'quote': item.get('evidence', ''),
                    'occurrences': occurrences}
        if kind == 'message':
            messages = [self.messages[canonical]]
        else:
            span = self.chunks[canonical]['message_range']
            messages = [message for message in self.messages.values()
                        if span['start'] <= message['position'] <= span['end']]
        return {**description, 'grounding_status': 'source',
                'occurrences': [{'source_refs': [self.message_ref(message) for message in messages]}]}
