"""核对复核器明确列出的出处，区分原文可见与本条已登记引用。"""
from copy import deepcopy


BASIS_VERSION = 'claim-field-review-v3'
BASIS_INSTRUCTIONS = '''
先为每个field列basis，再完成四个检查。basis逐项对应本字段中的实际断言，statement必须是字段原样子串，sources列出该断言实际需要的来源和逐字引语。
不要只从registered_citations挑编号：若实际依据在其他可见原文，仍照实列其P_序号，程序会核对是否漏登记。依据不存在时不要伪造来源，content_supported须false并在reason说明。
仅说明资料不足/外部结果未确认的部分可不列basis，也可列出statement并将sources留空；其他事实、解释和范围依据逐项列出。完整字段均是未知说明时basis可以为空。来源只列与本断言相关的短句，保留原话限定词，不把邻文当成本句引语。
sources不限一个。需要两条消息才能判断动作对象或对应关系时同时列出。引用消息中内层引语只解释对象，不能自动当作外层发言者自己的行动。
每个来源的usage为assertion或context。assertion表示该字段实际声称的事实或关系依据，须登记本条引用；context仅用来辨认话题/指代，不把其中额外动作写进当前判断。可省略usage，程序按assertion处理。不要因某条原文有帮助就当成必需事实引用，也不能把已写进判断的动作、结果、先后或配对关系标为context来绕过漏引。配对关系两端均属assertion。
可见的图片、文件、通话等占位文字可按逐字标签列为context，仅用于辨认话题。标签不提供附件内容，也不能证明执行或确认了结果；这类断言仍需可引用的文本依据。
“分别”“各自”“均”“都”表达的是逐项对应关系：每对承认与回应属于同一事件才成立。各个成员在不同原文中分别出现不等于配对正确。若对应松散、跨事件错配或存在其他解释，content_supported应false，理由明确指出哪一对不成立，不能写有点松散后仍给true。
reason与布尔值必须一致：若理由确认本字段存在未登记事实、对应错误、确定程度提高或已知后续缺失，相应布尔值须false。
'''


def citation_catalog(citations):
    return {row['id']: {'quote': row['quote'], 'message_id': row['source']['message_id']}
            for row in citations}


def map_basis_ids(checks, identities):
    """复核短编号落盘时还原为来源编号，避免依赖临时排列。"""
    for row in checks['field_checks']:
        for item in row.get('basis', []):
            for source in item['sources']:
                source['id'] = identities[source['id']]


def basis_source_ids(checks):
    """只收集可读取的编号；畸形结构仍由协议解析器统一拒绝。"""
    rows = checks.get('field_checks')
    result = set()
    for row in rows if isinstance(rows, list) else []:
        basis = row.get('basis') if isinstance(row, dict) else None
        for item in basis if isinstance(basis, list) else []:
            sources = item.get('sources') if isinstance(item, dict) else None
            for source in sources if isinstance(sources, list) else []:
                if isinstance(source, dict) and isinstance(source.get('id'), str):
                    result.add(source['id'])
    return result


def check_field_basis(row, data):
    """逐字核验后返回引用遗漏；不替代模型对断言与引语含义的判断。"""
    basis = row['basis']
    if not isinstance(basis, list):
        raise RuntimeError('逐字段复核basis必须为数组。')
    catalog = data.get('_basis_sources', {})
    registered = data.get('registered_citations', [])
    missing = []
    for item in basis:
        if (not isinstance(item, dict) or set(item) != {'statement', 'sources'}
                or not isinstance(item['statement'], str) or not item['statement'].strip()
                or item['statement'] not in data['statement_fields'][row['field']]
                or not isinstance(item['sources'], list)):
            raise RuntimeError('逐字段复核basis须指向该字段原样片段并列出来源。')
        for source in item['sources']:
            if (not isinstance(source, dict) or set(source) not in ({'id', 'quote'}, {'id', 'quote', 'usage'})
                    or not isinstance(source['id'], str) or source['id'] not in catalog
                    or not isinstance(source['quote'], str) or not source['quote'].strip()
                    or source.get('usage', 'assertion') not in ('assertion', 'context')
                    or source['quote'] not in catalog[source['id']]['quote']):
                raise RuntimeError('逐字段复核basis的来源不可见或引语不属于该来源。')
            message_id = catalog[source['id']]['message_id']
            if source.get('usage', 'assertion') == 'assertion' and not any(citation['source']['message_id'] == message_id
                       and source['quote'] in citation['quote'] for citation in registered):
                missing.append(source['id'])
    if missing:
        failure = deepcopy(row)
        failure['citations_complete'] = False
        failure['reason'] = ('程序核对出处：' + '、'.join(dict.fromkeys(missing))
                             + ' 的所用原话未登记为本条引用；完整原文可见不等于已经引用。')
        return failure
    return None
