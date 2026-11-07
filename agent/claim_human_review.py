"""人工确认模式只完成结构与出处审计，始终保存为待人工确认。"""
from evidence_grounding import digest


HUMAN_REVIEW_MARKER = '## 人工确认报告模式 v1'
HUMAN_REVIEW_NOTICE = ('> 待人工确认：这份报告是模型生成的草稿。程序已检查引用出处与结构，'
                       '未确认解释是否成立。请点击引用回查原文，核对人物、动作、先后和结论强度。')


def human_review_record(compiled):
    return {'schema_version': 'claim-human-review-v1', 'status': 'pending',
            'claims_sha256': digest(compiled['claims']), 'input_sha256': compiled['input_sha256']}


def audit_human_review(stored, prompt):
    requested = HUMAN_REVIEW_MARKER in prompt
    record = stored.get('human_review')
    if requested != (record is not None):
        raise ValueError('人工确认记录与实际报告输入不一致。')
    if not requested:
        return False
    if record != human_review_record(stored) or 'semantic_review' in stored:
        raise ValueError('人工确认报告只能标记为待确认，且须绑定当前结论和输入。')
    return True
