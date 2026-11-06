"""离线比对固定命题的人工标注和输入覆盖；不调用模型，不充当通用语义裁判。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ASSESSMENTS = {'supported', 'contradicted', 'insufficient'}
ROLES = {'support', 'counter', 'limitation', 'irrelevant'}


def evaluate_cases(cases, predictions):
    """命题原文、角色和所给材料分别验收；缺失案例计失败，不能挑选好样本。"""
    case_ids = [case['id'] for case in cases]
    if not cases or len(set(case_ids)) != len(case_ids):
        raise ValueError('案例不能为空或包含重复编号。')
    by_id = {}
    for prediction in predictions:
        cid = prediction['case_id']
        if cid in by_id or cid not in case_ids:
            raise ValueError('预测包含重复或未知案例编号。')
        by_id[cid] = prediction
    results = []
    for case in cases:
        positions = {row['position'] for row in case['messages']}
        required = set(case['required_positions'])
        expected = case['expected']
        if (not required <= positions or expected['assessment'] not in ASSESSMENTS
                or any(str(n) not in {str(p) for p in positions} or role not in ROLES
                       for n, role in expected['roles'].items())):
            raise ValueError('人工标注含未知位置、角色或判断。')
        prediction = by_id.get(case['id'])
        failures, missing = [], sorted(required)
        if prediction is None:
            failures.append('missing_prediction')
        else:
            if prediction.get('proposition') != case['proposition']:
                failures.append('proposition')
            if prediction.get('assessment') != expected['assessment']:
                failures.append('assessment')
            if prediction.get('roles') != expected['roles']:
                failures.append('roles')
            available = prediction.get('available_positions')
            if (not isinstance(available, list) or any(type(n) is not int for n in available)
                    or not set(available) <= positions):
                failures.append('invalid_material_positions')
            else:
                missing = sorted(required - set(available))
                if missing:
                    failures.append('material_coverage')
        results.append({'case_id': case['id'], 'passed': not failures,
                        'failed_checks': failures, 'missing_positions': missing})
    return {'schema_version': 'claim-semantic-evaluation-v1', 'cases': len(cases),
            'passed': sum(row['passed'] for row in results), 'results': results,
            'scope': 'fixed_proposition_human_annotations_and_recorded_input_coverage',
            'general_semantic_correctness': 'not_evaluated'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    predictions = json.loads(args.predictions.read_text(encoding='utf-8'))
    if (cases['schema_version'] != 'claim-semantic-cases-v1'
            or predictions['schema_version'] != 'claim-semantic-predictions-v1'):
        raise ValueError('不支持的案例或预测版本。')
    result = evaluate_cases(cases['cases'], predictions['predictions'])
    output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(output, encoding='utf-8', newline='\n')
    else:
        print(output, end='')
    return 0 if result['passed'] == result['cases'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
