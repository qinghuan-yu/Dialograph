"""固定本体上的人格恢复指标；仅依赖标准库。"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict


def planned_rows(cases, completed, recovery_ids=()):
    """失败与额外恢复案例保留在原流程分母中，预测按缺失处理。"""
    lookup = {row['id']: row for row in completed}
    result = []
    for case in cases:
        row = {**case, **lookup.get(case['id'], {})}
        empty = {d: None for d in case['gold']}
        row.setdefault('direct', dict(empty))
        row['available'] = case['id'] in lookup
        row['recovery'] = case['id'] in recovery_ids
        if not row['available'] or row['recovery']:
            row['pred'] = dict(empty)
        result.append(row)
    return result


def validate_extraction(data: dict, dimensions: list[str], source: str) -> dict:
    if not isinstance(data, dict) or set(data) != set(dimensions):
        raise ValueError('提取结果必须恰好覆盖本体维度')
    for key, item in data.items():
        if not isinstance(item, dict):
            raise ValueError(f'{key}: 非对象')
        value = item.get('value')
        if value is not None and (type(value) is not int or value not in (0, 1, 2)):
            raise ValueError(f'{key}: 非法等级')
        confidence = item.get('confidence')
        if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
            raise ValueError(f'{key}: 非法置信度')
        quote = item.get('quote', '')
        if not isinstance(quote, str) or (value is not None and (not quote.strip() or quote not in source)):
            raise ValueError(f'{key}: 引文不在原文')
    return data


def _ratio(n, d):
    return n / d if d else None


def _agreement(pairs):
    n = len(pairs)
    if not n:
        return None, None, None
    left, right = Counter(g for g, _ in pairs), Counter(p for _, p in pairs)
    po = sum(g == p for g, p in pairs) / n
    pe = sum(left[k] * right[k] for k in range(3)) / n ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else None
    observed = sum((g - p) ** 2 / 4 for g, p in pairs) / n
    expected = sum(left[g] * right[p] * (g - p) ** 2 / 4 for g in range(3) for p in range(3)) / n ** 2
    weighted = 1 - observed / expected if expected else None
    # ICC(2,1)：双向随机效应、绝对一致性；两名评分者为真值与预测。
    icc = None
    if n > 1:
        rowmeans = [(g + p) / 2 for g, p in pairs]
        colmeans = [sum(g for g, _ in pairs) / n, sum(p for _, p in pairs) / n]
        grand = sum(rowmeans) / n
        msr = 2 * sum((x - grand) ** 2 for x in rowmeans) / (n - 1)
        msc = n * sum((x - grand) ** 2 for x in colmeans)
        mse = sum((x - rowmeans[i] - colmeans[j] + grand) ** 2
                  for i, pair in enumerate(pairs) for j, x in enumerate(pair)) / (n - 1)
        denominator = msr + mse + 2 * (msc - mse) / n
        icc = (msr - mse) / denominator if abs(denominator) > 1e-12 else None
    return kappa, weighted, icc


def score(golds: list[dict], predictions: list[dict]) -> dict:
    if len(golds) != len(predictions) or any(set(g) != set(p) for g, p in zip(golds, predictions)):
        raise ValueError('样本数和维度必须对齐')
    tp = fp = fn = covered = known = unknown = assertions = 0
    abs_error = penalized = hamming = jaccard = 0
    sample_f1 = []
    labels = defaultdict(lambda: [0, 0, 0])
    pairs = []
    for gold, pred in zip(golds, predictions):
        stp = sfp = sfn = 0
        for dim, g in gold.items():
            p = pred[dim]
            if any(v is not None and (type(v) is not int or v not in (0, 1, 2)) for v in (g, p)):
                raise ValueError('等级只能是 0/1/2/null')
            if g is None:
                unknown += 1
                assertions += p is not None
                continue
            known += 1
            if p is None:
                fn += 1
                sfn += 1
                labels[(dim, g)][2] += 1
                penalized += 1
                hamming += 1
                continue
            covered += 1
            pairs.append((g, p))
            error = abs(g - p) / 2
            abs_error += error
            penalized += error
            if g == p:
                tp += 1
                stp += 1
                labels[(dim, g)][0] += 1
            else:
                fp += 1
                fn += 1
                sfp += 1
                sfn += 1
                labels[(dim, g)][2] += 1
                labels[(dim, p)][1] += 1
                hamming += 2
        if stp + sfp + sfn:
            sample_f1.append(2 * stp / (2 * stp + sfp + sfn))
            jaccard += stp / (stp + sfp + sfn)
    kappa, weighted, icc = _agreement(pairs)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    coverage = _ratio(covered, known)
    mae_penalized = _ratio(penalized, known)
    label_f1 = [2 * a / (2 * a + b + c) for a, b, c in labels.values() if 2 * a + b + c]
    return dict(tp=tp, fp=fp, fn=fn, known=known, covered=covered,
                precision=_ratio(tp, tp + fp), recall=_ratio(tp, known), micro_f1=f1,
                sample_macro_f1=_ratio(sum(sample_f1), len(sample_f1)),
                label_macro_f1=_ratio(sum(label_f1), len(label_f1)),
                coverage=coverage, mismatch_rate=_ratio(fp, covered),
                unanswerable_assertion_rate=_ratio(assertions, unknown),
                jaccard=_ratio(jaccard, len(sample_f1)), hamming_loss=_ratio(hamming, known * 3),
                mae_covered=_ratio(abs_error, covered), mae_penalized=mae_penalized,
                cohen_kappa=kappa, weighted_kappa=weighted, icc_2_1=icc,
                coupling=(f1 + 1 - mae_penalized) / 2 if known else None)


def bootstrap(rows: list[dict], repeats=2000, seed=20261017) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row['profile']].append(row)
    keys = sorted(groups)
    if not keys:
        return {'clusters': 0}
    rng, values = random.Random(seed), defaultdict(list)
    for _ in range(repeats):
        sampled = [row for key in rng.choices(keys, k=len(keys)) for row in groups[key]]
        metrics = score([r['gold'] for r in sampled], [r['pred'] for r in sampled])
        for key in ('micro_f1', 'coverage', 'mae_penalized', 'coupling', 'weighted_kappa', 'icc_2_1'):
            if metrics[key] is not None and math.isfinite(metrics[key]):
                values[key].append(metrics[key])
    result = {'clusters': len(keys), 'repeats': repeats, 'seed': seed}
    for key, items in values.items():
        items.sort()
        result[key] = [items[int(.025 * (len(items) - 1))], items[int(.975 * (len(items) - 1))]]
    return result
