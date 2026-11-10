#!/usr/bin/env python3
"""Analyze paired ML-SATD pilot predictions without replacing parse failures.

Standard-library only. Gold labels are taken from the frozen cohort. A paired,
label-stratified bootstrap holds this deliberately selected cohort's class
counts fixed; its intervals do not establish population or cross-project
generalization. Human review forms contain neither gold nor model predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

LABEL_NAMES = {
    "Machine Learning Knowledge": "MLKnow", "Model Interpretability": "MLInter",
    "Model Code Comprehension": "MCC", "Duplicate Model Code": "DMC",
    "Duplicate Feature Extraction Code": "DFEC", "Weight Configuration": "Weight Conf",
    "Layer Configuration": "Layer Conf", "Hyper-parameter Configuration": "HP Conf",
    "Code Dependency": "MLDepend", "Machine Learning Dependency": "MLDepend",
    "Glue Code": "GC", "Custom Data Type": "CDT", "Multiple Languages": "ML",
    "Unnecessary Model Code": "UMC", "Data Processing Configuration": "Data Conf",
    "Plain Old Data Type": "PODT", "Data Storage Configuration": "DS Conf",
    "Abstraction": "Abs", "Boundary Erosion": "BE", "Model Code Modifiability": "MCM",
    "Model Code Reusability": "MCR", "Prediction Quality": "PQ",
    "Machine Learning Reliability": "MLRelia", "Prototype": "Proto",
}
LABELS = list(dict.fromkeys(LABEL_NAMES.values()))
ARMS = ("comment_only", "window", "function")
ERROR_LABEL = "<PARSE_ERROR>"
METRICS = ("accuracy", "weighted_precision", "weighted_recall", "weighted_f1",
           "macro_precision", "macro_recall", "macro_f1", "macro_f1_23")


def canonical_label(value):
    if value is None:
        return None
    value = str(value).strip()
    if value in LABELS:
        return value
    if value in LABEL_NAMES:
        return LABEL_NAMES[value]
    return None


def first(row, *keys, default=None):
    return next((row[k] for k in keys if row.get(k) is not None), default)


def row_id(row):
    value = first(row, "sample_id", "id", "source_id", "row_id")
    if value is None:
        raise ValueError("Every cohort/prediction row needs sample_id (or id).")
    return str(value)


def read_jsonl(path, allow_partial=False):
    if not path.exists():
        return [], []
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    rows, warnings = [], []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if allow_partial and i == len(lines) and not raw.endswith("\n"):
                warnings.append(f"Ignored incomplete final JSONL line {i} in {path.name}.")
                continue
            raise
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{i}: expected JSON object.")
        rows.append(row)
    return rows, warnings


def prediction(row):
    label = canonical_label(first(row, "predicted_label", "prediction", "pred_label", "pred"))
    # Never infer correctness from a raw completion; parsing is the runner's job.
    ok = row.get("parse_ok", label is not None)
    if isinstance(ok, str):
        ok = ok.strip().lower() in ("1", "true", "yes")
    return label if ok and label is not None else ERROR_LABEL


def scores(gold, pred):
    if len(gold) != len(pred):
        raise ValueError("Gold and prediction lengths differ.")
    if any(y not in LABELS for y in gold):
        raise ValueError("Unknown gold label; refusing to silently drop rows.")
    support, guessed, tp = Counter(gold), Counter(pred), Counter()
    for y, p in zip(gold, pred):
        if y == p:
            tp[y] += 1
    n = len(gold)
    per_label = []
    for label in LABELS:
        p = tp[label] / guessed[label] if guessed[label] else 0.0
        r = tp[label] / support[label] if support[label] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_label.append(dict(label=label, support=support[label], predicted=guessed[label],
                              true_positive=tp[label], precision=p, recall=r, f1=f))
    result = {"n": n, "accuracy": sum(tp.values()) / n if n else None,
              "parse_failures": guessed[ERROR_LABEL], "labels_with_support": sum(s > 0 for s in support.values())}
    observed_labels = set(gold)
    for average in ("weighted", "macro", "macro_23"):
        for metric in ("precision", "recall", "f1"):
            if average == "weighted":
                weights = [x["support"] / n if n else 0.0 for x in per_label]
            elif average == "macro":
                weights = [int(x["label"] in observed_labels) / len(observed_labels) if observed_labels else 0.0 for x in per_label]
            else:
                weights = [1 / len(LABELS)] * len(LABELS)
            key = f"{average}_{metric}" if average != "macro_23" else f"macro_{metric}_23"
            result[key] = sum(w * x[metric] for w, x in zip(weights, per_label)) if n else None
    return result, per_label


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    x = fraction * (len(values) - 1)
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def bootstrap(gold, predictions, repeats, seed):
    """Paired samples within each true class; same indices for every arm."""
    groups = defaultdict(list)
    for i, label in enumerate(gold):
        groups[label].append(i)
    draws = {a: {m: [] for m in METRICS} for a in predictions}
    diffs = {a: {m: [] for m in METRICS} for a in predictions if a != "comment_only"}
    rng = random.Random(seed)
    for _ in range(repeats if gold else 0):
        indices = [rng.choice(group) for group in groups.values() for _ in group]
        sampled_gold = [gold[i] for i in indices]
        metrics = {a: scores(sampled_gold, [ps[i] for i in indices])[0] for a, ps in predictions.items()}
        for arm in predictions:
            for metric in METRICS:
                draws[arm][metric].append(metrics[arm][metric])
                if arm != "comment_only":
                    diffs[arm][metric].append(metrics[arm][metric] - metrics["comment_only"][metric])
    interval = lambda xs: [quantile(xs, .025), quantile(xs, .975)]
    return ({a: {m: interval(xs) for m, xs in ms.items()} for a, ms in draws.items()},
            {a: {m: interval(xs) for m, xs in ms.items()} for a, ms in diffs.items()})


def exact_mcnemar(corrected, regressed):
    """Two-sided exact binomial McNemar; exploratory, without multiplicity correction."""
    n = corrected + regressed
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(corrected, regressed) + 1)) / 2**n)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def review_forms(cohort, output, seed):
    """Randomized anonymous forms. The key is a separate evaluation artifact."""
    key, form_manifest = [], {}
    for arm, form_name in zip(ARMS, ("A", "B", "C")):
        rng = random.Random(f"{seed}:{arm}")
        shuffled = list(cohort.items())
        rng.shuffle(shuffled)
        rows = []
        for sample_id, item in shuffled:
            review_id = hashlib.sha256(f"{seed}:{arm}:{sample_id}".encode()).hexdigest()[:12]
            comment = first(item, "comment", "comment_text", "Comment Text", default="")
            context = "" if arm == "comment_only" else first(item, f"{arm}_context", f"context_{arm}", arm, default="")
            rows.append(dict(review_id=review_id, comment=comment, code_context=context,
                             assigned_labels="", evidence_sufficient="", cited_evidence="",
                             confidence_1_to_5="", annotator_id="", notes=""))
            key.append(dict(review_id=review_id, sample_id=sample_id, form=form_name, arm=arm))
        name = f"blind_review_form_{form_name}.csv"
        write_csv(output / name, rows)
        form_manifest[form_name] = {"path": name, "n": len(rows)}
    write_csv(output / "blind_review_key_EVALUATION_ONLY.csv", key)
    instructions = (
        "# 独立人工复核表\n\n"
        "这些表是待人工填写的材料，尚未开展人工标注。表中没有原始标签或模型预测。\n\n"
        "请分别向至少两名标注者提供同一信息条件下的表格和统一23类定义；先独立填写，冻结答案后再讨论分歧。"
        "每条可选多个标签；证据不足时填 UNKNOWN，并填写 evidence_sufficient=no 和缺少的证据。"
        "请标注支持标签的评论或代码片段及1至5信心评分。\n\n"
        "避免同一标注者立即复核同一条的另一条件，可用独立标注组或足够的间隔并重新随机排序。"
        "若仅有两名标注者，不能把不同信息条件的分歧直接当作同条件一致性。\n\n"
        "blind_review_key_EVALUATION_ONLY.csv 仅供评估者合并结果；标注者不应接触原始cohort、预测、指标或映射表。"
        "原始标签只是待核对参照，人工修改必须单独保存，不能覆盖模型评估用金标。\n"
    )
    (output / "blind_review_instructions.md").write_text(instructions, encoding="utf-8")
    return form_manifest


def analyze(cohort_path, predictions_path, output, repeats=2000, seed=20261016,
            require_complete=False):
    cohort_rows, warnings = read_jsonl(cohort_path)
    if not cohort_rows:
        raise ValueError("Frozen cohort is empty or missing.")
    cohort = {}
    for row in cohort_rows:
        sid = row_id(row)
        if sid in cohort:
            raise ValueError(f"Duplicate cohort ID: {sid}")
        gold = canonical_label(first(row, "gold_label", "gold", "label", "ML TD Type"))
        if gold is None:
            raise ValueError(f"Unknown gold label for {sid}.")
        cohort[sid] = dict(row, gold_label=gold)
    records, extra_warnings = read_jsonl(predictions_path, allow_partial=True)
    warnings.extend(extra_warnings)
    by_arm = {a: {} for a in ARMS}
    duplicates = 0
    for row in records:
        sid, arm = row_id(row), row.get("arm")
        if sid not in cohort or arm not in ARMS:
            raise ValueError(f"Unknown sample/arm in predictions: {sid}/{arm}")
        row_gold = canonical_label(first(row, "gold_label", "gold"))
        if row_gold is not None and row_gold != cohort[sid]["gold_label"]:
            raise ValueError(f"Prediction gold disagrees with frozen cohort for {sid}.")
        if sid in by_arm[arm]:
            duplicates += 1
        by_arm[arm][sid] = row
    if duplicates:
        warnings.append(f"{duplicates} duplicate sample/arm rows: latest record used (explicit retries).")
    ids = [sid for sid in cohort if all(sid in by_arm[a] for a in ARMS)]
    complete = all(len(by_arm[a]) == len(cohort) for a in ARMS)
    if require_complete and not complete:
        raise ValueError("Pilot incomplete: each of the three arms must cover the entire frozen cohort.")
    gold = [cohort[sid]["gold_label"] for sid in ids]
    predictions = {a: [prediction(by_arm[a][sid]) for sid in ids] for a in ARMS}
    common_metrics, per_label_rows, all_available = {}, [], {}
    for arm in ARMS:
        common_metrics[arm], per_class = scores(gold, predictions[arm])
        per_label_rows.extend(dict(arm=arm, **row) for row in per_class)
        available = [sid for sid in cohort if sid in by_arm[arm]]
        all_available[arm] = scores([cohort[s]["gold_label"] for s in available],
                                    [prediction(by_arm[arm][s]) for s in available])[0]
        times = [first(by_arm[arm][s], "elapsed_seconds", "elapsed_s", "seconds") for s in ids]
        times = [float(t) for t in times if t is not None]
        common_metrics[arm]["mean_elapsed_seconds"] = statistics.mean(times) if times else None
        common_metrics[arm]["total_elapsed_seconds"] = sum(times) if times else None
        for token_key in ("prompt_tokens", "completion_tokens"):
            counts = [by_arm[arm][s].get(token_key) for s in ids]
            counts = [int(t) for t in counts if t is not None]
            common_metrics[arm][f"mean_{token_key}"] = statistics.mean(counts) if counts else None
    ci, difference_ci = bootstrap(gold, predictions, repeats, seed)
    comparisons, flip_rows = {}, []
    for arm in ARMS[1:]:
        corrected = regressed = changed = 0
        for i, sid in enumerate(ids):
            before, after, y = predictions["comment_only"][i], predictions[arm][i], gold[i]
            b, a = before == y, after == y
            corrected += int(not b and a)
            regressed += int(b and not a)
            changed += int(before != after)
            outcome = "corrected" if not b and a else "regressed" if b and not a else "both_correct" if b and a else "both_wrong"
            flip_rows.append(dict(sample_id=sid, comparison=f"{arm}-comment_only", gold_label=y,
                                  comment_only_prediction=before, context_prediction=after,
                                  changed=before != after, outcome=outcome,
                                  comment=first(cohort[sid], "comment", "comment_text", "Comment Text", default="")))
        differences = {m: (common_metrics[arm][m] - common_metrics["comment_only"][m]) if ids else None for m in METRICS}
        comparisons[arm] = dict(n=len(ids), corrected=corrected, regressed=regressed,
                                changed_predictions=changed, metric_differences=differences,
                                difference_ci_95=difference_ci[arm],
                                mcnemar_exact_p=exact_mcnemar(corrected, regressed),
                                p_value_note="Exploratory, no adjustment for multiple comparisons.")
    result = dict(cohort_n=len(cohort), complete=complete, common_paired_n=len(ids), labels=LABELS,
                  cohort_support=dict(Counter(x["gold_label"] for x in cohort.values())),
                  macro_policy="macro_* averages labels with gold support in common paired cohort; macro_*_23 fixes all 23 original labels (zero for absent labels).",
                  primary_common_cohort=common_metrics, primary_ci_95=ci,
                  all_available_per_arm=all_available, comparisons=comparisons,
                  bootstrap=dict(repeats=repeats, seed=seed, method="paired within-gold-label percentile bootstrap"),
                  parse_policy="Invalid/missing parsed label counts as incorrect; no majority-class fallback.",
                  scope="Conditional on selected stratified cohort. Not a population estimate, independent replication, or cross-project validation.",
                  warnings=warnings)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "metrics.csv", [dict(arm=a, **m) for a, m in common_metrics.items()])
    write_csv(output / "per_label_metrics.csv", per_label_rows)
    write_csv(output / "paired_flips.csv", flip_rows, fields=["sample_id", "comparison", "gold_label", "comment_only_prediction", "context_prediction", "changed", "outcome", "comment"])
    review_forms(cohort, output, seed)
    def fmt(value):
        return "NA" if value is None else f"{value:.4f}"
    lines = ["# ML-SATD 上下文初步实验", "", f"冻结样本：{len(cohort)}；三臂共同完成：{len(ids)}；全部完成：{complete}。", "",
             "| 输入条件 | n | Accuracy | Weighted F1 | Macro F1（有支持类） | Macro F1（23类） | 解析失败 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for a, m in common_metrics.items():
        lines.append(f"| {a} | {m['n']} | {fmt(m['accuracy'])} | {fmt(m['weighted_f1'])} | {fmt(m['macro_f1'])} | {fmt(m['macro_f1_23'])} | {m['parse_failures']} |")
    lines += ["", "所有主要指标均使用三臂共同完成的同一组样本；解析失败按错误计入，未用多数类替代。Macro F1同时报告本次有金标支持的类别及固定23类两种口径。", ""]
    for arm, values in comparisons.items():
        interval = values["difference_ci_95"]["weighted_f1"]
        lines.append(f"{arm} 相对 comment_only：Weighted F1 差值 {fmt(values['metric_differences']['weighted_f1'])}，95%配对分层bootstrap区间 [{fmt(interval[0])}, {fmt(interval[1])}]；纠正 {values['corrected']} 条，退步 {values['regressed']} 条。")
        lines.append("")
    lines += ["这是所选分层样本上的探索性结果。重采样固定每类样本数，极少样本类别的区间会低估对新样本的不确定性；未调整仓库内相关性，也未涵盖随机解码的种子间差异。不能外推到393条总体或新项目。多项比较p值未校正。", "",
              "人工复核表为待填写材料；当前结果使用原始标签，未声称人工确认上下文充分性或标签正确性。", "",
              "完整置信区间见 metrics.json；逐类结果见 per_label_metrics.csv；变化样本见 paired_flips.csv。"]
    if warnings:
        lines += ["", "数据读取提示："] + [f"- {w}" for w in warnings]
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--cohort", type=Path, default=base / "data" / "cohort.jsonl")
    parser.add_argument("--predictions", type=Path, default=base / "predictions.jsonl")
    parser.add_argument("--output", type=Path, default=base / "analysis")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    result = analyze(args.cohort, args.predictions, args.output, args.bootstrap, args.seed, args.require_complete)
    print(json.dumps({k: result[k] for k in ("cohort_n", "common_paired_n", "complete", "primary_common_cohort")}, indent=2))


if __name__ == "__main__":
    main()
