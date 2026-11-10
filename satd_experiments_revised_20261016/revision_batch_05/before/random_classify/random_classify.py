#!/usr/bin/env python3
# [R22/R37 原代码保留] 历史入口整体归档；以下原文不参与执行。
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# satd_uniform_random_classifier.py
# ---------------------------------
# 完全随机：对每条样本在 23 个标签中“均匀随机”选择 1 个。
# 不加载模型、不设种子（每次结果不同）、无任何 fallback 到 Data Conf 的逻辑。
# 输出目录: results/run-YYYYmmdd-HHMMSS-random
# 保存: predictions.csv, confusion_matrix.csv, report.csv, settings.json
# """
# 
# import os, gc, json
# import random as pyrandom  # 脚本文件名不要叫 random.py
# from pathlib import Path
# from datetime import datetime
# 
# import pandas as pd
# from tqdm import tqdm
# from sklearn.metrics import classification_report, confusion_matrix
# 
# # ---------- 0. 路径 ----------
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 占位，不影响随机
# CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"
# 
# MODEL_NAME = "random"
# RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
# RUN_DIR.mkdir(parents=True, exist_ok=True)
# 
# # ---------- 1. 标签映射（与原保持一致） ----------
# LABEL_ABBR = {
#     "Machine Learning Knowledge":"MLKnow","Model Interpretability":"MLInter",
#     "Model Code Comprehension":"MCC","Duplicate Model Code":"DMC",
#     "Duplicate Feature Extraction Code":"DFEC","Weight Configuration":"Weight Conf",
#     "Layer Configuration":"Layer Conf","Hyper-parameter Configuration":"HP Conf",
#     "Code Dependency":"MLDepend","Glue Code":"GC","Custom Data Type":"CDT",
#     "Multiple Languages":"ML","Unnecessary Model Code":"UMC",
#     "Data Processing Configuration":"Data Conf","Plain Old Data Type":"PODT",
#     "Data Storage Configuration":"DS Conf","Abstraction":"Abs","Boundary Erosion":"BE",
#     "Model Code Modifiability":"MCM","Model Code Reusability":"MCR",
#     "Prediction Quality":"PQ","Machine Learning Reliability":"MLRelia","Prototype":"Proto",
# }
# ABBR2FULL  = {v: k for k, v in LABEL_ABBR.items()}
# LABELS     = list(LABEL_ABBR.keys())     # 用“全称”评估/混淆矩阵
# ABBR_LIST  = sorted(ABBR2FULL.keys())    # 随机抽样集合（缩写）
# 
# # ---------- 2. 读取数据 & 随机预测 ----------
# df = pd.read_csv(CSV_PATH)
# rows = []
# 
# print("[INFO] Classifying randomly …")
# for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
#                       total=len(df), ncols=80):
#     # 从 23 个缩写中均匀随机抽取一个
#     rand_abbr = pyrandom.choice(ABBR_LIST)
#     rows.append({
#         "comment": cmt,
#         "real": ABBR2FULL.get(gold, gold),   # gold 可能是缩写或全称，这里统一转全称
#         "pred": ABBR2FULL[rand_abbr],        # 预测同样转全称
#         "raw_output": rand_abbr              # 原始输出就记录缩写
#     })
# 
# pred_df = pd.DataFrame(rows)
# 
# # ---------- 3. 评估 ----------
# report_text = classification_report(
#     pred_df["real"], pred_df["pred"],
#     labels=LABELS, zero_division=0, digits=3
# )
# print("\n[INFO] Overall metrics:\n", report_text)
# 
# # ---------- 4. 保存结果 ----------
# # 4.1 逐样本预测
# pred_df.to_csv(RUN_DIR / "predictions.csv", index=False)
# 
# # 4.2 混淆矩阵（行真列预测，顺序与 LABELS 一致）
# pd.DataFrame(
#     confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
#     index=LABELS, columns=LABELS
# ).to_csv(RUN_DIR / "confusion_matrix.csv")
# 
# # 4.3 classification_report 保存为 CSV（不追加任何 fallback 信息）
# report_dict = classification_report(
#     pred_df["real"], pred_df["pred"],
#     labels=LABELS, zero_division=0, output_dict=True
# )
# rep_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
# rep_df.to_csv(RUN_DIR / "report.csv", index=False)
# 
# # 4.4 保存设置
# (Path(RUN_DIR / "settings.json")).write_text(json.dumps({
#     "csv": CSV_PATH,
#     "model": "UNIFORM_RANDOM_BASELINE",
#     "model_name": MODEL_NAME,
#     "run_dir": str(RUN_DIR),
#     "note": "No seed set; results vary each run. No fallback to Data Conf."
# }, indent=2))
# 
# print(f"\n[INFO] Files saved in {RUN_DIR}")
# 
# gc.collect()
# 
# 
# 
# 

# [R22/R37 修订] 显式多seed协议；历史单次run保留，不复写。
"""Uniform random 23-class baseline with per-seed predictions and uncertainty summary.

The empirical 2.5%-97.5% seed range describes run-to-run variation, not a
confidence interval for the population mean. Expected accuracy is exactly 1/23.
"""
import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auxiliary_protocol import (ABBR2FULL, ABBREVIATIONS, file_sha256, load_labeled,
                                new_output_dir, stable_sample_seed, versions, write_json)


def quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (index - low) * (ordered[high] - ordered[low])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    # [R37 修订] 与现有论文描述的10次随机基线重复一致；实际seed列表保存至新结果。
    # parser.add_argument('--seeds', nargs='+', type=int, default=list(range(30)),
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)),
                        help='Predeclared integer seeds; at least two distinct seeds')
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Supply at least two distinct predeclared seeds')
    rows = load_labeled(args.input_csv)
    output = new_output_dir(args.output_dir)
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    labels = [ABBR2FULL[a] for a in ABBREVIATIONS]
    results = []
    for seed in args.seeds:
        seed_dir = output / ('seed-' + str(seed))
        seed_dir.mkdir()
        records = []
        for row in rows:
            sample_seed = stable_sample_seed(seed, row['sample_id'])
            prediction = random.Random(sample_seed).choice(ABBREVIATIONS)
            records.append({**row, 'seed': seed, 'sample_seed': sample_seed,
                            'real': ABBR2FULL[row['gold_label']], 'pred': ABBR2FULL[prediction],
                            'predicted_label': prediction, 'raw_output': prediction})
        with (seed_dir / 'predictions.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        gold, pred = [r['real'] for r in records], [r['pred'] for r in records]
        report = classification_report(gold, pred, labels=labels, zero_division=0, output_dict=True)
        write_json(seed_dir / 'report.json', report)
        with (seed_dir / 'report.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['label', 'precision', 'recall', 'f1-score', 'support'])
            for label, values in report.items():
                if isinstance(values, dict):
                    writer.writerow([label] + [values.get(key, '') for key in ['precision', 'recall', 'f1-score', 'support']])
        matrix = confusion_matrix(gold, pred, labels=labels)
        with (seed_dir / 'confusion_matrix.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['real/pred'] + labels)
            for label, values in zip(labels, matrix.tolist()):
                writer.writerow([label] + values)
        result = {'seed': seed, 'accuracy': accuracy_score(gold, pred),
                  'macro_f1': f1_score(gold, pred, labels=labels, average='macro', zero_division=0),
                  'weighted_f1': f1_score(gold, pred, labels=labels, average='weighted', zero_division=0)}
        write_json(seed_dir / 'settings.json', {'seed': seed, 'input_sha256': file_sha256(args.input_csv),
                                              'sample_seed_policy': 'SHA256([seed, sample_id, 0])', **result})
        results.append(result)
    summary = {}
    for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
        values = [float(row[metric]) for row in results]
        summary[metric] = {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values),
                           'empirical_seed_q025': quantile(values, .025), 'empirical_seed_q975': quantile(values, .975)}
    write_json(output / 'summary.json', {'per_seed': results, 'summary': summary,
                                        'analytic_expected_accuracy': 1 / len(ABBREVIATIONS),
                                        'interval_interpretation': 'Empirical seed-run distribution; NOT a confidence interval of its mean'})
    write_json(output / 'settings.json', {'input_csv': str(args.input_csv.resolve()),
                                         'input_sha256': file_sha256(args.input_csv), 'seeds': args.seeds,
                                         'seed_policy': 'stable independent per-sample seed; input order does not change predictions',
                                         'labels': list(ABBREVIATIONS), 'fallback': None,
                                         'script_sha256': file_sha256(Path(__file__)),
                                         'helper_sha256': file_sha256(Path(__file__).resolve().parents[1] / 'auxiliary_protocol.py'),
                                         'versions': versions(['scikit-learn']),
                                         'historical_single_seed_runs': 'Preserved; old seed values cannot be recovered'})


if __name__ == '__main__':
    main()
