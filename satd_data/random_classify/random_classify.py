#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
satd_uniform_random_classifier.py
---------------------------------
完全随机：对每条样本在 23 个标签中“均匀随机”选择 1 个。
不加载模型、不设种子（每次结果不同）、无任何 fallback 到 Data Conf 的逻辑。
输出目录: results/run-YYYYmmdd-HHMMSS-random
保存: predictions.csv, confusion_matrix.csv, report.csv, settings.json
"""

import os, gc, json
import random as pyrandom  # 脚本文件名不要叫 random.py
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 0. 路径 ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 占位，不影响随机
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"

MODEL_NAME = "random"
RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1. 标签映射（与原保持一致） ----------
LABEL_ABBR = {
    "Machine Learning Knowledge":"MLKnow","Model Interpretability":"MLInter",
    "Model Code Comprehension":"MCC","Duplicate Model Code":"DMC",
    "Duplicate Feature Extraction Code":"DFEC","Weight Configuration":"Weight Conf",
    "Layer Configuration":"Layer Conf","Hyper-parameter Configuration":"HP Conf",
    "Code Dependency":"MLDepend","Glue Code":"GC","Custom Data Type":"CDT",
    "Multiple Languages":"ML","Unnecessary Model Code":"UMC",
    "Data Processing Configuration":"Data Conf","Plain Old Data Type":"PODT",
    "Data Storage Configuration":"DS Conf","Abstraction":"Abs","Boundary Erosion":"BE",
    "Model Code Modifiability":"MCM","Model Code Reusability":"MCR",
    "Prediction Quality":"PQ","Machine Learning Reliability":"MLRelia","Prototype":"Proto",
}
ABBR2FULL  = {v: k for k, v in LABEL_ABBR.items()}
LABELS     = list(LABEL_ABBR.keys())     # 用“全称”评估/混淆矩阵
ABBR_LIST  = sorted(ABBR2FULL.keys())    # 随机抽样集合（缩写）

# ---------- 2. 读取数据 & 随机预测 ----------
df = pd.read_csv(CSV_PATH)
rows = []

print("[INFO] Classifying randomly …")
for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
                      total=len(df), ncols=80):
    # 从 23 个缩写中均匀随机抽取一个
    rand_abbr = pyrandom.choice(ABBR_LIST)
    rows.append({
        "comment": cmt,
        "real": ABBR2FULL.get(gold, gold),   # gold 可能是缩写或全称，这里统一转全称
        "pred": ABBR2FULL[rand_abbr],        # 预测同样转全称
        "raw_output": rand_abbr              # 原始输出就记录缩写
    })

pred_df = pd.DataFrame(rows)

# ---------- 3. 评估 ----------
report_text = classification_report(
    pred_df["real"], pred_df["pred"],
    labels=LABELS, zero_division=0, digits=3
)
print("\n[INFO] Overall metrics:\n", report_text)

# ---------- 4. 保存结果 ----------
# 4.1 逐样本预测
pred_df.to_csv(RUN_DIR / "predictions.csv", index=False)

# 4.2 混淆矩阵（行真列预测，顺序与 LABELS 一致）
pd.DataFrame(
    confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
    index=LABELS, columns=LABELS
).to_csv(RUN_DIR / "confusion_matrix.csv")

# 4.3 classification_report 保存为 CSV（不追加任何 fallback 信息）
report_dict = classification_report(
    pred_df["real"], pred_df["pred"],
    labels=LABELS, zero_division=0, output_dict=True
)
rep_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
rep_df.to_csv(RUN_DIR / "report.csv", index=False)

# 4.4 保存设置
(Path(RUN_DIR / "settings.json")).write_text(json.dumps({
    "csv": CSV_PATH,
    "model": "UNIFORM_RANDOM_BASELINE",
    "model_name": MODEL_NAME,
    "run_dir": str(RUN_DIR),
    "note": "No seed set; results vary each run. No fallback to Data Conf."
}, indent=2))

print(f"\n[INFO] Files saved in {RUN_DIR}")

gc.collect()




