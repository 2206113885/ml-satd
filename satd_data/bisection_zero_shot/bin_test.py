#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-step YES/NO SATD classifier
  1) iterate 23 labels → YES/NO
  2) if 1 YES — ask for reasoning; if >1 YES — ask model to choose best + reason
"""

import os, re, gc, json
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------- 0. 基础路径 -----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/Qwen3-8B"   # ← 可换任何聊天模型
RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS_YESNO  = 10     # “YES” / “NO” 不需要太多 token
MAX_NEW_TOKENS_REASON = 200    # 解释部分

# --------------- 1. 标签定义 -----------------
LABEL_INFO = {
    "MLKnow" : "Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.",
    "MLInter": "Machine-learning models are a black box, causing poor understanding of model functionality.",
    "MCC"    : "Model code carries extra legibility concerns not present in traditional software.",
    "DMC"    : "Code duplication frequently occurs in model code.",
    "DFEC"   : "Code duplication frequently occurs in feature-extraction code.",
    "Weight Conf": "Editing code that involves the weights of a model, or configuring a model's weights directly.",
    "Layer Conf" : "Editing code that deals with model layers, or configuring layers themselves.",
    "HP Conf": "Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.",
    "MLDepend": "A needed change is blocked by an external dependency (library / subsystem).",
    "GC"     : "Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.",
    "CDT"    : "Using data types provided by general-purpose packages can cause extensive inter-operability burden.",
    "ML"     : "Components written in other languages may introduce difficulties in ML development.",
    "UMC"    : "Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.",
    "Data Conf": "Configuring how data is processed, by editing the data directly or adding new processing steps.",
    "PODT"   : "Using raw data types in ML causes confusion when interpreting processes.",
    "DS Conf": "Configuring how data is represented internally or stored externally.",
    "Abs"    : "Lack of abstractions in ML systems causes cascading changes when a component is modified.",
    "BE"     : "Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.",
    "MCM"    : "Model code should enable easy maintenance and future modifications.",
    "MCR"    : "Model code should be generalised for reuse in varying situations.",
    "PQ"     : "Workflow changes may affect performance.",
    "MLRelia": "Model functionality depends on data quality; robustness measures should be in place.",
    "Proto"  : "Deploying small-scale prototypes into full systems can be dangerous.",
}
LABEL_ORDER = list(LABEL_INFO.keys())   # 固定顺序遍历

FULL2ABBR = {v: k for k, v in LABEL_INFO.items()}  # 备用
ABBR2FULL = {abbr: desc.split(":")[0] if ":" in desc else desc for abbr, desc in LABEL_INFO.items()}

# --------------- 2. Prompt 模板 -----------------
YESNO_SYS = "You are a classifier for Machine Learning Self-Admitted Technical Debt (SATD) in code comments. Read the given comment and determine whether it belongs to the given category. Output only \"Yes\" if it belongs, or \"No\" if it does not."
YESNO_USER_TMPL = (
    ""
    "Code Comment:\n\"\"\"\n{comment}\n\"\"\"\n\n"
    "Label: {abbr}\n"
    "Definition: {definition}\n\n"
    "Does the code comment belong to this label? Answer with YES or NO."
)

EXPLAIN_SYS = (
    "You are an expert classifier for Machine Learning SATD. "
    "Return the best fitting label's abbreviation and a short reason (one sentence). "
    "Format: <ABBR> - <reason>."
)

EXPLAIN_USER_TMPL_SINGLE = (
    "Comment:\n\"\"\"\n{comment}\n\"\"\"\n\n"
    "Label: {abbr}\nDefinition: {definition}\n\n"
    "Only give the abbreviation and reason."
)

EXPLAIN_USER_TMPL_MULTI = (
    "Comment:\n\"\"\"\n{comment}\n\"\"\"\n\n"
    "Candidate labels:\n{cands}\n"
    "Pick the SINGLE best abbreviation and give a short reason."
)

# --------------- 3. 处理多标签选择与回退机制 -----------------
def chat(messages, max_new_tokens, tokenizer, model):
    """Simple wrapper for chat-style generation."""
    # 拼接消息
    text = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        text += f"{role}: {content}\n"
    
    
    # print(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,

    )
    ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
    return ans.splitlines()[0].strip()  # 取首行

@torch.inference_mode()
def yes_or_no(comment, abbr, tokenizer, model):
    messages = [
        {"role": "system", "content": YESNO_SYS},
        {"role": "user",
         "content": YESNO_USER_TMPL.format(
             comment=comment,
             abbr=abbr,
             definition=LABEL_INFO[abbr]
         )},
    ]
    ans = chat(messages, MAX_NEW_TOKENS_YESNO, tokenizer, model).upper()
    return "YES" in ans

@torch.inference_mode()
def explain(comment, chosen_abbr, tokenizer, model, candidates=None):
    """Return 'ABBR - reason'"""
    if candidates is None:   # 单一候选
        user = EXPLAIN_USER_TMPL_SINGLE.format(
            comment=comment,
            abbr=chosen_abbr,
            definition=LABEL_INFO[chosen_abbr]
        )
    else:                    # 多候选
        cand_str = "\n".join(f"- {abbr}: {LABEL_INFO[abbr]}" for abbr in candidates)
        user = EXPLAIN_USER_TMPL_MULTI.format(comment=comment, cands=cand_str)
    messages = [
        {"role": "system", "content": EXPLAIN_SYS},
        {"role": "user",   "content": user},
    ]
    ans = chat(messages, MAX_NEW_TOKENS_REASON, tokenizer, model)
    # 解析缩写
    m = re.match(r"\s*([A-Za-z]+)\s*[-–]\s*(.*)", ans)
    if m:
        abbr, reason = m.group(1).strip(), m.group(2).strip()
    else:
        # 宽容 fallback
        parts = ans.split(None, 1)
        abbr = parts[0]
        reason = parts[1] if len(parts) > 1 else ""
    return abbr, reason

# --------------- 4. 主流程 -----------------
df = pd.read_csv(CSV_PATH)
rows = []
print("[INFO] Two-step classification …")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")

for comment, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
                          total=len(df), ncols=80):
    # 第一轮：YES/NO
    positives = [abbr for abbr in LABEL_ORDER if yes_or_no(comment, abbr, tokenizer, model)]

    if len(positives) == 0:       # 无匹配
        chosen = "Data Conf"
        reason = "No strong match; defaulting to Data Conf."
    elif len(positives) == 1:     # 唯一 YES
        chosen, reason = explain(comment, positives[0], tokenizer, model)
    else:                         # 多个 YES — 让模型二选一
        chosen, reason = explain(comment, positives[0], tokenizer, model, candidates=positives)

    rows.append({
        "comment": comment,
        "real": gold,
        "pred": chosen,
        "reason": reason,
        "candidates": ";".join(positives),
    })

# --------------- 5. 评估 & 保存 -----------------
pred_df = pd.DataFrame(rows)
LABELS = list({row["real"] for row in rows} | set(LABEL_ORDER))
report = classification_report(pred_df["real"], pred_df["pred"],
                               labels=LABELS, zero_division=0, digits=3)
print("\n[INFO] Overall metrics:\n", report)

pred_df.to_csv(RUN_DIR / "predictions_yesno.csv", index=False)
pd.DataFrame(confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
             index=LABELS, columns=LABELS).to_csv(RUN_DIR / "confusion_matrix.csv")
(RUN_DIR / "report.txt").write_text(report)
print(f"[INFO] Results saved in {RUN_DIR}")

gc.collect(); torch.cuda.empty_cache()






