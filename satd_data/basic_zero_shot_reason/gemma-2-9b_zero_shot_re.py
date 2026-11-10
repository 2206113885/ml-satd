#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_satd_gemma2_zero_shot.py  (改版)
-----------------------------------
零样本 · gemma-2-9b-it · 纯对话 prompt
输出目录: results/run-YYYYmmdd-HHMMSS-模型名
保存: predictions.csv, confusion_matrix.csv, report.csv, settings.json
新增: predictions.csv 增加 reason 列（模型给出的简短理由）
"""

import re, os, gc, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_float32_matmul_precision('high')

# ---------- 0. 路径（仅两处需改） ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "2"         # GPU ID
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/gemma-2-9b-it"

# 目录名包含时间戳 + 模型名
MODEL_NAME = Path(MODEL_PATH).name
RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

PRINT_UNPARSED  = True
FALLBACK_ABBR   = "Data Conf"
MAX_NEW_TOKENS  = 512             # 贪婪解码，保持较快；理由要求简短（<=12词）

# ---------- 1. 标签映射 ----------
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

ABBR2FULL  = {v:k for k,v in LABEL_ABBR.items()}
LABELS     = list(LABEL_ABBR.keys())
ABBR_SET   = set(ABBR2FULL.keys())
ABBR_LOWER = {a.lower():a for a in ABBR_SET}

# -------- 2. Prompt 头（描述，无 few-shot） ----
PROMPT_HEAD = r"""
There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):

MACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.

MODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.

MODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software.

DUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.

DUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.

WEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.

LAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.

HYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.

CODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency.

GLUE CODE (GC): Supporting code written to interface with other code.

CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.

MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.

UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.

DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed by editing data directly or adding processing steps.

PLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.

DATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.

ABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes.

BOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance.

MODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.

MODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.

PREDICTION QUALITY (PQ): Workflow changes may affect performance.

MACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.

PROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.
""".strip()

# 要求两行固定格式输出，第二行给出简短理由（<=12词）
PROMPT_TAIL = f"""Please determine which type of ML Self-Admitted Technical Debt (SATD) the following Code Comment belongs to.
Return your answer in EXACTLY two lines and nothing else:
Abbreviation: <ONE from: {', '.join(sorted(ABBR_SET))}>
Reason: <<=12 words; short, specific, and grounded in the comment>
""".strip()

def build_prompt(comment: str) -> str:
    return (
        f"{PROMPT_HEAD}\n\n{PROMPT_TAIL}\n\n"
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n"
    )

# ---------- 3. 解析函数 ----------
_ABBR_PREFIX = re.compile(r'^\s*(abbreviation|abbr)\s*[:：-]\s*', re.I)
_REASON_PREFIX = re.compile(r'^\s*(reason)\s*[:：-]\s*', re.I)

def simple_extract(raw: str) -> str:
    """
    原有缩写解析：优先尝试“去非字母小写”的匹配，再尝试全称与去空格匹配。
    注意：这里保持与旧版兼容的行为，不做大幅度改动。
    """
    token = re.sub(r"[^a-z]", "", raw.lower())
    if token in ABBR_LOWER:
        return ABBR_LOWER[token]
    if raw in ABBR2FULL:              # 精确匹配缩写（区分大小写）
        return raw
    join = raw.replace(" ", "")
    if join.lower() in ABBR_LOWER:
        return ABBR_LOWER[join.lower()]
    return "<UNK>"

def parse_abbr_and_reason(full_text: str):
    """
    解析两行输出：
      第1行：Abbreviation: <abbr>
      第2行：Reason: <text>
    兜底：若找不到第2行，则从首行里尝试 'because|reason|:|-' 之后的文本作为理由。
    返回：(abbr, abbr_raw, reason_text)
    """
    # 取非空行
    lines = [l.strip() for l in full_text.strip().splitlines() if l.strip()]
    abbr_raw = ""
    reason = ""

    if lines:
        abbr_raw = _ABBR_PREFIX.sub("", lines[0]).strip()
    if len(lines) >= 2:
        reason = _REASON_PREFIX.sub("", lines[1]).strip()

    if not reason:
        # 尝试从整段文本里抽一个简短理由
        m = re.search(r'(?:because|reason)\s*[:：-]?\s*(.+)$', full_text, flags=re.I)
        if m:
            reason = m.group(1).strip()

    # 限宽一下（避免极端长文本写入 CSV）
    if len(reason) > 500:
        reason = reason[:500].rstrip()

    abbr = simple_extract(abbr_raw)
    return abbr, abbr_raw, reason

# ---------- 4. 载入模型 ----------
print("[INFO] Loading model …")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# ---------- 5. classify ----------
@torch.inference_mode()
def classify(comment: str):
    messages = [
        {"role": "user", "content": build_prompt(comment)},   # gemma 只要 user
    ]
    text = tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=4096).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    full = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()

    # 解析缩写与理由
    abbr, abbr_raw, reason = parse_abbr_and_reason(full)

    # 兼容旧逻辑：raw_first 仍用于调试/回退统计
    raw_first = abbr_raw if abbr_raw else (full.split("\n", 1)[0] if full else "")
    return abbr, raw_first, reason, full

# ---------- 6. 主流程 ----------
df = pd.read_csv(CSV_PATH)
rows, unk_cnt = [], 0
print("[INFO] Classifying …")
for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
                      total=len(df), ncols=80):
    abbr, raw_first, reason, raw_full = classify(cmt)
    if abbr == "<UNK>":
        unk_cnt += 1
        if PRINT_UNPARSED:
            with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
                f.write(raw_first + "\n")
        abbr = FALLBACK_ABBR
        # 如果缩写解析失败，理由也不太可信；但仍原样记录
    rows.append({
        "comment": cmt,
        "real": ABBR2FULL.get(gold, gold),
        "pred": ABBR2FULL[abbr],
        "reason": reason,          # 新增：模型给出的简短理由
        "raw_output": raw_first    # 仍保留首行原始输出，便于排查
    })

pred_df = pd.DataFrame(rows)

# 终端打印文本版
report_text = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, digits=3)
print("\n[INFO] Overall metrics:\n", report_text)

# ---------- 7. 保存结果 ----------
# 7.1 逐样本预测（包含 reason）
pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)

# 7.2 混淆矩阵（行真列预测，顺序与 LABELS 一致）
pd.DataFrame(
    confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
    index=LABELS, columns=LABELS
).to_csv(RUN_DIR/"confusion_matrix.csv")

# 7.3 把 classification_report 保存为 CSV，并追加 INFO 行
report_dict = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, output_dict=True)
rep_df = pd.DataFrame(report_dict).transpose()
rep_df.index.name = "label"
rep_df = rep_df.reset_index()
rep_df["info"] = ""  # 先占位一列

info_line = f"Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}"
rep_df.loc[len(rep_df)] = {
    "label": "[INFO]",
    "precision": "",
    "recall": "",
    "f1-score": "",
    "support": "",
    "info": info_line
}
rep_df.to_csv(RUN_DIR/"report.csv", index=False)

# 7.4 运行配置
(Path(RUN_DIR/"settings.json")).write_text(json.dumps({
    "csv": CSV_PATH,
    "model": MODEL_PATH,
    "model_name": MODEL_NAME,
    "run_dir": str(RUN_DIR),
    "max_new_tokens": MAX_NEW_TOKENS,
    "fallback_abbr": FALLBACK_ABBR,
    "output_format": "two-lines (Abbreviation / Reason <=12 words)"
}, indent=2))

print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}")

gc.collect(); torch.cuda.empty_cache()
