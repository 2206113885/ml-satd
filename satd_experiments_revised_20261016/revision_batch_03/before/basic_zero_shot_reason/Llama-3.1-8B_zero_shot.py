#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_satd_llama31_zero_shot.py
-----------------------------
零样本 · Meta-Llama-3.1-8B-Instruct · 官方 pipeline 对话模板
输出目录: results/run-YYYYmmdd-HHMMSS-模型名
保存: predictions.csv, confusion_matrix.csv, report.csv, settings.json
新增: predictions.csv 增加 reason 列（模型给出的简短理由）
"""

import re, os, gc, json, torch, transformers
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 0. 路径 ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # 如需改 GPU，请改这里
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/Llama-3.1-8B-Instruct"

MODEL_NAME = Path(MODEL_PATH).name
RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

PRINT_UNPARSED  = True
FALLBACK_ABBR   = "Data Conf"   # 解析失败回退缩写
MAX_NEW_TOKENS  = 512            # 需要放下缩写 + 简短理由（可按需调大）

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
ABBR_LOWER = {a.lower(): a for a in ABBR_SET}
ABBR_REGEX = re.compile(r"\b(" + "|".join(re.escape(a) for a in ABBR_SET) + r")\b", re.I)

# ---------- 2. Prompt 头（描述，无 few-shot） ----------
PROMPT_HEAD = r"""
There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):

MACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.

MODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.

MODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software (e.g., poorly-named temporary matrix variables).

DUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.

DUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.

WEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.

LAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.

HYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.

CODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency (library / subsystem).

GLUE CODE (GC): Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.

CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.

MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.

UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.

DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.

PLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.

DATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.

ABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes when a component is modified.

BOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.

MODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.

MODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.

PREDICTION QUALITY (PQ): Workflow changes may affect performance.

MACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.

PROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.
""".strip()

# 收紧输出：两行固定格式（Abbreviation + Reason）
PROMPT_TAIL = (
    "Please determine which type of ML Self-Admitted Technical Debt (SATD) the following Code Comment belongs to.\n"
    "Return your answer in EXACTLY two lines and nothing else:\n"
    f"Abbreviation: <ONE from: {', '.join(sorted(ABBR_SET))}>\n"
    "Reason: <<=12 words; short, specific, and grounded in the comment>\n"
).strip()

def build_prompt(comment: str) -> str:
    return (
        f"{PROMPT_HEAD}\n\n{PROMPT_TAIL}\n\n"
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n"
    )

# ---------- 3. 稳健解析 ----------
_ABBR_PREFIX = re.compile(r'^\s*(abbreviation|abbr)\s*[:：-]\s*', re.I)
_REASON_PREFIX = re.compile(r'^\s*(reason)\s*[:：-]\s*', re.I)

def parse_abbr_and_reason(full_text: str):
    """
    解析两行输出：
      第1行：Abbreviation: <abbr>
      第2行：Reason: <text>
    若缺失 Reason，则尝试从全文里找 'because|reason:' 后的内容。
    返回：(abbr, abbr_raw, reason)
    """
    lines = [l.strip() for l in full_text.strip().splitlines() if l.strip()]
    abbr_raw = ""
    reason = ""

    if lines:
        abbr_raw = _ABBR_PREFIX.sub("", lines[0]).strip()
    if len(lines) >= 2:
        reason = _REASON_PREFIX.sub("", lines[1]).strip()
    if not reason:
        m = re.search(r'(?:because|reason)\s*[:：-]?\s*(.+)$', full_text, flags=re.I)
        if m:
            reason = m.group(1).strip()
    if len(reason) > 500:
        reason = reason[:500].rstrip()

    # 复用原来的缩写抽取策略
    abbr = extract_abbr(abbr_raw if abbr_raw else (lines[0] if lines else ""))
    return abbr, abbr_raw, reason

def extract_abbr(raw: str) -> str:
    """
    旧版缩写抽取：优先从首行用正则定位；再做归一化 token 比对；最后尝试以全称首词启发。
    """
    first = re.split(r"[\r\n]", str(raw).strip(), maxsplit=1)[0].strip()
    first = re.sub(r"(^[*_`]+|[*_`]+$)", "", first)  # 去装饰
    m = ABBR_REGEX.search(first)
    if m:
        return ABBR_LOWER[m.group(1).lower()]
    token = re.sub(r"[^a-z]", "", first.lower())
    if token in ABBR_LOWER:
        return ABBR_LOWER[token]
    # [R03 原代码保留]
    # for full, ab in LABEL_ABBR.items():
    #     if full.split()[0].lower() in token:
    #         return ab
    # [R03 修订]
    full_name_matches = [
        ab for full, ab in LABEL_ABBR.items()
        if re.sub(r"[^a-z]", "", full.lower()) == token
    ]
    if len(full_name_matches) == 1:
        return full_name_matches[0]
    return "<UNK>"

print("[INFO] Loading model …")

# 4) 用我们设好的 model/tokenizer 构建官方 pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_PATH,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def text_gen(messages, MAX_NEW_TOKENS):
    return pipeline(
        messages,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )

# ---------- 5. classify（按官方 messages 走 pipeline） ----------
@torch.inference_mode()
def classify(comment: str):
    messages = [
        {"role": "user", "content": build_prompt(comment)},
    ]
    out = text_gen(messages, MAX_NEW_TOKENS)
    # out[0]["generated_text"] 是对话消息列表，最后一个是 assistant 的回复
    assistant_msg = out[0]["generated_text"][-1]
    raw_full = assistant_msg.get("content", "").strip()

    # 解析缩写与理由
    abbr, abbr_raw, reason = parse_abbr_and_reason(raw_full)

    # 保留首行用于调试/统计
    raw_first = abbr_raw if abbr_raw else (raw_full.split("\n", 1)[0].strip() if raw_full else "")
    return abbr, raw_first, reason, raw_full

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
        abbr = FALLBACK_ABBR  # 缩写失败仍回退；理由原样保存，便于排查
    rows.append({
        "comment": cmt,
        "real": ABBR2FULL.get(gold, gold),
        "pred": ABBR2FULL[abbr],
        "reason": reason,          # 新增：简短理由
        "raw_output": raw_first    # 保留首行输出（便于排错）
    })

pred_df = pd.DataFrame(rows)

# 终端打印
report_text = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, digits=3)
print("\n[INFO] Overall metrics:\n", report_text)

# ---------- 7. 保存结果 ----------
# 7.1 逐样本预测
pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)

# 7.2 混淆矩阵（行真列预测）
pd.DataFrame(
    confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
    index=LABELS, columns=LABELS
).to_csv(RUN_DIR/"confusion_matrix.csv")

# 7.3 classification_report 保存为 CSV，并追加 INFO 行
report_dict = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, output_dict=True)
rep_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index":"label"})
rep_df["info"] = ""
info_line = f"Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}"
rep_df.loc[len(rep_df)] = {"label":"[INFO]", "precision":"", "recall":"", "f1-score":"", "support":"", "info":info_line}
rep_df.to_csv(RUN_DIR/"report.csv", index=False)

# 7.4 保存设置
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
