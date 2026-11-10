#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_satd_gptoss20b_two_stage.py
--------------------------------
模型：/data/yangzhifei/pro/ML_SATD/models/gpt-oss-20b
方法：Two-stage（Coarse 9 → Fine 23）
  - Stage-1：输出 9 粗类缩写（AWR/RDB/.../SCL）
  - Stage-2：在对应粗类的细类候选中输出唯一一个 23 缩写之一
解析：兼容 "**AWR**" / "Abbreviation: AWR" / 全称等格式；保留 assistantfinal 之后的文本
输出：results/run-YYYYmmdd-HHMMSS-模型名 / predictions.csv, confusion_matrix_fine.csv,
      confusion_matrix_coarse.csv, report_fine.txt, report_coarse.txt, settings.json
"""

import re, os, gc, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import pipeline

# ========== 0. 基础路径（仅此处改） ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 号
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"  # ← 修改
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/gpt-oss-20b"                                     # ← 修改（或 "openai/gpt-oss-20b"）

# 目录名包含时间戳 + 模型名
MODEL_NAME = Path(MODEL_PATH).name
RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ========== 1. 其它参数 ==========
PRINT_UNPARSED = True
MAX_NEW_TOKENS = 4096

# ========== 2. 细粒度标签（23 类） ==========
FINE_LABEL_ABBR = {
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
FINE_ABBR2FULL  = {v:k for k,v in FINE_LABEL_ABBR.items()}
FINE_LABELS     = list(FINE_LABEL_ABBR.keys())
FINE_ABBR_SET   = set(FINE_ABBR2FULL.keys())

# ========== 3. 粗粒度（9 类）与细↔粗映射 ==========
COARSE_LABELS = ["AWR","RDB","DCE","CFO","CDD","DTD","MDL","PRF","SCL"]
COARSE_ABBR_SET = set(COARSE_LABELS)

# 细(缩写) → 粗
FINE2COARSE_ABBR = {
    "MLKnow":"AWR","MLInter":"AWR",
    "MCC":"RDB",
    "DMC":"DCE","DFEC":"DCE",
    "Weight Conf":"CFO","Layer Conf":"CFO","HP Conf":"CFO",
    "MLDepend":"CDD","GC":"CDD","CDT":"CDD","ML":"CDD","UMC":"CDD",
    "Data Conf":"DTD","PODT":"DTD","DS Conf":"DTD",
    "Abs":"MDL","BE":"MDL","MCM":"MDL","MCR":"MDL",
    "PQ":"PRF","MLRelia":"PRF",
    "Proto":"SCL",
}

# 粗 → 细(缩写) 候选集合
COARSE2FINE_ABBR = {
    "AWR":  ["MLKnow","MLInter"],
    "RDB":  ["MCC"],
    "DCE":  ["DMC","DFEC"],
    "CFO":  ["Weight Conf","Layer Conf","HP Conf"],
    "CDD":  ["MLDepend","GC","CDT","ML","UMC"],
    "DTD":  ["Data Conf","PODT","DS Conf"],
    "MDL":  ["Abs","BE","MCM","MCR"],
    "PRF":  ["PQ","MLRelia"],
    "SCL":  ["Proto"],
}

# 细(全称) → 粗（用于评估映射）
FINE2COARSE_FULL = {
    "Machine Learning Knowledge":"AWR", "Model Interpretability":"AWR",
    "Model Code Comprehension":"RDB",
    "Duplicate Model Code":"DCE", "Duplicate Feature Extraction Code":"DCE",
    "Weight Configuration":"CFO", "Layer Configuration":"CFO", "Hyper-parameter Configuration":"CFO",
    "Code Dependency":"CDD", "Glue Code":"CDD", "Custom Data Type":"CDD", "Multiple Languages":"CDD", "Unnecessary Model Code":"CDD",
    "Data Processing Configuration":"DTD", "Plain Old Data Type":"DTD", "Data Storage Configuration":"DTD",
    "Abstraction":"MDL", "Boundary Erosion":"MDL", "Model Code Modifiability":"MDL", "Model Code Reusability":"MDL",
    "Prediction Quality":"PRF", "Machine Learning Reliability":"PRF",
    "Prototype":"SCL",
}

# 各粗类下细类的回退（细类解析失败时使用）
FALLBACK_FINE_BY_COARSE = {
    "AWR":"MLKnow", "RDB":"MCC", "DCE":"DMC", "CFO":"HP Conf", "CDD":"MLDepend",
    "DTD":"Data Conf", "MDL":"MCM", "PRF":"PQ", "SCL":"Proto",
}

# ========== 4. Prompt ==========
COARSE_PROMPT_HEAD = r"""
There are 9 coarse-grained types of ML SATD:

Awareness (AWR): Lack of developer understanding or certainty about ML algorithms, APIs, data, or model behavior. Often arises due to “black box” nature of ML models.
Example Symptoms:Doubts on correctness, API usage uncertainty, unclear design decisions, unknown error causes, lack of domain knowledge.

Readability (RDB):Model code has poor legibility or unclear logic, especially in ML contexts (e.g., poorly named variables, dense math code).
Example Symptoms:Refactoring for clarity, simplifying complex statements, improving naming.

Duplicate Code Elimination (DCE): Identical or near-identical ML model or preprocessing logic appears in multiple places.
Example Symptoms:Repeated feature extraction, repeated model evaluation loops.

Configurable Options (CFO): Missing or poorly implemented configuration points for weights, network layers, or model parameters.
Example Symptoms:Hard-coded parameters, unimplemented config flags, missing flexibility for experiments.

Code Dependency (CDD): Code depends on other code or libraries in a way that blocks maintenance, requires workarounds, or is outdated.
Example Symptoms:Waiting for library updates, integration hacks, dependency-induced bugs, deprecated model code.

Data Dependency (DTD): Issues in how data is processed, represented, or stored that hinder maintainability or correctness.
Example Symptoms:Missing preprocessing steps, poor choice of data formats, unoptimized storage.

Modularity (MDL): Poor modular boundaries or design make the system harder to evolve, reuse, or maintain.
Example Symptoms:Missing abstractions, test parameters used in production, model-specific code that should be generalized.

Performance (PRF): Performance or robustness is insufficient, requiring evaluation, optimization, or reliability improvements.
Example Symptoms:Model accuracy concerns, robustness checks, performance comparisons.

Scalability (SCL): Prototype-level code is used in production without scaling optimizations.
Example Symptoms:Inefficient loops, naive algorithms, lack of batching/parallelism.
""".strip()

def build_prompt_coarse(comment: str) -> str:
    tail = (
        "Please classify the following Code Comment into exactly ONE of these coarse abbreviations:\n"
        + ", ".join(COARSE_LABELS) +
        "\nReturn the abbreviation only (e.g., AWR)."
    )
    return f"{COARSE_PROMPT_HEAD}\n\n{tail}\n\nCode Comment:\n\"\"\"\n{comment}\n\"\"\"\n\nAnswer:"

FINE_PROMPT_HEAD = r"""
There are 23 types of Machine Learning (ML) Self-Admitted Technical Debt (SATD):

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

def build_prompt_fine(comment: str, coarse_abbr: str) -> str:
    allowed = COARSE2FINE_ABBR.get(coarse_abbr, sorted(FINE_ABBR_SET))
    tail = (
        "Please determine the fine-grained ML SATD type. You MUST answer with only ONE abbreviation from:\n"
        + ", ".join(allowed) + "\nReturn the abbreviation only (e.g., HP Conf)."
    )
    return f"{FINE_PROMPT_HEAD}\n\n{tail}\n\nCode Comment:\n\"\"\"\n{comment}\n\"\"\"\n\nAnswer:"

# ========== 5. 解析函数（健壮版） ==========
COARSE_REGEX = re.compile(r'\b(AWR|RDB|DCE|CFO|CDD|DTD|MDL|PRF|SCL)\b', re.I)
ABBR_SET = FINE_ABBR_SET
ABBR_LOWER = {a.lower(): a for a in ABBR_SET}
ABBR_REGEX = re.compile(r"\b(" + "|".join(re.escape(a) for a in ABBR_SET) + r")\b", re.I)

def extract_coarse(raw: str) -> str:
    """从任意首行文本里提取粗类缩写（容忍 '**AWR**' / 'Abbreviation: AWR' 等）。"""
    if not isinstance(raw, str):
        return "<UNK>"
    s = raw.splitlines()[0].strip()

    m = COARSE_REGEX.search(s)
    if m:
        return m.group(1).upper()

    m = re.search(r'\*\*\s*(AWR|RDB|DCE|CFO|CDD|DTD|MDL|PRF|SCL)\s*\*\*', s, flags=re.I)
    if m:
        return m.group(1).upper()

    letters = re.sub(r'[^A-Za-z]', '', s).lower()
    for abbr in COARSE_LABELS:
        if abbr.lower() in letters:
            return abbr
    return "<UNK>"

def extract_fine(raw: str, allowed_abbr_list) -> str:
    """从 raw 中提取细类缩写，仅允许 allowed_abbr_list 内的候选。兼容空格差异与 markdown。"""
    if not isinstance(raw, str):
        return "<UNK>"
    s = raw.splitlines()[0].strip()

    # 允许 “HP\s*Conf” 这种空白差异
    pat_choices = [re.escape(x).replace(r'\ ', r'\s*') for x in allowed_abbr_list]
    pat = r'\b(' + '|'.join(pat_choices) + r')\b'
    m = re.search(pat, s, flags=re.I)
    if m:
        hit = re.sub(r'\s+', ' ', m.group(1)).strip().lower()
        for a in allowed_abbr_list:
            if re.sub(r'\s+', ' ', a).strip().lower() == hit:
                return a

    # **HP Conf** 之类
    m = re.search(r'\*\*\s*(' + '|'.join(pat_choices) + r')\s*\*\*', s, flags=re.I)
    if m:
        hit = re.sub(r'\s+', ' ', m.group(1)).strip().lower()
        for a in allowed_abbr_list:
            if re.sub(r'\s+', ' ', a).strip().lower() == hit:
                return a

    # 兜底：letters-only 包含匹配
    letters = re.sub(r'[^A-Za-z]', '', s).lower()
    for a in allowed_abbr_list:
        if re.sub(r'[^A-Za-z]', '', a).lower() in letters:
            return a
    return "<UNK>"

# ========== 6. 加载模型（沿用你之前的 pipeline 方式） ==========
print("[INFO] Loading model …")

pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)



@torch.inference_mode()
def generate_first_line(messages, max_new_tokens: int) -> str:
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
    )
    text = outputs[0]["generated_text"][-1]['content']
    # print(text)
    raw_answer = text.split('assistantfinal')[-1]
    return raw_answer

# ========== 7. 两阶段推理 ==========
@torch.inference_mode()
def classify_two_stage(comment: str):
    # Stage-1: Coarse
    coarse_prompt = build_prompt_coarse(comment)
    messages_coarse = [
        {"role": "system", "content": "Return exactly one coarse abbreviation (AWR/RDB/DCE/CFO/CDD/DTD/MDL/PRF/SCL)."},
        {"role": "user",   "content": coarse_prompt},
    ]
    coarse_raw = generate_first_line(messages_coarse, MAX_NEW_TOKENS)
    coarse_abbr = extract_coarse(coarse_raw)

    # Stage-2: Fine（限制在该粗类的候选集合）
    allowed = COARSE2FINE_ABBR.get(coarse_abbr, sorted(FINE_ABBR_SET))
    fine_prompt = build_prompt_fine(comment, coarse_abbr)
    messages_fine = [
        {"role": "system", "content": "Return exactly one fine abbreviation from the allowed candidates."},
        {"role": "user",   "content": fine_prompt},
    ]
    fine_raw = generate_first_line(messages_fine, MAX_NEW_TOKENS)
    fine_abbr = extract_fine(fine_raw, allowed)

    if fine_abbr == "<UNK>":
        fine_abbr = FALLBACK_FINE_BY_COARSE.get(coarse_abbr, "Data Conf")
    print(fine_abbr)
    return coarse_abbr, coarse_raw, fine_abbr, fine_raw

# ========== 8. 主流程 ==========
df = pd.read_csv(CSV_PATH)
rows = []
unk_coarse_cnt = 0
unk_fine_cnt   = 0

print("[INFO] Classifying (two-stage, gpt-oss-20b) …")
for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
                      total=len(df), ncols=80):
    coarse_abbr, coarse_raw, fine_abbr, fine_raw = classify_two_stage(str(cmt))

    if coarse_abbr not in COARSE_ABBR_SET:
        unk_coarse_cnt += 1
    if fine_abbr not in FINE_ABBR_SET:
        unk_fine_cnt += 1

    rows.append({
        "comment": cmt,
        "real": FINE_ABBR2FULL.get(gold, gold),                 # 细类全称（真实）
        "pred": FINE_ABBR2FULL.get(fine_abbr, fine_abbr),       # 细类全称（预测）
        "coarse_pred": coarse_abbr,                              # 粗类缩写（预测）
        "coarse_raw_output": coarse_raw,                         # 粗类原始输出
        "fine_abbr": fine_abbr,                                  # 细类缩写（预测）
        "raw_output": fine_raw,                                  # 细类原始输出
    })

pred_df = pd.DataFrame(rows)

# ========== 9. 评估 ==========
# 细（23）
fine_report_text = classification_report(
    pred_df["real"], pred_df["pred"],
    labels=FINE_LABELS, zero_division=0, digits=3
)
print("\n[INFO] Fine-grained metrics (23):\n", fine_report_text)

# 粗（9）
pred_df["real_coarse"] = pred_df["real"].map(FINE2COARSE_FULL)
pred_df["pred_coarse"] = pred_df["pred"].map(FINE2COARSE_FULL)
coarse_report_text = classification_report(
    pred_df["real_coarse"], pred_df["pred_coarse"],
    labels=COARSE_LABELS, zero_division=0, digits=3
)
print("\n[INFO] Coarse-grained metrics (9):\n", coarse_report_text)

# ========== 10. 保存 ==========
pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)

pd.DataFrame(
    confusion_matrix(pred_df["real"], pred_df["pred"], labels=FINE_LABELS),
    index=FINE_LABELS, columns=FINE_LABELS
).to_csv(RUN_DIR/"confusion_matrix_fine.csv")

pd.DataFrame(
    confusion_matrix(pred_df["real_coarse"], pred_df["pred_coarse"], labels=COARSE_LABELS),
    index=COARSE_LABELS, columns=COARSE_LABELS
).to_csv(RUN_DIR/"confusion_matrix_coarse.csv")

(Path(RUN_DIR/"report_fine.txt")).write_text(fine_report_text, encoding="utf-8")
(Path(RUN_DIR/"report_coarse.txt")).write_text(coarse_report_text, encoding="utf-8")

(Path(RUN_DIR/"settings.json")).write_text(json.dumps({
    "csv": CSV_PATH,
    "model": MODEL_PATH,
    "model_name": MODEL_NAME,
    "run_dir": str(RUN_DIR),
    "max_new_tokens": MAX_NEW_TOKENS,
    "two_stage": True,
    "unknown_coarse": unk_coarse_cnt,
    "unknown_fine": unk_fine_cnt
}, indent=2))

print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Unknown coarse predictions: {unk_coarse_cnt}")
print(f"[INFO] Unknown fine predictions:   {unk_fine_cnt}")

gc.collect(); torch.cuda.empty_cache()
