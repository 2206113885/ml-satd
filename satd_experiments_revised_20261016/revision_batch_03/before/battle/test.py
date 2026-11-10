#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, os, gc, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import pipeline

# -------- 0. 基础路径（仅此处改数据文件路径；推理参数一概不改） --------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CSV_PATH        = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"
MODEL_PATH      = "/data/yangzhifei/pro/ML_SATD/models/gpt-oss-20b"   # 或 "openai/gpt-oss-20b"

# Qwen / Gemma 的预测文件（来自你之前各自运行的 predictions.csv）
QWEN_PRED_PATH  = "/data/yangzhifei/pro/ML_SATD/basic_zero_shot_CoT/results/run-20250821-064248-Qwen3-14B-38.4/predictions.csv"    # ← 改成你的文件
GEMMA_PRED_PATH = "/data/yangzhifei/pro/ML_SATD/basic_zero_shot_CoT/results/run-20250822-011558-gpt-oss-20b-35.8/predictions.csv"  # ← 改成你的文件

# 目录名包含时间戳 + 模型名
MODEL_NAME = Path(MODEL_PATH).name
RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}-arbiter"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# -------- 1. 其它参数（保持不变） --------
PRINT_UNPARSED    = True
FALLBACK_ABBR     = "Data Conf"
MAX_NEW_TOKENS    = 32768
MAX_INPUT_TOKENS  = 4096   # 未显式使用，保留

# ----------------- 标签映射（全称 -> 缩写） -----------------
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
LABELS      = list(LABEL_ABBR.keys())
ABBR_SET    = set(LABEL_ABBR.values())
ABBR2FULL   = {abbr: full for full, abbr in LABEL_ABBR.items()}
FULL2ABBR   = LABEL_ABBR

# —— 规范化工具（沿用你原逻辑） —— #
def _canon(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

FULL_BY_CANON = {_canon(full): full for full in LABELS}
FULL_BY_ABBR_CANON = {_canon(abbr): full for full, abbr in LABEL_ABBR.items()}

def normalize_to_full(label: str):
    if not isinstance(label, str):
        return None
    s = label.strip()
    if s in LABELS:
        return s
    if s in ABBR2FULL:
        return ABBR2FULL[s]
    cz = _canon(s)
    if cz in FULL_BY_CANON:
        return FULL_BY_CANON[cz]
    if cz in FULL_BY_ABBR_CANON:
        return FULL_BY_ABBR_CANON[cz]
    return None

# ============ 2. 23 类定义（保留） ============
PROMPT_HEAD = r"""
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

# —— 裁判提示：让 gpt-oss 在两候选中二选一（不改推理参数，只改输入内容） —— #
def build_referee_prompt(comment: str,
                         qwen_abbr: str, gemma_abbr: str,
                         qwen_full: str, gemma_full: str,
                         qwen_raw: str = None, gemma_raw: str = None) -> str:
    """
    给 gpt-oss-20b 的用户提示：
    - 仍提供 23 类定义（利于依据定义判别）。
    - 指明“只在两个候选缩写中二选一”，并要求输出单行缩写。
    - 可选附带两模型的原始输出，帮助裁判理解它们各自的依据（不强制）。
    """
    parts = [
        PROMPT_HEAD,
        "You are an arbiter. Two other models (Qwen and Gemma) have predicted categories for the code comment below.",
        "Your job is to choose the better-matching abbreviation ONLY from the two candidates, based on the comment and the definitions above.",
        f"Candidates (abbreviation → full):",
        f" - Qwen:  {qwen_abbr} → {qwen_full}",
        f" - Gemma: {gemma_abbr} → {gemma_full}",
    ]
    if qwen_raw:
        parts.append(f"Qwen raw output: {qwen_raw}")
    if gemma_raw:
        parts.append(f"Gemma raw output: {gemma_raw}")

    parts.append(
        "\nReturn exactly ONE abbreviation on a single line, with no extra text. "
        f"Choose strictly from: {qwen_abbr}, {gemma_abbr}.\n"
        "If both look equally plausible, prefer the one more directly evidenced by the comment."
    )
    parts.append(
        f"\nCode Comment:\n```text\n{comment}\n```\n\n#Chosen Abbreviation:"
    )
    return "\n\n".join(parts)

# —— 从文本中提取缩写（保持你原有的鲁棒解析） —— #
def extract_abbr_from_text(text: str) -> str:
    if not text:
        return "<UNK>"
    cz = _canon(text)
    if cz in {_canon(a): a for a in ABBR_SET}:
        for a in ABBR_SET:
            if _canon(a) == cz:
                return a
    for abbr in ABBR_SET:
        pat = r'\b' + re.sub(r'\s+', r'\\s*', re.escape(abbr)) + r'\b'
        if re.search(pat, text, flags=re.IGNORECASE):
            return abbr
    for full, abbr in FULL2ABBR.items():
        pat = r'\b' + re.sub(r'\s+', r'\\s*', re.escape(full)) + r'\b'
        if re.search(pat, text, flags=re.IGNORECASE):
            return abbr
    return "<UNK>"

# -------- 3. 加载 gpt-oss-20b（保持原样，不改参数） --------
print("[INFO] Loading model …")
pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)

# -------- 4. 分类函数（让 gpt-oss 在二者中裁判） --------
@torch.inference_mode()
def choose_between(comment: str,
                   qwen_full: str, gemma_full: str,
                   qwen_raw: str = None, gemma_raw: str = None):
    # 把全称映成缩写；若遇到异常/缺失，用 FALLBACK_ABBR 顶上（但仍尽量二选一）
    q_abbr = FULL2ABBR.get(qwen_full, FALLBACK_ABBR)
    g_abbr = FULL2ABBR.get(gemma_full, FALLBACK_ABBR)

    # 若两者缩写完全一样，直接返回即可，无需调用模型
    if q_abbr == g_abbr and q_abbr in ABBR_SET:
        return q_abbr, f"(both agree) {q_abbr}"

    messages = [
        {"role": "system", "content":
         "You are a strict classifier arbiter. Reasoning: low. "
         "Output ONLY the abbreviation, one line, and it must be one of the two candidates."},
        {"role": "user", "content": build_referee_prompt(
            comment, q_abbr, g_abbr,
            ABBR2FULL[q_abbr], ABBR2FULL[g_abbr],
            qwen_raw=qwen_raw, gemma_raw=gemma_raw
        )},
    ]

    outputs = pipe(messages, max_new_tokens=MAX_NEW_TOKENS)
    answer  = outputs[0]["generated_text"][-1]['content']
    raw_final = answer.split('assistantfinal')[-1]  # 保持你原先的 final 段截断
    picked = extract_abbr_from_text(raw_final)

    # 强约束：若不在{q_abbr, g_abbr}内，则默认选 Qwen 的（也可以改成 Gemma 或更复杂规则）
    if picked not in {q_abbr, g_abbr}:
        picked = q_abbr

    return picked, raw_final

# -------- 5. 读入数据集与两模型预测，并以 comment 为键对齐 --------
def _read_preds(path):
    """读取 predictions.csv；返回 {comment -> (full_label, raw_output)}"""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    # 兼容不同字段命名：优先 pred(全称)，其次 fine_abbr / fine / pred_abbr
    has_pred_full = "pred" in df.columns
    has_raw = "raw_output" in df.columns
    has_cmt = "comment" in df.columns
    if not has_cmt or (not has_pred_full and "fine_abbr" not in df.columns and "pred_abbr" not in df.columns):
        # 结构意外，直接返回空，让主流程做兜底
        return {}
    out = {}
    for _, r in df.iterrows():
        cmt = str(r["comment"]) if has_cmt else None
        if not cmt:
            continue
        if has_pred_full:
            full = str(r["pred"])
        else:
            # 如果是缩写列，映回全称
            ab = str(r["fine_abbr"] if "fine_abbr" in df.columns else r.get("pred_abbr", ""))
            full = ABBR2FULL.get(ab, None)
            if full is None:
                continue
        raw = str(r["raw_output"]) if has_raw and not pd.isna(r["raw_output"]) else None
        out[cmt] = (full, raw)
    return out

qwen_map  = _read_preds(QWEN_PRED_PATH)
gemma_map = _read_preds(GEMMA_PRED_PATH)

# -------- 6. 主流程：对每条样本，拿 Qwen/Gemma 的结果给 gpt-oss 二选一 --------
df = pd.read_csv(CSV_PATH)

rows = []
unk_cnt = 0
gold_unmapped = 0
missing_side = 0   # 统计有一侧缺失的样本

print("[INFO] Arbitrating between Qwen & Gemma …")
for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
                      total=len(df), ncols=80):
    real_full = normalize_to_full(str(gold))
    if real_full is None:
        gold_unmapped += 1
        continue

    q_full, q_raw = qwen_map.get(str(cmt), (None, None))
    g_full, g_raw = gemma_map.get(str(cmt), (None, None))

    # 如果任意一侧缺失：直接采用存在的一侧；两侧都缺则回退到 FALLBACK_ABBR
    if q_full is None and g_full is None:
        pred_abbr = FALLBACK_ABBR
        raw_final = "(no qwen/gemma predictions) fallback to " + pred_abbr
        missing_side += 1
    elif q_full is None:
        pred_abbr = FULL2ABBR.get(g_full, FALLBACK_ABBR)
        raw_final = "(only gemma available) " + pred_abbr
        missing_side += 1
    elif g_full is None:
        pred_abbr = FULL2ABBR.get(q_full, FALLBACK_ABBR)
        raw_final = "(only qwen available) " + pred_abbr
        missing_side += 1
    else:
        pred_abbr, raw_final = choose_between(str(cmt), q_full, g_full, q_raw, g_raw)

    if pred_abbr not in ABBR_SET:
        unk_cnt += 1
        if PRINT_UNPARSED:
            with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
                f.write(str(raw_final) + "\n")
        pred_abbr = FALLBACK_ABBR

    rows.append({
        "comment": cmt,
        "real": real_full,                     # 真实全称
        "qwen_pred": q_full,                   # Qwen 全称（可能缺）
        "gemma_pred": g_full,                  # Gemma 全称（可能缺）
        "chosen_abbr": pred_abbr,              # 仲裁出的缩写
        "pred": ABBR2FULL.get(pred_abbr, pred_abbr),  # 转成全称供评估
        "raw_output": raw_final
    })

pred_df = pd.DataFrame(rows)

# 若全部样本都被跳过
if pred_df.empty:
    print("\n[WARN] No valid samples after label normalization.")
else:
    report_text = classification_report(
        pred_df["real"], pred_df["pred"],
        labels=LABELS, zero_division=0, digits=3
    )
    print("\n[INFO] Overall metrics (after arbitration):\n", report_text)

    # -------- 7. 保存结果 --------
    pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)

    cm = confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(RUN_DIR/"confusion_matrix.csv")

    report_dict = classification_report(
        pred_df["real"], pred_df["pred"],
        labels=LABELS, zero_division=0, output_dict=True
    )
    rep_df = pd.DataFrame(report_dict).transpose()
    rep_df.index.name = "label"
    rep_df = rep_df.reset_index()
    rep_df["info"] = ""

    info_lines = [
        f"Unrecognized outputs mapped to '{FALLBACK_ABBR}': {unk_cnt}",
        f"Gold labels unmapped (skipped): {gold_unmapped}",
        f"Samples with missing Qwen/Gemma side: {missing_side}",
        f"Total evaluated samples: {len(pred_df)} / original {len(df)}"
    ]
    rep_df.loc[len(rep_df)] = {
        "label": "[INFO]",
        "precision": "",
        "recall": "",
        "f1-score": "",
        "support": "",
        "info": " | ".join(info_lines)
    }
    rep_df.to_csv(RUN_DIR/"report.csv", index=False)

print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}")
print(f"[INFO] Gold labels unmapped (skipped): {gold_unmapped}")

gc.collect(); torch.cuda.empty_cache()
